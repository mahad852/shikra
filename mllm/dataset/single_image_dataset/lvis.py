import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import torch
from torchvision.ops import box_iou

from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)

from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class LVISDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        name = item['category_name']

        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, name)
        boxes_placeholder_string = ' '.join([BOXES_PLACEHOLDER] * len(item['bboxes']))
        ret = {
            'image': image,
            'target': {
                'boxes': item['bboxes'],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {boxes_placeholder_string} .',
                    'boxes_seq': [[i] for i in range(len(item['bboxes']))],
                }
            ]
        }
        return ret


@METRICS.register_module()
class LVISComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0


        pred_boxes, target_boxes = [], []
        for pred, target in zip(preds, targets):
            print("PREDS and TARGETS:")
            print("Preds:")
            extract_pred = self.extract_ans(pred)
            print("targets:")
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}")
                continue
            if extract_pred is None:
                failed += 1
                logger.warning(f"failed to extract ans for pred: {pred}")
                extract_pred = [0, 0, 0, 0]
            target_boxes.append(extract_target)
            pred_boxes.append(extract_pred)

        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            # normalized box value is too small, so that the area is 0.
            ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            print("extracted box list:", list_of_boxes)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
