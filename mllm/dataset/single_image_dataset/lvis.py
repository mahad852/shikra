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
        extended_question = "Please only give coordinates of the <expr> and not of other objects. Give coordinates of all the <expr> found.".replace(EXPR_PLACEHOLDER, name)
        question = question + " " + extended_question
        
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

        total_precision = 0
        total_success = 0
        all_ious = []

        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)

            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}")
                continue
            if extract_pred is None:
                failed += 1
                logger.warning(f"failed to extract ans for pred: {pred}")
                extract_pred = [[0, 0, 0, 0]]
                
            selected = [False] * len(extract_target)
            true_positives = 0
            
            with torch.no_grad():
                targets = torch.tensor(extract_target)
                preds = torch.tensor(extract_pred)

                ious = box_iou(preds, targets)

                for p in range(len(preds)):
                    chosen_index = -1
                    max_iou = 0
                    for t in range(len(targets)):
                        if ious[p][t].item() > 0.5 and not selected[t] and ious[p][t].item() > max_iou:
                            max_iou = ious[p][t].item()
                            chosen_index = t
                    if chosen_index != -1:
                        selected[chosen_index] = True
                        true_positives += 1
                        all_ious.append(ious[p][chosen_index].item())
            
            total_precision += 1.0 * true_positives / len(targets)
            total_success += 1

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'precision': total_precision/total_success,
            'target_failed': target_failed,
            'failed': failed,
            'iou': sum(all_ious)/len(all_ious),
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            
            if len(list_of_boxes) == 0:
                return None
            
            boxes = []
            
            for b in list_of_boxes:
                if len(b) == 0:
                    return None
                box = b[0]
                if len(box) != 4:
                    return None
                boxes.append(box)
            return boxes
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
