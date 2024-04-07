import json
import re
import logging
import sys

from ..root import DATASETS, IMAGE_PLACEHOLDER, BOXES_PLACEHOLDER, QUESTION_PLACEHOLDER, METRICS
from ..utils.flickr30k_entities_utils import PHRASE_ST_PLACEHOLDER, PHRASE_ED_PLACEHOLDER
from ..utils import MInstrDataset, BaseComputeMetrics

from typing import Dict, Any, Sequence

from ..process_function import (
    BoxFormatter,
)

REFID_PAT = re.compile(r'(\s\((?:(?:\d+(?:,\d+)*)|-)\)\s?)')
ANS_EXTRACT_PAT = re.compile(r'(?:(?:(?:(?:(?:So t)|(?:T)|(?:t))he answer is)|(?:Answer:)) (.+))')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

@DATASETS.register_module()
class GQADataset(MInstrDataset):
    def __init__(
            self,
            *args,
            scene_graph_file,
            scene_graph_index,
            version,
            question_box_prob=0.5,
            **kwargs
    ):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.scene_graph_file = scene_graph_file
        self.scene_graph_index = scene_graph_index
        self.version = version
        self.question_box_prob = question_box_prob
        qtype, atype = version.split('-')
        assert qtype in ['q', 'qb', 'qbp']
        assert atype in ['a', 'c', 'bc', 's', 'bs', 'l', 'bl']
        self.qtype = qtype
        self.atype = atype

        # assert bool(scene_graph_file) == bool(scene_graph_index)
        if scene_graph_file is not None: #and scene_graph_index is not None:
            self.scene_graph = json.load(open(scene_graph_file, 'r'))
            # self.scene_index = json.load(open(scene_graph_index, 'r', encoding='utf8'))
        else:
            self.scene_graph = None
            self.scene_index = None

    def get_raw_item(self, index):
        question = json.loads(self.data[index])
        if self.scene_graph is None:
            return question, None
        scene = self.scene_graph[question['imageId']]
        return question, scene

    def __getitem__(self, index):
        question, scene = self.get_raw_item(index)
        img_path = f"{question['imageId']}.jpg"
        image = self.get_image(img_path)

        # answer
        if self.atype == 'bc':
            boxes = question['cot']['boxes']
            answer = question['cot']['value'].replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
            answer_boxes_seq = question['cot']['seq']
        elif self.atype == 'c':
            boxes = []
            answer = question['cot']['value'].replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, "")
            answer_boxes_seq = []
        elif self.atype == 'bs':
            boxes, bss, answer_boxes_seq = get_bss_example(question, scene)
            answer = f"{bss}. The answer is {question['answer']}."
        elif self.atype == 's':
            boxes = []
            ss = REFID_PAT.sub('', question['semanticStr'])
            answer = f"{ss}. The answer is {question['answer']}."
            answer_boxes_seq = []
        elif self.atype == 'bl':
            boxes, answer, answer_boxes_seq = get_bl_example(question, scene)
        elif self.atype == 'l':
            boxes = []
            _, answer, _ = get_bl_example(question, scene)
            answer = answer.replace(BOXES_PLACEHOLDER, "")
            answer_boxes_seq = []
        elif self.atype == 'a':
            boxes = []
            answer = f"The answer is {question['answer']}."
            answer_boxes_seq = []
        else:
            assert False

        # question
        if self.qtype == 'q':
            boxes, query, query_boxes_seq = prepare_query_dummy(boxes, question, scene)
        elif self.qtype == 'qb':
            boxes, query, query_boxes_seq = prepare_query_box(boxes, question, scene)
        elif self.qtype == 'qbp':
            if self.rng.uniform() > self.question_box_prob:
                boxes, query, query_boxes_seq = prepare_query_dummy(boxes, question, scene)
            else:
                boxes, query, query_boxes_seq = prepare_query_box(boxes, question, scene)
        else:
            assert False

        final_query = self.get_template().replace(QUESTION_PLACEHOLDER, query)

        ret = {
            'image': image,
            'target': {'boxes': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': final_query,
                    'boxes_seq': query_boxes_seq,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                    'boxes_seq': answer_boxes_seq,
                }
            ]
        }
        return ret


def prepare_query_dummy(boxes_list, q, scene):
    return boxes_list, q['question'], []


def prepare_query_box(boxes_list, q, scene):
    def get_boxes_idx(box):
        if box in boxes_list:
            return boxes_list.index(box)
        else:
            boxes_list.append(box)
            return len(boxes_list) - 1

    def add_boxes_by_rids(rids):
        def get_box_xyxy(obj):
            x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
            return x, y, x + w, y + h

        boxes_idx = []
        for rid in rids:
            ref = scene['objects'][rid]
            ref_box = list(get_box_xyxy(ref))
            boxes_idx.append(get_boxes_idx(ref_box))
        return boxes_idx

    sent = list(q['question'].split())
    query_boxes_seq = []
    for span, rids_str in q['annotations']['question'].items():
        span = tuple(map(int, span.split(':')))
        if len(span) == 1:
            span = [span[0], span[0] + 1]
        sent[span[1] - 1] = f"{sent[span[1] - 1]}{BOXES_PLACEHOLDER}"
        boxes_idx = add_boxes_by_rids(rids_str.split(','))
        query_boxes_seq.append(boxes_idx)
    sent_converted = " ".join(sent).strip()
    return boxes_list, sent_converted, query_boxes_seq


def add_boxes_by_rids(boxes_list, rids, scene):
    def get_boxes_idx(boxes_list, box):
        if box in boxes_list:
            return boxes_list.index(box)
        else:
            boxes_list.append(box)
            return len(boxes_list) - 1

    def get_box_xyxy(obj):
        x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        return x, y, x + w, y + h

    boxes_idx = []
    for rid in rids:
        ref = scene['objects'][rid]
        ref_box = list(get_box_xyxy(ref))
        boxes_idx.append(get_boxes_idx(boxes_list, ref_box))
    return boxes_idx


def get_bss_example(question, scene):
    def format_refids(item):
        item = item.strip()[1:-1]
        return item.split(',')

    s = question['semanticStr']
    print(REFID_PAT.findall(s))
    formats = []
    boxes = []
    seqs = []

    for item in REFID_PAT.findall(s):
        if '-' in item:
            formats.append('')
        else:
            formats.append('<boxes>')
            refids = format_refids(item)
            idx = add_boxes_by_rids(boxes, refids, scene)
            seqs.append(idx)
    answer = REFID_PAT.sub('{}', s).format(*formats)

    print(answer)
    print(boxes)
    print(seqs)
    return boxes, answer, seqs


def get_bl_example(ann, scene):
    boxes = []
    boxes_seq = []

    origin_sent = ann['fullAnswer']
    origin_sent = re.sub('(?:^Yes,)|(?:^No,)', '', origin_sent).strip()
    sent = list(origin_sent.split())
    for span, rids_str in ann['annotations']['fullAnswer'].items():
        span = tuple(map(int, span.split(':')))
        if len(span) == 1:
            span = [span[0], span[0] + 1]
        sent[span[1] - 1] = f"{sent[span[1] - 1]}{BOXES_PLACEHOLDER}"
        rids = rids_str.split(',')
        boxes_idx = add_boxes_by_rids(boxes, rids, scene)
        boxes_seq.append(boxes_idx)

    answer = "".join(sent)
    answer += f"The answer is {ann['answer']}."
    return boxes, answer, boxes_seq


@METRICS.register_module()
class GQAComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        correct = 0
        failed = 0
        target_failed = 0

        for pred, target in zip(preds, targets):
            print("answer: ", pred)
            print("\n\n\n")
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue
            if extract_pred is None:
                failed += 1
            if extract_pred == extract_target:
                correct += 1


            
        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
        }
    
    def extract_ans(self, string: str):
        try:
            found = ANS_EXTRACT_PAT.findall(string.strip())
            if len(found) != 1:
                return None
            return found[0].strip().rstrip('.').strip()
        except (IndexError, AttributeError):
            return None

    def extract_boxes(self, string: str):
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
