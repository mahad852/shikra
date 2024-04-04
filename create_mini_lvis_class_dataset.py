import json
import os
import cv2
import numpy as np


limit = 4
num_rare = 0
num_common = 0

rare_accurate_chosen = 0
rare_inaccurate_chosen = 0

common_accurate_chosen = 0
common_inaccurate_chosen = 0

saved = []
image_chosen = {}

def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = [x1, y1, x2 - x1, y2 - y1]
    return box

is_category_rare = {}

with open("../data/lvis_by_class.jsonl", "r") as f:
    for line in f.readlines():
        category = json.loads(line)
        is_category_rare[category["name"]] = category["is_category_rare"]
        if category["is_category_rare"] and num_rare >= 4:
            continue
        if not category["is_category_rare"] and num_common >= 4:
            continue    

        if category["is_category_rare"]:
            if ((category["mAP"] == 1.0 and rare_accurate_chosen >= limit/2) or 
                (category["mAP"] < 1.0 and rare_inaccurate_chosen >= limit/2)):
                continue

            num_rare += 1
            saved.append(category)

            if category["mAP"] == 1.0:
                rare_accurate_chosen += 1
            else:
                rare_inaccurate_chosen += 1
        
        else:
            if ((category["mAP"] == 1.0 and common_accurate_chosen >= limit/2) or 
                (category["mAP"] < 1.0 and common_inaccurate_chosen >= limit/2)):
                continue

            num_common += 1
            saved.append(category)

            if category["mAP"] == 1.0:
                common_accurate_chosen += 1
            else:
                common_inaccurate_chosen += 1

        if num_rare >= 4 and num_common >= 4:
            break


for category in saved:
    if category["name"] not in image_chosen:
        image_chosen[category["name"]] = {}

    for image_path in category["images"]:
        image_chosen[category["name"]][image_path] = True

final_ds = []
fpath_final_ds = "../data/lvis_mini_class.jsonl"
ann_objs = []

with open("../data/lvis_ann.jsonl", "r") as f:
    for i, line in enumerate(f.readlines()):
        ann_objs.append(json.loads(line))

lvis_logs = []
with open("../data/lvis_log.jsonl", "r") as f:
    for line in f.readlines():
        lvis_logs.append(json.loads(line))


for i, ann_obj in enumerate(ann_objs):
    if ann_obj["category_name"] in image_chosen and ann_obj["img_path"] in image_chosen[ann_obj["category_name"]]:
        final_ds.append({**ann_obj, 
                         "pred_bboxes": list(map(lambda bbox: de_norm_box_xyxy(bbox, w=ann_obj["width"], h=ann_obj["height"]), lvis_logs[i]["pred_bboxes"])),
                         "is_category_rare" : is_category_rare[ann_obj["category_name"]]})

rare = 0
common = 0

for obj in final_ds:
    img_path = os.path.join('/datasets/MSCOCO17', obj["img_path"])
    gt_boxes = obj["bboxes"]
    pred_boxes = obj["pred_bboxes"]
    is_category_rare = obj["is_category_rare"]
    category_name = obj["category_name"]

    img = cv2.imread(img_path)

    for box in gt_boxes:
        x,y,w,h = box
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

    for box in pred_boxes:
        x,y,w,h = box
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)

    
    img_name = category_name + "_" + ("rare" if is_category_rare else "common") + ".jpg"
    if is_category_rare:
        rare += 1
    else:
        common += 1

    cv2.imwrite("..data/images/" + img_name, img)
    print("image written to:", "..data/images/" + img_name)
    


# print(final_ds)
print(len(final_ds), len(saved))