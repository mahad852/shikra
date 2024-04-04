import json

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
    box = x1, y1, x2, y2
    return box

with open("../data/lvis_by_class.jsonl", "r") as f:
    for line in f.readlines():
        category = json.loads(line)

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
        final_ds.append({**ann_obj, "pred_bboxes": list(map(de_norm_box_xyxy, lvis_logs[i]["pred_bboxes"]))})


print(final_ds)
print(len(final_ds), len(saved))