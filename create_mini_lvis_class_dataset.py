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
category_name_chosen = {}

with open("../data/lvis_by_class.jsonl", "w") as f:
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
    for image_path in category["images"]:
        image_chosen[image_path] = True
    category_name_chosen[category["name"]] = True


final_ds = []
fpath_final_ds = "../data/lvis_mini_class.jsonl"
ann_objs = []

with open("../data/lvis_ann.jsonl", "r") as f:
    for i, line in enumerate(f.readlines()):
        ann_objs.append(json.loads(line))

lvis_logs = []
with open("../data/lvis_log.jsonl", "r") as f:
    for line in enumerate(f.readlines()):
        lvis_logs.append(json.loads(line))


for i, ann_obj in enumerate(ann_objs):
    if ann_obj["category_name"] in category_name_chosen and ann_obj["img_path"] in image_chosen:
        final_ds.append({**ann_obj, "pred_bboxes": lvis_logs[i]["pred_bboxes"]})


print(final_ds)