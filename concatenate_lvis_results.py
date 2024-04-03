import json 

precision_file = "../data/lvis_log.jsonl"
ann_file = "../data/lvis_ann.jsonl"

categories_dict = {}
with open("../data/lvis_v1_val.json", "r") as f:
    obj = json.load(f)
    categories = obj["categories"]

is_category_rare = {}

for category in categories:
    is_category_rare[category["id"]] = category["instance_count"] <= 10        

images = {}
categories = {}

lvis_run_logs = []

with open(precision_file, "r") as f:
    for line in f.readlines():
        lvis_run_logs.append(json.loads(line))

with open(ann_file, "r") as f:
    for i, line in enumerate(f.readlines()):
        obj = json.loads(line)

        if obj["category_id"] not in categories:
            categories[obj["category_id"]] = {"images" : [obj["img_path"]], "appearances" : 1, "mAP" : lvis_run_logs[i]["precision"], "name" : obj["category_name"], "is_category_rare" : is_category_rare[obj["category_id"]]}
        else:
            categories[obj["category_id"]]["images"].append(obj["img_path"])
            categories[obj["category_id"]]["appearances"] += 1
            categories[obj["category_id"]]["mAP"] += lvis_run_logs[i]["precision"]


        if obj["image_id"] not in images:
            images[obj["image_id"]] = {"path" : obj["img_path"], "num_objects" : obj["num_objects"], "precision" : lvis_run_logs[i]["precision"] * obj["num_objects"]}
        else:
            images[obj["image_id"]]["num_objects"] += obj["num_objects"]
            images[obj["image_id"]]["precision"] += lvis_run_logs[i]["precision"] * obj["num_objects"]



object_count_precision = {
    1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0,
    "6-10" : 0,
    "11-20" : 0,
    "20+" : 0
}

object_count_total = {
    1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0,
    "6-10" : 0,
    "11-20" : 0,
    "20+" : 0
}

for img_id in images.keys():
    img_obj = images[img_id]
    precision = img_obj["precision"]/img_obj["num_objects"]

    if img_obj["num_objects"] < 6:
        key = img_obj["num_objects"]
    elif img_obj["num_objects"] >= 6 and img_obj["num_objects"] <= 10:
        key = "6-10"
    elif img_obj["num_objects"] >= 11 and img_obj["num_objects"] <= 20:
        key = "11-20"
    else:
        key = "20+"

    object_count_precision[key] += precision
    object_count_total[key] += 1

print("Mean Average Precision (mAP) by the number of objects in the image:")
for obj_count in object_count_precision.keys():
    print("Count:", obj_count, "Precision:", object_count_precision[obj_count]/object_count_total[obj_count])


rare_precision = 0
rare_count = 0

common_count = 0
common_precision = 0


for category_id in categories.keys():
    categories[category_id]["mAP"] /= categories[category_id]["appearances"]

    if is_category_rare[category_id]:
        rare_count += 1
        rare_precision += categories[category_id]["mAP"]
    else:
        common_count += 1
        common_precision += categories[category_id]["mAP"]

print("-----------------------------------------------------------")

print("Precision for Rare objects:", rare_precision/rare_count)
print("Precision for Common objects:", common_precision/common_count)

with open("../data/lvis_by_class.jsonl", "w") as f:
    for i, k in enumerate(categories.keys()):
        jout = json.dump(categories[k], f)
        if i != len(categories) - 1:
            f.write("\n")