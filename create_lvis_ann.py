import json

with open("../data/lvis_v1_val.json", "r") as f:
    obj = json.load(f)

annotations_dict = obj["annotations"]
images_dict = obj["images"]
categories_dict = obj["categories"]

category_id_to_name = {}
max_cat_id = 0

for i in range(len(categories_dict)):
    category_id_to_name[categories_dict[i]["id"]] = categories_dict[i]["name"]
    max_cat_id = max_cat_id if categories_dict[i]["id"] <= max_cat_id else categories_dict[i]["id"]

image_id_to_info = {}
max_image_id = 0

for i in range(len(images_dict)):
    image_name = images_dict[i]["coco_url"].split('/')[-1]
    image_id_to_info[images_dict[i]["id"]] = {"img_path" : "val2017/" + image_name, 
                                              "height" : images_dict[i]["height"], 
                                              "width" : images_dict[i]["width"]}
    max_image_id = max_image_id if images_dict[i]["id"] <= max_image_id else images_dict[i]["id"]

final_res_indexer = [[-1 for _ in range(max_cat_id + 1)] for _ in range(max_image_id + 1)]
final_res = []

for ann_obj in (annotations_dict):
    image_id = ann_obj["image_id"]
    category_id = ann_obj["category_id"]

    if final_res_indexer[image_id][category_id] != -1:
        final_res[final_res_indexer[image_id][category_id]]["num_objects"] += 1
        final_res[final_res_indexer[image_id][category_id]]["bboxes"].append(ann_obj["bbox"])
        continue

    final_res_indexer[image_id][category_id] = len(final_res)

    bbox = ann_obj["bbox"]
    updated_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    final_res.append({
        "image_id" : image_id,
        "category_id" : category_id,
        **image_id_to_info[image_id],
        "category_name" : category_id_to_name[category_id],
        "bboxes" : [updated_bbox],
        "num_objects" : 1
    })

with open("../data/lvis_ann.jsonl", "w") as f:
    for i, res_dict in enumerate(final_res):
        jout = json.dump(res_dict, f)
        if i != len(final_res) - 1:
            f.write("\n")
