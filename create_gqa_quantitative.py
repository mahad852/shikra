import json
import os
import cv2

def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = [x1, y1, x2 - x1, y2 - y1]
    return box

total = 10

val_questions = []
with open("../data/questions1.2/val_balanced_questions.jsonl", "r") as f:
    for line in f.readlines():
        val_questions.append(json.loads(line))

with open("../data/gqa_log.jsonl", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == total:
            break

        res = json.loads(line)
        pred_boxes = res["pred_boxes"]
        target_boxes = res["target"]
        question = res["question"]
        image_id = val_questions[i]["imageId"]

        image_path = os.path.join("/datasets/GQA/images", image_id + ".jpg")
        

        img = cv2.imread(image_path)

        for box in target_boxes:
            x,y,w,h = de_norm_box_xyxy(box)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

        for box in pred_boxes:
            x,y,w,h = de_norm_box_xyxy(box)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)

        
        img_name = f"gqa_{i}_{image_id}" + ".jpg"

        cv2.imwrite("../data/images/gqa/" + img_name, img)
        print("image written to:", "../data/images/gqa" + img_name)
        print("question:", i, question)
