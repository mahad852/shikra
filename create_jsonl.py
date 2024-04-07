import json

json_file = "../data/questions1.2/val_balanced_questions.json"
jsonl_file = "../data/questions1.2/val_balanced_questions.jsonl"

obj = json.load(open(json_file, "r"))

final_key = list(obj.keys())[-1]

for key in obj.keys():
    with open(jsonl_file, "a") as f:
        json.dump(obj[key], f)
        if key != final_key:
            f.write("\n")