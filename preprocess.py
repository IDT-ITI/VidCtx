# Andreas Goulas <agoulas@iti.gr>

import json
import os
import csv

#
# NEXT-QA
#

with open("data/map_vid_vidorID.json", "r") as f:
    video_map = json.load(f)

rows = []
with open("data/nextqa/val.csv", "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        rows.append(row)

questions = []
answers = {}

for idx, row in enumerate(rows):
    qid = "{}_{}_{}".format(row[7][0], row[0], row[6])

    video_id = row[0]
    filepath = video_map[video_id].split("/")

    questions.append(
        {
            "qid": qid,
            "path": os.path.join("data/NExTVideo", filepath[0], filepath[1]) + ".mp4",
            "q": row[4] + "?",
            "a0": row[8],
            "a1": row[9],
            "a2": row[10],
            "a3": row[11],
            "a4": row[12],
        }
    )

    answers[qid] = int(row[5])

with open("questions_nextqa.json", "w") as f:
    json.dump(questions, f, indent=2)

with open("answers_nextqa.json", "w") as f:
    json.dump(answers, f, indent=2)

#
# IntentQA
#

with open("data/map_vid_vidorID.json", "r") as f:
    video_map = json.load(f)

rows = []
with open("data/intentqa/test.csv", "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        rows.append(row)

questions = []
answers = {}

for idx, row in enumerate(rows):
    qid = "{}_{}_{}".format(row[7][0], row[0], row[6])

    video_id = row[0]
    filepath = video_map[video_id].split("/")

    questions.append(
        {
            "qid": qid,
            "path": os.path.join("data/NExTVideo", filepath[0], filepath[1]) + ".mp4",
            "q": row[4] + "?",
            "a0": row[8],
            "a1": row[9],
            "a2": row[10],
            "a3": row[11],
            "a4": row[12],
        }
    )

    answers[qid] = int(row[5])

with open("questions_intentqa.json", "w") as f:
    json.dump(questions, f, indent=2)

with open("answers_intentqa.json", "w") as f:
    json.dump(answers, f, indent=2)
    
#
# STAR
#

with open("data/star/STAR_val.json") as f:
    lines = json.load(f)

questions = []
answers = {}

for line in lines:
    qid = line["question_id"]

    gt = -1
    for idx in range(4):
        if line["choices"][idx]["choice"] == line["answer"]:
            gt = idx
            break

    questions.append(
        {
            "qid": qid,
            "start": line["start"],
            "end": line["end"],
            "path": os.path.join("data/Charades_v1_480", line["video_id"]) + ".mp4",
            "q": line["question"],
            "a0": line["choices"][0]["choice"].rstrip("."),
            "a1": line["choices"][1]["choice"].rstrip("."),
            "a2": line["choices"][2]["choice"].rstrip("."),
            "a3": line["choices"][3]["choice"].rstrip("."),
        }
    )

    answers[qid] = gt

with open("questions_star.json", "w") as f:
    json.dump(questions, f, indent=2)

with open("answers_star.json", "w") as f:
    json.dump(answers, f, indent=2)
