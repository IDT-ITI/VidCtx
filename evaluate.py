# Andreas Goulas <agoulas@iti.gr>

import json
import torch
from tqdm import tqdm
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--answers", required=True)
parser.add_argument("--scores", required=True)
args = parser.parse_args()

with open(args.answers, "r") as f:
    answers = json.load(f)

with open(args.scores, "r") as f:
    scores = json.load(f)

correct = defaultdict(int)
total = defaultdict(int)
for qid, gt in tqdm(answers.items()):
    qtype = qid.split("_")[0]

    frame_scores = torch.tensor(scores[qid])
    norm_scores = torch.nn.functional.normalize(frame_scores, dim=1, p=1)

    pred_scores = torch.max(norm_scores, dim=0)[0]
    pred_scores_without_f = pred_scores[:-1]

    answer = torch.argmax(pred_scores_without_f)
    if answer == gt:
        correct[qtype] += 1

    total[qtype] += 1

for qtype in total.keys():
    print("{}: {}".format(qtype, round(correct[qtype] / total[qtype] * 100, 1)))

print("Total: {}".format(round(sum(correct.values()) / sum(total.values()) * 100, 1)))
