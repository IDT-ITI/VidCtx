# Andreas Goulas <agoulas@iti.gr>

import json
import torchvision
import os
import torch
from tqdm import tqdm
import argparse

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument("--questions", required=True)
parser.add_argument("--outfile", required=True)
parser.add_argument("--capdir", required=True)
parser.add_argument("--num_frames", default=64, type=int)
parser.add_argument("--num_options", default=5, type=int)
args = parser.parse_args()

qconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    quantization_config=qconfig
)

target_chars = ["A", "B", "C", "D", "E", "F"][:args.num_options+1]
targets = processor.tokenizer.convert_tokens_to_ids(target_chars)

with open(args.questions, "r") as f:
    questions = json.load(f)

result = {}
for line in tqdm(questions):
    frames, _, _ = torchvision.io.read_video(
        line["path"],
        start_pts=line["start"] if "start" in line else 0,
        end_pts=line["end"] if "end" in line else None,
        pts_unit="sec"
    )
    
    step = frames.shape[0] / args.num_frames
    idxs = [int(step * (0.5 + i)) for i in range(args.num_frames)]
    frames = frames[idxs]

    cappath = os.path.join(args.capdir, "{}.json".format(line["qid"]))

    if not os.path.exists(cappath):
        cap = []
        for i in range(args.num_frames):
            prompt = "[INST] <image>\nPlease provide a short description of the image giving information related to the following question: {} [/INST]".format(
                line["q"]
            )

            inputs = processor(text=prompt, images=frames[i], return_tensors="pt").to(
                "cuda"
            )

            ids = model.generate(**inputs, max_new_tokens=200)
            out = processor.batch_decode(ids, skip_special_tokens=True)[0]
            cap.append(out[out.find("[/INST]") + 8 :])

        with open(cappath, "w") as f:
            json.dump(cap, f, indent=2)
    else:
        with open(cappath, "r") as f:
            cap = json.load(f)

    video_scores = []
    for i in range(args.num_frames):
        half_len = args.num_frames // 2
        
        if args.num_options == 5:
            prompt = "[INST] <image>\nHere is what happens {} in the video: {}\nQuestion: {} Option A: {}. Option B: {}. Option C: {}. Option D: {}. Option E: {}. Option F: No Answer. Considering the information presented in the caption and the video frame, select the correct answer in one letter from the options (A,B,C,D,E,F). [/INST]".format(
                "earlier" if i >= half_len else "later",
                cap[i - half_len] if i >= half_len else cap[i + half_len],
                line["q"],
                line["a0"],
                line["a1"],
                line["a2"],
                line["a3"],
                line["a4"],
            )
        else:
            prompt = "[INST] <image>\nHere is what happens {} in the video: {}\nQuestion: {} Option A: {}. Option B: {}. Option C: {}. Option D: {}. Option E: No Answer. Considering the information presented in the caption and the video frame, select the correct answer in one letter from the options (A,B,C,D,E). [/INST]".format(
                "earlier" if i >= half_len else "later",
                cap[i - half_len] if i >= half_len else cap[i + half_len],
                line["q"],
                line["a0"],
                line["a1"],
                line["a2"],
                line["a3"],
            )

        inputs = processor(text=prompt, images=frames[i], return_tensors="pt").to(
            "cuda"
        )

        scores = model.generate(
            **inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True
        )["scores"][0]
        
        scores = scores[0, targets]
        video_scores.append(scores.tolist())

    result[line["qid"]] = video_scores

with open(args.outfile, "w") as f:
    json.dump(result, f, indent=2)
