# VidCtx: Context-aware Video Question Answering with Image Models

## Introduction [[Paper preprint](https://arxiv.org/abs/2412.17415)] [[Cite](#citation)]

This repository contains the code for our paper: "VidCtx: Context-aware Video Question Answering with Image Models".

To address computational and memory limitations of Large Multimodal Models in the Video Question-Answering task, several recent methods extract textual representations per frame (e.g., by captioning) and feed them to a Large Language Model (LLM) that processes them to produce the final response. However, in this way, the LLM does not have access to visual information and often has to process repetitive textual descriptions of nearby frames. To address those shortcomings, in this paper, we introduce VidCtx, a novel training-free VideoQA framework which integrates both modalities, i.e. both visual information from input frames and textual descriptions of others frames that give the appropriate context. More specifically, in the proposed framework a pre-trained Large Multimodal Model (LMM) is prompted to extract at regular intervals, question-aware textual descriptions (captions) of video frames. Those will be used as context when the same LMM will be prompted to answer the question at hand given as input a) a certain frame, b) the question and c) the context/caption of an appropriate frame. To avoid redundant information, we chose as context the descriptions of distant frames. Finally, a simple yet effective max pooling mechanism is used to aggregate the frame-level decisions. This methodology enables the model to focus on the relevant segments of the video and scale to a high number of frames. Experiments show that VidCtx achieves competitive performance among approaches that rely on open models on three public Video QA benchmarks, NExT-QA, IntentQA and STAR.

We release our evaluation and inference code. VidCtx is a zero-shot approach and requires no additional training.

We use the following pre-trained model: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

## Requirements

* torch
* torchvision
* transformers
* bitsandbytes
* tqdm

## Datasets

We use the following 3 datasets for evaluation. Download the video files and annotations from the official sources:
- NExT-QA: https://github.com/doc-doc/NExT-QA
- IntentQA: https://github.com/JoseponLee/IntentQA
- STAR: https://bobbywu.com/STAR/

Then, place the files in the ```data``` folder with the following structure:
```
data
- nextqa
- - val.csv             # validation set of NExT-QA
- intentqa
- - test.csv            # test set of IntentQA
- star
- - STAR_val.json       # validation set of STAR
- map_vid_vidorID.json  # video paths for NExT-QA and IntentQA
- NExTVideo             # video files for NExT-QA and IntentQA
- Charades_v1_480       # video files for STAR
```

Finally, run the following script to convert the annotations to JSON: ```python preprocess.py```

This script will create a `questions.json` and `answers.json` file for each dataset.

## Evaluation

### NExT-QA

```
mkdir captions_nextqa

python run.py \
  --questions questions_nextqa.csv \
  --outfile scores_nextqa.json \
  --capdir captions_nextqa \
  --num_frames 64 \
  --num_options 5

python evaluate.py \
  --answers answers_nextqa.json \
  --scores scores_nextqa.json
```

### IntentQA

```
mkdir captions_intentqa

python run.py \
  --questions questions_intentqa.csv \
  --outfile scores_intentqa.json \
  --capdir captions_intentqa \
  --num_frames 64 \
  --num_options 5

python evaluate.py \
  --answers answers_intentqa.json \
  --scores scores_intentqa.json
```

### STAR

```
mkdir captions_star

python run.py \
  --questions questions_star.csv \
  --outfile scores_star.json \
  --capdir captions_star \
  --num_frames 32 \
  --num_options 4

python evaluate.py \
  --answers answers_star.json \
  --scores scores_star.json
```

## Citation

If you find our work or code useful, please cite our publication:

A. Goulas, V. Mezaris, I. Patras, "VidCtx: Context-aware Video Question Answering with Image Models", IEEE Int. Conf. on Multimedia and Expo (ICME 2025), Nantes, France, June-July 2025.

```
@inproceedings{goulas2025vidctx,
  title={VidCtx: Context-aware Video Question Answering with Image Models},
  author={Goulas, Andreas and Mezaris, Vasileios and Patras, Ioannis},
  booktitle={IEEE Int. Conf. on Multimedia and Expo (ICME 2025)},
  year={2025},
  organization={IEEE}
}
```

Link to preprint: https://arxiv.org/abs/2412.17415

## License

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgments

This work was supported by the EU Horizon Europe programme under grant agreement 101070109 TransMIXR.
