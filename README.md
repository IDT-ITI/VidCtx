# VidCtx

## Introduction

This repository contains the code for our paper: "VidCtx: Context-aware Video Question Answering with Image Models".

## Requirements

* torch
* torchvision
* transformers
* bitsandbytes
* tqdm

## Datasets

Download the video files and annotations from the official sources:
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

Finally, run the following script to preprocess the datasets: ```python preprocess.py```

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

```
@article{goulas2024vidctx,
  title={VidCtx: Context-aware Video Question Answering with Image Models},
  author={Goulas, Andreas and Mezaris, Vasileios and Patras, Ioannis},
  journal={arXiv preprint arXiv:2412.17415},
  year={2024}
}
```

## License

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgments

This work was supported by the EU Horizon Europe programme under grant agreement 101070109 TransMIXR.
