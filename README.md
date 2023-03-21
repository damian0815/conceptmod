Based on 'Erasing Concepts from Diffusion Models' [https://erasing.baulab.info](https://erasing.baulab.info)

Takes several hours of training. Does not require a dataset, only concept mod text.

You can train with multiple concepts at once separated with '|'. Example: 'vibrant colors^|boring--'

`train_sequential.sh` sequentially trains multiple concept mods with the proper commands.

## Notes

`mod_count` is set to two conceptmods being trained in parallel. You can reduce it if needed.
`negative_guidance`, `start_guidance` which are positive in the original repository, is negative in this one. See `train_sequential.sh` for usage example.

## Concept modifications

* Enhance: To enhance a concept, simply add a caret (^) after it.
  Example: "fluffy^" enhances the concept of "fluffy".

* Replace: To replace a concept with another, use the "=" operator.
  Example: "black and white=vibrant color" replaces "black and white" with "vibrant color".

* Increase Occurrence: To increase the occurrence of a concept, use the "++" operator.
  Example: "alpaca++" increases the occurrence of "alpaca".

* Reduce Occurrence: To reduce the occurrence of a concept, use the "--" operator.
  Example: "monochrome--" reduces the occurrence of "monochrome".

* Orthogonal: To make two concepts orthogonal, use the "%" operator.
  Example: "cat%dog" makes "cat" and "dog" orthogonal.

* Forget: To forget a concept, use the "=" operator followed by the concept.
  Example: "=alpaca" causes the system to forget or ignore the concept of "alpaca" during content generation.

* Write to Unconditional: To write a concept to the unconditional model, use the "=" operator after the concept.
  Example: "alpaca=" causes the system to treat "alpaca" as a default concept or a concept that should always be considered during content generation.

* Blend: To blend two concepts, use the "+" operator.
  Example: "anime~hyperrealistic" blends "anime" and "hyperrealistic".

## Installation Guide

* To get started clone the following repository of Original Stable Diffusion [Link](https://github.com/CompVis/stable-diffusion)
* Then download the files from our iccv-esd repository to `stable-diffusion` main directory of stable diffusion. This would replace the `ldm` folder of the original repo with our custom `ldm` directory
* Download the weights from [here]([https://huggingface.co/CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt)) and move them to `stable-diffusion/models/ldm/` (This will be `ckpt_path` variable in `train-scripts/train-esd.py`)
* [Only for training] To convert your trained models to diffusers download the diffusers Unet config from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json)  (This will be `diffusers_config_path` variable in `train-scripts/train-esd.py`)

## Training Guide

After installation, follow these instructions to train a custom ESD model:

* `cd stable-diffusion` to the main repository of stable-diffusion
* [IMPORTANT] Edit `train-script/train-esd.py` and change the default argparser values according to your convenience (especially the config paths)
* To choose train_method, pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'` 
* `python train-scripts/train-esd.py --prompt 'your prompt' --train_method 'your choice of training' --devices '0,1'`

Note that the default argparser values must be changed!

The optimization process for erasing undesired visual concepts from pre-trained diffusion model weights involves using a short text description of the concept as guidance. The ESD model is fine-tuned with the conditioned and unconditioned scores obtained from frozen SD model to guide the output away from the concept being erased. The model learns from it's own knowledge to steer the diffusion process away from the undesired concept.
<div align='center'>
<img src = 'images/ESD.png'>
</div>

## Generating Images

To generate images from one of the custom models use the following instructions:

* To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
* To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
* The path to model can be customised in the script.
* It is to be noted that the current version requires the model to be in saved in `stable-diffusion/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
* `python eval-scripts/generate-images.py --model_name='compvis-word_VanGogh-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05' --prompts_path 'stable-diffusion/art_prompts.csv' --save_path 'evaluation_folder' --num_samples 10` 

## Citing our work

Cite the original
