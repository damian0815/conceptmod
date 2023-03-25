Based on 'Erasing Concepts from Diffusion Models' [https://erasing.baulab.info](https://erasing.baulab.info)

## ConceptMod

Finetuning with words.

Allows manipulation of Stable Diffusion with it's own learned representations.

Example: 'vibrant colors++|boring--'

Will erase `boring` concept and exaggerate `vibrant colors` concept.

## Concept modifications

* Exaggerate: To exaggerate a concept, use the "++" operator.

  Example: "alpaca++" exaggerates "alpaca".

* Erase: To reduce a concept, use the "--" operator.

  Example: "monochrome--" reduces "monochrome".

* Freeze: Freeze by using the "#" operator. This reduces movement of specified term during training steps.

  Example: "@1woman#1woman" with "badword--" freezes the first phrase while deleting the badword.

  Note: "@#" means resist changing the unconditional.

* Orthogonal: To make two concepts orthogonal, use the "%" operator.

  Example: "cat%dog" makes "cat" and "dog" orthogonal. *untested term*

  *this term is unstable without regularizer. You will see NaN loss.*

  Set the alpha negative to pull dog to cat. "cat%dog:-0.1" *untested term*

* Replace: To replace use the following syntax:

  "@#|target%source:-0.03|source=target:0.1"

  *barely tested*

* Write to Unconditional: To write a concept to the unconditional model, use the "=" operator after the concept.

  Example: "alpaca=" causes the system to treat "alpaca" as a default concept or a concept that should always be considered during content generation.

  *untested term*

* Blend: Blend by using the "%" operator with ":-1.0", which means in reverse.

  Example: "anime%hyperrealistic:-1.0" blends "anime" and "hyperrealistic".

  *untested term*

## Prompt options

* Regularize: Prefix any term with '@' to move to priority queue (run each turn).

  Example: "@=priority term|=normal term"

  *untested term*

* Alpha: Add alpha to scale terms.

  Example: "=day time:0.75|=night time:0.25|@=enchanted lake"

  *untested term*

## Installation Guide

* To get started clone the following repository of Original Stable Diffusion [Link](https://github.com/CompVis/stable-diffusion)
* Then download the files from our iccv-esd repository to `stable-diffusion` main directory of stable diffusion. This would replace the `ldm` folder of the original repo with our custom `ldm` directory
* Download the weights from [here]([https://huggingface.co/CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt)) and move them to `stable-diffusion/models/ldm/` (This will be `ckpt_path` variable in `train-scripts/train-esd.py`)
* [Only for training] To convert your trained models to diffusers download the diffusers Unet config from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json)  (This will be `diffusers_config_path` variable in `train-scripts/train-esd.py`)

## Training Guide

Checkout `train_sequential.sh` for an example.

## Generating Images

To generate images from one of the custom models use the following instructions:

* To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
* To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
* The path to model can be customised in the script.
* It is to be noted that the current version requires the model to be in saved in `stable-diffusion/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
* `python eval-scripts/generate-images.py --model_name='compvis-word_VanGogh-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05' --prompts_path 'stable-diffusion/art_prompts.csv' --save_path 'evaluation_folder' --num_samples 10` 

## Notes

`mod_count` is set to two conceptmods being trained in parallel. You can reduce it if needed.
`negative_guidance`, `start_guidance` which are positive in the original repository, is negative in this one. See `train_sequential.sh` for usage example.

## Citing our work

Cite the original, maybe gpt-4
