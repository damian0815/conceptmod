from omegaconf import OmegaConf
from collections import defaultdict
import copy

from safetensors import safe_open
from datasets import load_dataset

import sys; sys.path.append('.')
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
#import ImageReward as reward
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
import torchvision.transforms.functional as F


import time
from contextlib import nullcontext
from PIL import Image


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    tensors = {}
    mPath=ckpt
    if "safetensors" in mPath:
        with safe_open(mPath, framework="pt", device="cpu") as f:
           for key in f.keys():
               tensors[key] = f.get_tensor(key).cpu()

        #global_step = pl_sd["global_step"]
        sd = tensors#pl_sd["state_dict"]
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd#["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu'):
    folder_path = f'opposite/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.ckpt'
    else:
        path = f'{folder_path}/{name}.ckpt'
    print("Saved model to "+path)
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--trained', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='/sd-models/SDv1-5.ckpt')
    parser.add_argument('--base', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='/sd-models/SDv1-5.ckpt')
    parser.add_argument('--output', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='opposite.ckpt')
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    args = parser.parse_args()
    config_path = args.config_path
    base = load_model_from_config(config_path, args.base, "cpu")
    trained = load_model_from_config(config_path, args.trained, "cpu")
    opposite_model = load_model_from_config(config_path, args.base, "cpu")
    target_state = trained.state_dict()
    source_state = base.state_dict()
    opposite_state = opposite_model.state_dict()

    for key in target_state:
        opposite_state[key] = 2*source_state[key] - target_state[key]

    opposite_model.load_state_dict(opposite_state)

    save_model(opposite_model, args.output, 0)
