from omegaconf import OmegaConf
import torch.optim.lr_scheduler as lr_scheduler

from safetensors import safe_open

import sys; sys.path.append('.')
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
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
from convertModels import savemodelDiffusers


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

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def parse_input_string(input_str):
    params = {
        "alpha": 1.0,  # Default alpha value
    }

    # Split the input string by ':' to get the concepts and parameters
    parts = input_str.split(':')

    # Set the concept
    params["concept"] = parts[0]

    # Iterate through the remaining parts to parse parameters
    for part in parts[1:]:
        # Check if the parameter has a '=' sign, indicating a key-value pair
        if '=' in part:
            key, value = part.split('=', 1)
            params[key] = float(value)
        else:
            # If it's just a value, assume it's the alpha value
            params["alpha"] = float(part)

    return params


def select_rules(rules, num_rules):
    if num_rules > len(rules):
        raise ValueError("num_rules must be less than or equal to the length of the rules list.")

    selected_rules = [rule for rule in rules if rule.startswith('@')]

    if num_rules == 1 and len(rules) == 1:
        selected_rules = rules
    else:
        while num_rules > len(selected_rules):
            remaining_rules = [rule for rule in rules if rule not in selected_rules]
            selected_rules.append(random.choice(remaining_rules))

    return selected_rules

def write_sample_png(name, model, sampler, sample_start_code, sample_emb, step, ddim_steps):
    start_code = sample_start_code
    device = sample_start_code.device

    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                tic = time.time()
                uc = None
                uc = model.get_learned_conditioning([""])
                c = sample_emb
                shape = [4, 64, 64]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=7.5,
                                                 unconditional_conditioning=uc,
                                                 eta=0.0,
                                                 x_T=start_code)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
                x_sample = x_samples_ddim

                x_sample = 255. * x_sample
                x_sample = x_sample.astype(np.uint8)
                img = Image.fromarray(x_sample)
                img.save(f"{name}/{step:05}.png")

def train_esd(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50, sample_prompt=None, accumulation_steps=1, mod_count=3):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.
    mod_count : int, optional
        Number of conceptmods to run in parallel. The default is 2.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(words)
    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)

    # Add a learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100, verbose=True, threshold=1e-4)

    criteria = torch.nn.MSELoss()
    history = []

    name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    name = name[0:50]
    # TRAINING CODE
    pbar = tqdm(range(iterations))

    if sample_prompt is not None:
        sample_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
        sample_emb = model.get_learned_conditioning([sample_prompt])

        os.makedirs("samples/"+name, exist_ok=True)
        write_sample_png("samples/"+name, model, sampler, sample_start_code, sample_emb, 0, ddim_steps)
    accumulation_counter=0

    for i in pbar:
        num_rules = mod_count
        rules = select_rules(words, num_rules)
        rules = [s[1:] if s.startswith("@") else s for s in rules]
        print("Selected:", rules)
        #model_orig.load_state_dict(model.state_dict())

        opt.zero_grad()

        rule_losses = []
        for rule_params in rules:
            rule_index = rules.index(rule_params)
            rule_obj = parse_input_string(rule_params)
            rule = rule_obj['concept']

            start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
            t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
            og_num = round((int(t_enc)/ddim_steps)*1000)
            og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
            insertion_guidance=negative_guidance


            if '=' in rule:
                # Handle the concept replacement case (original=target)
                concepts = rule.split('=')
                original_concept = concepts[0]
                target_concept = concepts[1]

                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])
                emb_o = model.get_learned_conditioning([original_concept])
                emb_t = model.get_learned_conditioning([target_concept])

                with torch.no_grad():
                    # Generate an image with the target concept from ESD model
                    z = quick_sample_till_t(emb_t.to(devices[0]), start_guidance, start_code, int(t_enc))

                    # Get conditional and unconditional scores from frozen model at time step t and image z
                    e_0 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
                    e_o = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_t.to(devices[1]))

                # Get conditional scores from ESD model for the original concept
                e_t = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_o.to(devices[0]))

                # Compute the loss function for concept replacement
                loss_replacement = criteria(e_t.to(devices[0]), e_0.to(devices[0]) - (negative_guidance * (e_o.to(devices[0]) - e_0.to(devices[0]))))
                loss_rule = rule_obj['alpha']*loss_replacement
                rule_losses.append(loss_rule)

            elif '#' in rule:
                concepts = rule.split('#')
                original_concept = concepts[0]
                target_concept = concepts[1]

                emb_o = model.get_learned_conditioning([original_concept])
                emb_t = model.get_learned_conditioning([target_concept])

                with torch.no_grad():
                    z = quick_sample_till_t(emb_t.to(devices[0]), start_guidance, start_code, int(t_enc))
                    e_o = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_t.to(devices[1]))

                e_t = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_o.to(devices[0]))

                loss_replacement = criteria(e_t.to(devices[0]), e_o.to(devices[0]))
                loss_rule = rule_obj['alpha']*loss_replacement
                rule_losses.append(loss_rule)

            elif '++' == rule[-2:] or rule[-2:] == '--':
                # Handle the concept insertion case (concept++)
                concept_to_insert = rule[:-2]
                if rule[:-2] == '--':
                    insertion_guidance = -insertion_guidance

                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])
                emb_i = model.get_learned_conditioning([concept_to_insert])

                with torch.no_grad():
                    # Generate an image from ESD model
                    z = quick_sample_till_t(emb_0.to(devices[0]), start_guidance, start_code, int(t_enc))

                    # Get conditional and unconditional scores from frozen model at time step t and image z
                    e_0 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
                    e_i = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_i.to(devices[1]))

                # Get conditional scores from ESD model for the concept to insert
                e_t = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_i.to(devices[0]))

                # Compute the loss function to encourage the presence of the concept in the generated images
                loss_i = criteria(e_t.to(devices[0]), e_0.to(devices[0]) - (insertion_guidance * (e_i.to(devices[0]) - e_0.to(devices[0]))))
                loss_rule = rule_obj["alpha"]*loss_i
                rule_losses.append(loss_rule)


            elif '%' in rule:
                # Handle the concept orthogonality case (concept1%concept2)
                concept1, concept2 = rule.split('%')
                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])

                emb_c1 = model.get_learned_conditioning([concept1])
                emb_c2 = model.get_learned_conditioning([concept2])

                with torch.no_grad():
                    # Generate an image from ESD model
                    z = quick_sample_till_t(emb_0.to(devices[0]), start_guidance, start_code, int(t_enc))

                    # Get conditional and unconditional scores from frozen model at time step t and image z
                    e_0 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
                    output_c1 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_c1.to(devices[1]))

                # Get conditional scores from ESD model for the two concepts
                output_c2 = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_c2.to(devices[0]))
                diff_c1 = output_c1 - e_0
                diff_c2 = output_c2 - e_0.to(devices[0])

                diff_c1_flat = diff_c1.view(1, -1)
                diff_c2_flat = diff_c2.view(1, -1)
                # Normalize the output embeddings
                normalized_output_c1 = diff_c1_flat / (torch.norm(diff_c1_flat, dim=1, keepdim=True)+1e-8)
                normalized_output_c2 = diff_c2_flat / (torch.norm(diff_c2_flat, dim=1, keepdim=True)+1e-8)

                # Calculate the cosine similarity between the normalized output embeddings
                cosine_similarity = torch.abs(torch.dot(normalized_output_c1.view(-1).to(devices[0]), normalized_output_c2.view(-1).to(devices[0])))

                loss_rule = rule_obj["alpha"]*0.01*cosine_similarity
                rule_losses.append(loss_rule)

            elif rule[-1] == '^':
                raise ValueError('^ removed (use ++)')

            # Handle the concept removal case (concept--)
            elif rule[-2:] == '--':
                concept_to_reduce = rule[:-2]

                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])
                emb_r = model.get_learned_conditioning([concept_to_reduce])

                with torch.no_grad():
                    # Generate an image from ESD model
                    z = quick_sample_till_t(emb_0.to(devices[0]), -start_guidance, start_code, int(t_enc))

                    # Get conditional and unconditional scores from frozen model at time step t and image z
                    e_0 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
                    e_r = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_r.to(devices[1]))

                # Get conditional scores from ESD model for the concept to reduce
                e_t = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_r.to(devices[0]))

                # Compute the loss function to discourage the presence of the concept in the generated images
                loss_i = criteria(e_t.to(devices[0]), e_0.to(devices[0]) + (insertion_guidance * (e_r.to(devices[0]) - e_0.to(devices[0]))))

                loss_rule = rule_obj["alpha"]*loss_i
                rule_losses.append(loss_rule)
            else:
                assert False, "Unable to parse rule: "+rule


        loss = sum(rule_losses)
        # Update weights to erase or reinforce the concept(s)
        loss.backward()
        for j, r in enumerate(rule_losses):
            print("{:.5f}".format(rule_losses[j].item()), rules[j])
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        accumulation_counter+=1
        if accumulation_counter % accumulation_steps == 0:
            print("Ya")
            opt.step()
            opt.zero_grad()
            if sample_prompt is not None:
                os.makedirs("samples/"+name, exist_ok=True)
                write_sample_png("samples/"+name, model, sampler, sample_start_code, sample_emb, (accumulation_counter//accumulation_steps), ddim_steps)


        # save checkpoint and loss curve
        if (i+1) % 30 == 0 and i+1 != iterations and i+1>= 20:
            save_model(model, name, i-1, save_compvis=True, save_diffusers=False)

        if i % 100 == 0:
            save_history(losses, name, word_print)


    model.eval()

    save_model(model, name, None, save_compvis=True, save_diffusers=False, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print)

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.ckpt'
    else:
        path = f'{folder_path}/{name}.ckpt'
    print("Saved model to "+path)
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--accumulation_steps', help='gradient accumulation steps', type=int, required=False, default=1)
    parser.add_argument('--sample_prompt', help='will create training images with this phrase as SD trains. This requires running through SD and is slower.', type=str, required=False, default=None)
    parser.add_argument('--mod_count', help='number of mods to use at once', type=int, required=False, default=2)
    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    sample_prompt = args.sample_prompt
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    mod_count = args.mod_count

    train_esd(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, mod_count=mod_count, sample_prompt=sample_prompt, accumulation_steps=args.accumulation_steps)
