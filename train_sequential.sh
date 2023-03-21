#!/bin/bash

# List of prompts
prompts=(
"=adorable happy robot|girl=robot|boy=robot"
"=vivid and colorful|black and white=vibrant color|enchanting and mystical++"
)

# Training parameters
train_method="full"
devices="0,1"
ckpt_path="../stable-diffusion-webui2/models/Stable-diffusion/criarcysFantasyTo_v30.safetensors"
negative_guidance=-1.5
start_guidance=-3
iterations=2000

# Train on each prompt sequentially
for prompt in "${prompts[@]}"; do
    echo "Training on prompt: '$prompt'"
    python train-scripts/train-esd.py --prompt "$prompt" --train_method "$train_method" --devices "$devices" --ckpt_path "$ckpt_path" --negative_guidance "$negative_guidance" --start_guidance "$start_guidance" --iterations "$iterations" --seperator "|"
    echo "Finished training on prompt: '$prompt'"
done

echo "All prompts have been trained."
