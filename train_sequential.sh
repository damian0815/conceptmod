#!/bin/bash

# List of prompts
prompts=(
"@#|++adorable happy robot++|girl=robot|boy=robot"
)

# Training parameters
train_method="full"
devices="0,1"
ckpt_path="../stable-diffusion-webui2/models/Stable-diffusion/criarcysFantasyTo_v30.safetensors"
negative_guidance=-1.5
start_guidance=-3
iterations=2000
sample_prompt="adorable happy robot sitting the middle of new york, busy shoppes in backdrop"

# Train on each prompt sequentially
for prompt in "${prompts[@]}"; do
    echo "Training on prompt: '$prompt'"
    python train-scripts/train-esd.py --prompt "$prompt" --train_method "$train_method" --devices "$devices" --ckpt_path "$ckpt_path" --negative_guidance "$negative_guidance" --start_guidance "$start_guidance" --iterations "$iterations" --seperator "|" --mod_count 2 --sample_prompt "$sample_prompt"
    echo "Finished training on prompt: '$prompt'"
done

echo "All prompts have been trained."
