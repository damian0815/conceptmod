#!/bin/bash

# List of prompts
prompts=(
"#:0.4|human=robot:0.8|robot%human:-0.1"
)

# Training parameters
train_method="selfattn"
devices="0,0"
ckpt_path="../stable-diffusion-webui2/models/Stable-diffusion/criarcysFantasyTo_v30.safetensors"
negative_guidance=-1.5
start_guidance=-3
iterations=2000
accumulation_steps=2
mod_count=3
sample_prompt="man, looking serious overlooking city, close up view of face, face fully visible"

# Train on each prompt sequentially
for prompt in "${prompts[@]}"; do
    echo "Training on prompt: '$prompt'"
    python train-scripts/train-esd.py --prompt "$prompt" --train_method "$train_method" --devices "$devices" --ckpt_path "$ckpt_path" --negative_guidance "$negative_guidance" --start_guidance "$start_guidance" --iterations "$iterations" --seperator "|" --accumulation_steps $accumulation_steps --sample_prompt "$sample_prompt" 
    echo "Finished training on prompt: '$prompt'"
done

echo "All prompts have been trained."
