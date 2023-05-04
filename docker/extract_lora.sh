#!/bin/bash

# Set the directory path
dir="../../stable-diffusion-webui2/models/Stable-diffusion/0new/"

# Loop through all files in the directory
for file in "${dir}"*; do
    # Extract the filename without extension
    filename=$(basename -- "${file%.*}")

    # Remove special characters from the filename
    clean_filename=$(echo "${filename}" | tr -d '|~#:')

    # Set the target file path
    target="../../stable-diffusion-webui2/models/Lora/0new/${clean_filename}.safetensors"

    # Check if the target file exists; if it does, skip the command
    if [ ! -f "${target}" ]; then
        # Run the command with the required arguments
        python3 networks/extract_lora_from_models.py \
            --save_precision fp16 \
            --save_to "${target}" \
            --model_org ../../stable-diffusion-webui2/models/Stable-diffusion/colorful_v23.safetensors \
            --model_tuned "${file}"
    else
        echo "Skipping ${filename}, target file already exists."
    fi
done
