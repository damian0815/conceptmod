#!/bin/bash

# Check if the number of arguments is not equal to 1
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <checkpoint>"
    exit 1
fi

cd /workspace/conceptmod/sd-scripts
source lora/bin/activate
# Set the directory path
dir="/workspace/stable-diffusion-webui/models/Stable-diffusion/conceptmod"
mkdir -p "$dir"
mv -v /workspace/conceptmod/models/*/*.ckpt "$dir"

# Loop through all files in the directory
for file in "${dir}/"*; do
    # Extract the filename without extension
    filename=$(basename -- "${file%.*}")

    # Remove special characters from the filename
    clean_filename=$(echo "${filename}" | tr -d '|~#:')

    # Set the target file path
    target="$dir/${clean_filename}.safetensors"

    # Check if the target file exists; if it does, skip the command
    if [ ! -f "${target}" ]; then
        # Run the command with the required arguments
        python3 networks/extract_lora_from_models.py \
            --save_precision fp16 \
            --save_to "${target}" \
            --model_org "${model}" \
            --model_tuned "${file}"
    else
        echo "Skipping ${filename}, target file already exists."
    fi
done

