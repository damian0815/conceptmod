#!/bin/bash

cd /sd-scripts
source lora/bin/activate
# Set the directory path
dir="/stable-diffusion-webui/models/Stable-diffusion/conceptmod"
mkdir "$dir"
mv -v models/*/*.ckpt "$dir"

# Loop through all files in the directory
for file in "${dir}"*; do
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
            --model_org $model \
            --model_tuned "${file}"
    else
        echo "Skipping ${filename}, target file already exists."
    fi
done


#!/bin/bash

# Set the default model path
default_model_path="/workspace/stable-diffusion-webui/models/Stable-diffusion/SDv1-5.ckpt"

# Define $model as the first argument passed in or use the default model path
model="${1:-$default_model_path}"

# Set the directory path
dir="/stable-diffusion-webui/models/Stable-diffusion/conceptmod"
mkdir "$dir"
mv -v models/*/*.ckpt "$dir"

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

