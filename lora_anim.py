import requests
import random
import base64
import os
import numpy as np
import io
import json
from PIL import Image
import os

folder = "anim"
for filename in os.listdir(folder):
    if filename.endswith(".png"):
        os.unlink(os.path.join(folder, filename))

seed = random.SystemRandom().randint(0, 2**32-1)
print(seed, "seed")
prompt = "male harry potter, photo of gucci fashion, lips pursed, stern aloof look, profile picture, full torso and head in shot, fashion magazine photoshoot, fashionable hairstyle, cheekbones, hair fluffed and down"
def generate_image(lora):
    url = "http://192.168.0.180:7777/sdapi/v1/txt2img"
    headers = {"Content-Type": "application/json"}
    prompt_ = prompt + "<lora:gucci:" +"{:.14f}".format(lora*3.5)+"><lora:weird_image.:1.0>"

    data = {
        "seed": seed,
        "width": 512,
        "height": 512,
        "sampler_name": "DDIM",
        #"prompt": "male harry potter <lora:gucci:"+str(lora)+">, photo of gucci fashion, lips pursed, stern aloof look, profile picture, full torso and head in shot, fashion magazine photoshoot, fashionable hairstyle, cheekbones, hair fluffed and down",
        #"prompt": "male harry potter <lora:balenciaga:"+str(lora)+">, photo of balenciaga fashion, lips pursed, stern aloof look, profile picture, full torso and head in shot, fashion magazine photoshoot, fashionable hairstyle, cheekbones, hair fluffed and down",
        #"prompt": "hermoine as an auror in a coffee shop <lora:balenciaga:"+"{:.14f}".format(lora)+">, photo of balenciaga , lips pursed, stern aloof look, profile picture, full torso and head in shot, fashionable hairstyle, hair fluffed and down, <lora:weird_image.:1.0>",
        #"prompt": "hermoine as an auror in a coffee shop <lora:gucci:"+"{:.14f}".format(lora)+">, photo of gucci , lips pursed, stern aloof look, profile picture, full torso and head in shot, fashionable hairstyle, hair fluffed and down, <lora:weird_image.:1.0>",
        #"prompt": "ron weasley in quidditch gear, on the field <lora:gucci:"+"{:.14f}".format(lora)+">, photo of gucci , lips pursed, stern aloof look, profile picture, full torso and head in shot, fashionable hairstyle, hair fluffed and down, <lora:weird_image.:1.0>",
        #"prompt": "male bearded hagrid in great shape, chad, ripped, in a forest <lora:gucci:"+"{:.14f}".format(lora)+">, photo of gucci , lips pursed, stern aloof look, profile picture, full torso and head in shot, fashionable hairstyle, hair fluffed and down, <lora:weird_image.:1.0>",
        "prompt": prompt_,
        #"prompt": "male bearded hagrid in great shape, chad, ripped, in a forest <lora:balenciaga:"+"{:.14f}".format(lora)+">, photo of balenciaga , lips pursed, stern aloof look, profile picture, full torso and head in shot, fashionable hairstyle, hair fluffed and down, <lora:weird_image.:1.0>",
        "negative_prompt": "weird image.",
        #"negative_prompt": "female",
        "steps": 40
    }
    print(data)

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        r = response.json()
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",",1)[0])))
        return image
    else:
        print(f"Request failed with status code {response.status_code}")
        assert False

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    # Convert Pillow images to numpy arrays
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # If the images are RGB, convert them to grayscale
    if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
        img1_np = np.dot(img1_np, [0.2989, 0.5870, 0.1140])
    if len(img2_np.shape) == 3 and img2_np.shape[2] == 3:
        img2_np = np.dot(img2_np, [0.2989, 0.5870, 0.1140])

    # Calculate SSIM
    return -ssim(img1_np, img2_np)

def mse(image1, image2):
    """Calculate the mean squared error between two images."""
    return np.mean((np.array(image1) - np.array(image2)) ** 2)

def compare(image1, image2):
    return calculate_ssim(image1,image2)

def find_optimal_lora(prev_lora, target_lora, prev_image, max_mse, tolerance):
    lo, hi = prev_lora, target_lora
    target_image = generate_image(target_lora)
    if compare(target_image, prev_image) < max_mse:
        print("Closeness 1", compare(target_image, prev_image), "lo", lo, "hi", hi)
        return hi, target_image

    mid = hi
    mid_image = target_image
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        mid_image = generate_image(mid)

        print(" - Closeness  - ", compare(mid_image, prev_image), "lo", lo, "hi", hi)
        if compare(prev_image, mid_image) > max_mse:
            hi = mid
        else:
            print("Closeness 2", compare(mid_image, prev_image), "lo", lo, "hi", hi)
            return mid, mid_image
    print("TOLERANCE", lo, hi, " - tol", tolerance)

    return mid, mid_image

def find_images(lora_start, lora_end, steps, max_mse=1000, tolerance=0.001):
    images = []
    lora_values = np.linspace(lora_start, lora_end, steps)


    # Create the "anim" directory if it doesn't exist
    os.makedirs("anim", exist_ok=True)
    prev_image = generate_image(lora_start)
    prev_image.save(os.path.join("anim", f"image_0000.png"))
    images.append(prev_image)
    image_idx = 1
    current_image = prev_image

    for i, target_lora in enumerate(lora_values[1:]):
        optimal_lora = lora_values[i]
        tolerance = 1e-9

        while not np.isclose(optimal_lora, target_lora, rtol=tolerance, atol=tolerance):
            prev_image = current_image
            print("pre optimal", optimal_lora, "target", target_lora)
            optimal_lora, current_image = find_optimal_lora(optimal_lora, target_lora, prev_image, max_mse, tolerance)
            print("optimal", optimal_lora)
            images.append(current_image)
            current_image.save(os.path.join("anim", f"image_{image_idx:04d}.png"))
            image_idx += 1
        print(compare(current_image, prev_image), "Done")

    return images

# Example usage
lora_end = 1.0
lora_start = 0
steps = 2
max_mse = -0.0#175.0  # Adjust this value to control the smoothness of the animation
tolerance = 2e-14  # Adjust this value to control the tolerance for finding the optimal lora value

generated_images = find_images(lora_start, lora_end, steps, max_mse, tolerance)

