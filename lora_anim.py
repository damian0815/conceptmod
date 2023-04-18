import requests
import glob
import argparse
import imageio
import torch
import random
import base64
import os
import numpy as np
import io
from PIL import Image
import os
import json

import cv2
import ImageReward as reward
from datasets import load_dataset
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, vfx
from moviepy.video.fx import fadein, fadeout

model = None
def score_image(prompt, fullpath):
    global model
    if model is None:
        model = reward.load("ImageReward-v1.0")
    with torch.no_grad():
        return model.score(prompt, fullpath)

folder = "anim"
for filename in os.listdir(folder):
    if filename.endswith(".png"):
        os.unlink(os.path.join(folder, filename))

seed = random.SystemRandom().randint(0, 2**32-1)
dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
def generate_image(prompt, negative_prompt, lora):
    url = "http://192.168.0.180:7777/sdapi/v1/txt2img"
    headers = {"Content-Type": "application/json"}
    prompt_ = prompt.replace("LORAVALUE",  "{:.14f}".format(lora))

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
        #"negative_prompt": "weird image.",
        "negative_prompt": negative_prompt,
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

def optical_flow(image1, image2):
    gray1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Compute the magnitude of the optical flow vectors
    magnitude = np.sqrt(np.sum(flow**2, axis=2))
    
    # Calculate the average magnitude of the flow vectors
    avg_magnitude = np.mean(magnitude)
    
    return avg_magnitude

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

def compare(image1, image2):
    """Calculate the mean squared error between two images."""
    return np.mean((np.array(image1) - np.array(image2)) ** 2)

def compare(image1, image2):
    #return calculate_ssim(image1,image2)
    return optical_flow(image1, image2)


def find_optimal_lora(prompt, negative_prompt, prev_lora, target_lora, prev_image, max_compare, tolerance):
    lo, hi = prev_lora, target_lora
    target_image = generate_image(prompt, negative_prompt, target_lora)
    if compare(target_image, prev_image) < max_compare:
        print("Closeness 1", compare(target_image, prev_image), "lo", lo, "hi", hi)
        return hi, target_image

    mid = hi
    mid_image = target_image
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        mid_image = generate_image(prompt, negative_prompt, mid)

        print(" - Closeness  - ", compare(mid_image, prev_image), "lo", lo, "hi", hi)
        if compare(prev_image, mid_image) > max_compare:
            hi = mid
        else:
            print("Closeness 2", compare(mid_image, prev_image), "lo", lo, "hi", hi)
            return mid, mid_image
    print("TOLERANCE", lo, hi, " - tol", tolerance)

    return mid, mid_image

def find_images(prompt, negative_prompt, lora_start, lora_end, steps, max_compare=1000, tolerance=2e-14):
    images = []
    lora_values = np.linspace(lora_start, lora_end, steps)


    # Create the "anim" directory if it doesn't exist
    os.makedirs("anim", exist_ok=True)
    prev_image = generate_image(prompt, negative_prompt, lora_start)
    prev_image.save(os.path.join("anim", f"image_0000.png"))
    images.append(prev_image)
    image_idx = 1
    current_image = prev_image

    for i, target_lora in enumerate(lora_values[1:]):
        optimal_lora = lora_values[i]

        while not np.isclose(optimal_lora, target_lora, rtol=tolerance, atol=tolerance):
            prev_image = current_image
            optimal_lora, current_image = find_optimal_lora(prompt, negative_prompt, optimal_lora, target_lora, prev_image, max_compare, tolerance)
            images.append(current_image)
            current_image.save(os.path.join("anim", f"image_{image_idx:04d}.png"))
            image_idx += 1
        print(compare(current_image, prev_image), "Done")

    return images

def find_best_seed(prompt, negative_prompt, num_seeds=10, steps=2, max_compare=20.0, lora_start=0.0, lora_end=1.0):
    global seed
    best_seed = None
    best_score = float('-inf')
    bscore1 = None
    bscore2 = None

    for _ in range(num_seeds):
        seed = random.SystemRandom().randint(0, 2**32-1)

        # Generate images with steps=2 and max_compare=-0.0
        generated_images = find_images(prompt, negative_prompt, lora_start, lora_end, steps, max_compare)

        # Score the images and sum the scores
        score1 = score_image(prompt, "anim/image_0000.png")
        score2 = score_image(prompt, "anim/image_0001.png")
        total_score = score1 + score2

        # Print the scores for debugging
        print(f"Seed: {_}, Score1: {score1}, Score2: {score2}, Total: {total_score}")

        # Update the best seed and score if the current total score is better
        if total_score > best_score:
            best_seed = seed
            best_score = total_score
            bscore1 = score1
            bscore2 = score2

    return best_seed, best_score, bscore1, bscore2

def main():
    parser = argparse.ArgumentParser(description='Generate images for a video between lora_start and lora_end')
    parser.add_argument('-s', '--lora_start', type=float, required=True, help='Start lora value')
    parser.add_argument('-e', '--lora_end', type=float, required=True, help='End lora value')
    parser.add_argument('-m', '--max_compare', type=float, default=1000.0, help='Maximum mean squared error (default: 1000)')
    parser.add_argument('-n', '--steps', type=int, default=32, help='Min frames in output animation')
    parser.add_argument('-sd', '--num_seeds', type=int, default=10, help='number of seeds to search')
    parser.add_argument('-t', '--tolerance', type=float, default=2e-14, help='Tolerance for optimal lora (default: 2e-14)')
    parser.add_argument('-l', '--lora', type=str, required=True, help='Lora to use')
    parser.add_argument('-lp', '--lora_prompt', type=str, default="", help='Lora prompt')
    parser.add_argument('-np', '--negative_prompt', type=str, default="weird image.", help='negative prompt')
    parser.add_argument('-pp', '--prompt_addendum', type=str, default="<lora:weird_image.:1.0>", help='add this to the end of prompts')
    parser.add_argument('-p', '--prompt', type=str, default=None, help='Prompt, defaults to random from Gustavosta/Stable-Diffusion-Prompts')
    args = parser.parse_args()


    prompt = args.prompt
    if prompt is None:
        prompt = random.choice(dataset['train']["Prompt"])
    lora_prompt = "<lora:"+args.lora+":LORAVALUE>"+args.prompt_addendum+" "+args.lora_prompt
    prompt = (prompt + ' ' + lora_prompt).strip()

    # Find the best seed
    best_seed, best_score, score1, score2 = find_best_seed(prompt, args.negative_prompt, num_seeds=args.num_seeds, steps=2, max_compare=1000, lora_start=args.lora_start, lora_end=args.lora_end)
    print(f"Best seed: {best_seed}, Best score: {best_score}")

    # Now generate images with the best seed, compare=-0.77, and steps=32
    global seed
    seed = best_seed  # Set the best seed as the current seed
    generated_images = find_images(prompt, args.negative_prompt, args.lora_start, args.lora_end, args.steps, args.max_compare)
    # Create an animated movie
    fps = len(generated_images)//5
    video_index = create_animated_movie("anim", "v4", fps=fps)

    # Save details to a JSON file
    details = {
        "seed": best_seed,
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "prompt_addendum": args.prompt_addendum,
        "lora": args.lora,
        "lora_prompt": args.lora_prompt,
        "lora_start": args.lora_start,
        "lora_end": args.lora_end,
        "score1": score1,
        "score2": score2,
        "best_score": best_score
    }

    with open(f"v4/{video_index}.json", "w") as f:
        json.dump(details, f)

# Function to create an animated movie
def create_animated_movie(images_folder, output_folder, fps=15):
    os.makedirs(output_folder, exist_ok=True)

    # Create a list of filepaths for the images in the "anim" directory
    image_filepaths = [os.path.join(images_folder, t) for t in sorted(os.listdir(images_folder))]

    # Create a clip from the image sequence
    clip = ImageSequenceClip(image_filepaths, fps=fps)  # Adjust fps value to control animation speed

    # Create a 2-second end frame
    end_frame = ImageSequenceClip([image_filepaths[-1]], fps=fps)
    end_frame = end_frame.set_duration(2)  # Set the duration of the end frame to 2 seconds

    # Create a fade out to black effect
    fade_out = vfx.fadeout(end_frame, 0.5)  # 1-second fade-out duration

    start_frame = ImageSequenceClip([image_filepaths[0]], fps=fps)
    # Create a fade in from black to the first frame
    fade_in = vfx.fadein(start_frame, 0.5)

    # Concatenate the clips, fade out, and fade in
    final_clip = concatenate_videoclips([clip, end_frame, fade_out, fade_in])
    # Find the next available video file index
    video_index = 1
    while os.path.exists(f"{output_folder}/{video_index}.mp4"):
        video_index += 1

    print("Writing mp4", len(image_filepaths), "images to", f"{output_folder}/{video_index}.mp4")
    # Save the clip as a high-quality GIF
    final_clip.write_videofile(f"{output_folder}/{video_index}.mp4", codec="libx264", audio=False)
    return video_index

if __name__ == '__main__':
    main()
