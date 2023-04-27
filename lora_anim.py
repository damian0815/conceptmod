import requests
import math
import glob
import argparse
import time
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
from decimal import Decimal

import cv2
import ImageReward as reward
from datasets import load_dataset
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, vfx
from moviepy.video.fx import fadein, fadeout

# Add a cache dictionary to store generated images and their corresponding lora values
image_cache = {}

model = None
folder = None
def score_image(prompt, fullpath):
    global model
    if model is None:
        model = reward.load("ImageReward-v1.0").to("cuda:0")
    with torch.no_grad():
        return model.score(prompt, fullpath)


seed = random.SystemRandom().randint(0, 2**32-1)
dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
txt2imgurl = "http://192.168.0.180:7777/sdapi/v1/txt2img"

def generate_image(prompt, negative_prompt, lora):
    url = txt2imgurl
    headers = {"Content-Type": "application/json"}
    prompt_ = prompt.replace("LORAVALUE",  "{:.14f}".format(lora))
    nprompt_ = negative_prompt.replace("LORAVALUE",  "{:.14f}".format(lora))
    uid = prompt_+"_"+negative_prompt+"_"+str(seed)+"_"+"{:.14f}".format(lora)

    #global image_cache
    #image_cache={}
    # Check if the image exists in the cache
    if uid in image_cache:
        return image_cache[uid]

    data = {
        "seed": seed,
        "width": 512,
        "height": 512,
        "sampler_name": "Euler",
        "prompt": prompt_,
        "negative_prompt": nprompt_,
        "steps": 28
    }
    #print(" calling: ", prompt_)

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        r = response.json()
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",",1)[0])))

        image_cache[uid] = image
        return image
    else:
        print(f"Request failed with status code {response.status_code}")
        return generate_image(prompt, negative_prompt, lora)

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


def find_closest_cache_key():
    closest_lora = [[key, Decimal(key.split('_')[-1])] for key in image_cache.keys()]
    sorted_list = sorted(closest_lora, key=lambda x: x[1])

    if(len(sorted_list) == 0):
        return None
    return sorted_list[0][0]

def find_optimal_lora(prompt, negative_prompt, prev_lora, target_lora, prev_image, max_compare, tolerance, budget):
    lo, hi = prev_lora, target_lora
    lo = Decimal(lo)
    hi = Decimal(hi)
    assert hi > lo
    if budget <= 0:
        target_image = generate_image(prompt, negative_prompt, target_lora)
        del image_cache[find_closest_cache_key()]
        return hi, target_image

    # Check if there's a close cached lora value
    closest_key = find_closest_cache_key()
    if closest_key is not None:
        target_image = image_cache[closest_key]
        closest_lora = Decimal(closest_key.split('_')[-1])
        compare_result = compare(prev_image, target_image)

        if compare_result < max_compare:
            print("  found frame in cache", compare_result)
            # Add the target_image to images and remove it from the cache
            del image_cache[closest_key]
            return closest_lora, target_image
        hi = closest_lora

    if closest_key is None:
        target_image = generate_image(prompt, negative_prompt, target_lora)
        compare_result = compare(prev_image, target_image)
        if compare_result < max_compare:
            print("  found frame in target ", compare_result)
            # Add the target_image to images and remove it from the cache
            del image_cache[find_closest_cache_key()]
            return hi, target_image

    mid = hi
    mid_image = None
    while hi - lo > tolerance and budget > 0:
        mid = (lo + hi) / 2
        mid_image = generate_image(prompt, negative_prompt, mid)
        comparison = compare(prev_image, mid_image)

        if max_compare < comparison:
            print("  descend  -  lora ", mid, "compare", comparison)
            hi = mid
            budget-=1
        else:
            print("  found frame in bsearch", comparison)
            del image_cache[find_closest_cache_key()]
            return mid, mid_image
    print("  found tolerance frame, may not be smooth", lo, hi, hi - lo > tolerance, "budget?", budget, budget > 0)
    if mid_image is None:
        mid_image = generate_image(prompt, negative_prompt, mid)
    del image_cache[find_closest_cache_key()]

    return mid, mid_image

def is_sequential(arr):
    # Check if the array is empty or has only one element
    if len(arr) <= 1:
        return True

    # Iterate over the array and check if each element is one more than the previous one
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            print("Found nonsequential at ", i)
            return False

    # If we've reached this point, the array is sequential
    return True

def smooth(images, threshold=0.1, similarity_threshold=0.05):
    smooth_images = [images[0]]
    i = 1

    while i < len(images):
        distance = compare(images[i-1], images[i])

        if distance >= threshold:
            similar = False

            # Check for similar non-consecutive frames
            for j in range(i + 1, min(len(images), i+30)):
                distance_similarity = compare(images[i-1], images[j])

                if distance_similarity <= similarity_threshold:
                    similar = True
                    print(f"Removed frames {i} to {j-1} due to similar frames {i-1} and {j}")
                    i = j
                    break

            if not similar:
                smooth_images.append(images[i])
                i += 1
            else:
                i += 1

        else:
            print(f"Removed frame {i} due to low distance: {distance}")
            i += 1

    return smooth_images


def find_images(prompt, negative_prompt, lora_start, lora_end, steps, max_compare=1000, tolerance=2e-13, max_budget=120):
    images = []
    lora_values = np.linspace(float(lora_start), float(lora_end), steps)
    global image_cache


    # Create the folder directory if it doesn't exist
    prev_image = generate_image(prompt, negative_prompt, lora_start)
    del image_cache[find_closest_cache_key()]
    prev_image.save(os.path.join(folder, f"image_0000.png"))
    images.append(prev_image)
    image_idx = 1
    budget = max_budget-1
    current_image = prev_image
    series = []
    optimal_lora = lora_start

    for i, target_lora in enumerate(lora_values[1:]):

        while optimal_lora is None or not math.isclose(optimal_lora, target_lora, abs_tol=tolerance):
            prev_image = current_image
            optimal_lora, current_image = find_optimal_lora(prompt, negative_prompt, optimal_lora, target_lora, prev_image, max_compare, tolerance, budget)
            budget =  max_budget- len(images)-len(image_cache.keys()) - len(lora_values[i+1:])
            print(f"-> frame {image_idx:03d} from lora {optimal_lora:.10f} / {lora_end} budget {budget:3d} cache size {len(image_cache.keys()):2d}")
            if len(series) > 0:
                #print("  optimal", optimal_lora, " last ", series[-1], series[-1] <= optimal_lora)
                #if(series[-1] > optimal_lora)
                assert series[-1] <= optimal_lora
            series += [optimal_lora]
            images.append(current_image)
            current_image.save(os.path.join(folder, f"image_{image_idx:04d}.png"))
            image_idx += 1
            if budget <= 0:
                while(len(image_cache.keys()) > 0):
                    current_image = image_cache[find_closest_cache_key()]
                    del image_cache[find_closest_cache_key()]
                    images.append(current_image)
                    current_image.save(os.path.join(folder, f"image_{image_idx:04d}.png"))
                    image_idx += 1


            if not is_sequential(series):
                print("Failure in sequence detected!!.")
                print(series)
                assert False

    return images

def find_best_seed(prompt, negative_prompt, num_seeds=10, steps=2, max_compare=20.0, lora_start=0.0, lora_end=1.0):
    global seed
    global image_cache
    best_seed = None
    best_score = float('-inf')
    bscore1 = None
    bscore2 = None

    for _ in range(num_seeds):
        seed = random.SystemRandom().randint(0, 2**32-1)
        image_cache = {}

        # Generate images with steps=2 and max_compare=-0.0
        generated_images = find_images(prompt, negative_prompt, lora_start, lora_end, steps, max_compare)

        # Score the images and sum the scores
        score1 = score_image(prompt, folder + "/image_0000.png")
        score2 = score_image(prompt, folder + "/image_0001.png")*3
        c = -compare(generated_images[0], generated_images[1])/8.0
        #c = calculate_ssim(generated_images[0], generated_images[1])*2
        total_score = score1 + score2 + c
        print("Score 1:", score1, "Score 2", score2, "Comparison", c, "total score", total_score)

        # Print the scores for debugging
        #print(f"Seed: {_}, Score1: {score1}, Score2: {score2}, Total: {total_score}")

        # Update the best seed and score if the current total score is better
        if total_score > best_score:
            best_seed = seed
            best_score = total_score
            bscore1 = score1
            bscore2 = score2

    return best_seed, best_score, bscore1, bscore2

def main():
    global txt2imgurl
    global folder
    parser = argparse.ArgumentParser(description='Generate images for a video between lora_start and lora_end')
    parser.add_argument('-s', '--lora_start', type=Decimal, required=True, help='Start lora value')
    parser.add_argument('-e', '--lora_end', type=Decimal, required=True, help='End lora value')
    parser.add_argument('-m', '--max_compare', type=float, default=1000.0, help='Maximum mean squared error (default: 1000)')
    parser.add_argument('-n', '--steps', type=int, default=32, help='Min frames in output animation')
    parser.add_argument('-sd', '--num_seeds', type=int, default=10, help='number of seeds to search')
    parser.add_argument('-b', '--budget', type=int, default=120, help='budget of image frames')
    parser.add_argument('-t', '--tolerance', type=Decimal, default=2e-14, help='Tolerance for optimal lora (default: 2e-14)')
    parser.add_argument('-l', '--lora', type=str, required=True, help='Lora to use')
    parser.add_argument('--negative_lora', action='store_true', default=False)
    parser.add_argument('--reverse', action='store_true', default=False)
    parser.add_argument('-lp', '--lora_prompt', type=str, default="", help='Lora prompt')
    parser.add_argument('-np', '--negative_prompt', type=str, default="weird image.", help='negative prompt')
    parser.add_argument('-pp', '--prompt_addendum', type=str, default="<lora:weird_image.:1.0>", help='add this to the end of prompts')
    parser.add_argument('-p', '--prompt', type=str, default=None, help='Prompt, defaults to random from Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('-f', '--folder', type=str, default="anim", help='Working directory')
    parser.add_argument('-url', '--text_to_image_url', type=str, default="http://192.168.0.180:7777/sdapi/v1/txt2img", help='Url for text to image')
    args = parser.parse_args()

    txt2imgurl = args.text_to_image_url
    folder = args.folder
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            os.unlink(os.path.join(folder, filename))

    prompt = args.prompt
    if prompt is None:
        prompt = random.choice(dataset['train']["Prompt"])
    lora_prompt = ""
    negative_prompt = args.negative_prompt
    if args.negative_lora == False:
        lora_prompt += "<lora:"+args.lora+":LORAVALUE>"
    else:
        negative_prompt += "<lora:"+args.lora+":LORAVALUE>"
    lora_prompt+=args.prompt_addendum+" "+args.lora_prompt
    prompt = (prompt + ' ' + lora_prompt).strip()

    # Find the best seed
    best_seed, best_score, score1, score2 = find_best_seed(prompt, negative_prompt, num_seeds=args.num_seeds, steps=2, max_compare=1000, lora_start=args.lora_start, lora_end=args.lora_end)
    print(f"Best seed: {best_seed}, Best score: {best_score}")

    # Now generate images with the best seed, compare=-0.77, and steps=32
    global seed
    seed = best_seed  # Set the best seed as the current seed
    images = find_images(prompt, negative_prompt, args.lora_start, args.lora_end, args.steps, args.max_compare, args.tolerance, args.budget)
    #print("Smoothing frames. This may take a while (deleting repeat sequences")
    generated_images = smooth(images)
    print("Before smoothing:", len(images), "frames after:", len(generated_images), "frames")
    if len(images) != len(generated_images):
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                os.unlink(os.path.join(folder, filename))

        for i, image in enumerate(generated_images):
            image.save(os.path.join(folder, f"image_{i+1:04d}.png"))

    # Create an animated movie
    fps = len(generated_images)//5
    if(fps ==0):
        fps = 1
    print("Generated", len(generated_images), "fps", fps)

    if args.reverse:
        generated_images = reversed(generated_images)
    video_index = create_animated_movie(folder, "v4", fps=fps)

    # Save details to a JSON file
    details = {
        "seed": best_seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_addendum": args.prompt_addendum,
        "lora": args.lora,
        "lora_prompt": args.lora_prompt,
        "lora_start": float(args.lora_start),
        "lora_end": float(args.lora_end),
        "score1": score1,
        "score2": score2,
        "best_score": best_score
    }

    with open(f"v4/{video_index}.json", "w") as f:
        json.dump(details, f)

# Function to create an animated movie
def create_animated_movie(images_folder, output_folder, fps=15):
    os.makedirs(output_folder, exist_ok=True)

    # Create a list of filepaths for the images in the folder directory
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

