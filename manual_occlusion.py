### This file is to create artificial occlusions for a video sequence. 
### The occlusion should be added to both the RGB and mask images.

import cv2
import numpy as np
import os
import yaml
import click
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
import pdb
from bundlenets import file_utils

def create_occlusions(vision_asset: str, keyframes: List[Tuple[int, int, int, int, int]] = [], 
                      occlude_side: str = "bottom", template = None):
    # Load video
    
    video_path = file_utils.video_dir(vision_asset)\

    rgb_folder = os.path.join(video_path, "rgb")
    masks_folder = os.path.join(video_path, "masks")

    # Get list of image files
    rgb_files = sorted(os.listdir(rgb_folder))
    masks_files = sorted(os.listdir(masks_folder))

    # Get video properties
    frame_height, frame_width = cv2.imread(os.path.join(rgb_folder, rgb_files[0])).shape[:2]
    total_frames = len(rgb_files)
    
    if template is not None:
        adaptive_label = "template"
        occlude_label = f"{adaptive_label}"
    elif keyframes == []:
        adaptive_label = "adaptive"
        occlude_label = f"{occlude_side}_{adaptive_label}"
    else:
        adaptive_label = "keyframes"
        occlude_label = f"{occlude_side}_{adaptive_label}"

    # Create output directories
    output_dir = os.path.join(video_path, f"rgb_occluded_{occlude_label}")
    os.makedirs(output_dir, exist_ok=True)
    output_masks_dir = os.path.join(video_path, f"masks_occluded_{occlude_label}")
    os.makedirs(output_masks_dir, exist_ok=True)

    # Iterate through frames
    for frame_idx in tqdm(range(total_frames)):
        # Read frame
        rgb_path = os.path.join(rgb_folder, rgb_files[frame_idx])
        masks_path = os.path.join(masks_folder, masks_files[frame_idx])
        frame = cv2.imread(rgb_path)
        mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)

        if template is not None:
            # Resize template to frame size
            occlusion_mask = cv2.resize(template, (frame_width, frame_height))
            # Convert template to binary mask (0 or 255). Need to reverse the mask.
            occlusion_mask = cv2.threshold(occlusion_mask, 127, 255, cv2.THRESH_BINARY_INV)[1]

        elif keyframes == []:
            # Use the mask to determine the occlusion region
            # Calculate the center of geometry of the mask
            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Calculate the occlusion region
            if occlude_side == "bottom":
                start_x = 0
                start_y = cY
                start_width = frame_width
                start_height = frame_height - cY
            elif occlude_side == "top":
                start_x = 0
                start_y = 0
                start_width = frame_width
                start_height = cY
            elif occlude_side == "left":
                start_x = 0
                start_y = 0
                start_width = cX
                start_height = frame_height
            elif occlude_side == "right":
                start_x = cX
                start_y = 0
                start_width = frame_width - cX
                start_height = frame_height
            else:
                raise ValueError(f"Invalid occlude_side: {occlude_side}")
            
            # Initialize occlusion mask
            occlusion_mask = np.ones((frame_height, frame_width), dtype=np.uint8)*255
            # Add occlusion to mask
            occlusion_mask[start_y:start_y+start_height, start_x:start_x+start_width] = 0
        else:
            # Check if the occlusion is normalized
            normalized = False
            if keyframes != []:
                if all([x <= 1 for x in keyframes[0]]):
                    normalized = True

            # Interpolate occlusion between keyframes
            for i in range(len(keyframes) - 1):
                start_frame, start_x, start_y, start_width, start_height = keyframes[i]
                end_frame, end_x, end_y, end_width, end_height = keyframes[i + 1]

                # Find the keyframes that the current frame is between
                if frame_idx < start_frame:
                    continue
                elif frame_idx >= end_frame:
                    start_x, start_y, start_width, start_height = end_x, end_y, end_width, end_height
                

                if normalized:
                    start_x = int(start_x * frame_width) if start_x >= 0 else -1
                    start_y = int(start_y * frame_height) if start_y >= 0 else -1
                    start_width = int(start_width * frame_width) if start_width >= 0 else -1
                    start_height = int(start_height * frame_height) if start_height >= 0 else -1

                    end_x = int(end_x * frame_width) if end_x >= 0 else -1
                    end_y = int(end_y * frame_height) if end_y >= 0 else -1
                    end_width = int(end_width * frame_width) if end_width >= 0 else -1
                    end_height = int(end_height * frame_height) if end_height >= 0 else -1

                # Handle missing x or y
                if start_x == -1:
                    start_x = 0
                if start_y == -1:
                    start_y = 0

                if end_x == -1:
                    end_x = 0
                if end_y == -1:
                    end_y = 0

                # Handle missing width or height
                if start_width == -1:
                    start_width = frame_width - start_x
                if start_height == -1:
                    start_height = frame_height - start_y

                if end_width == -1:
                    end_width = frame_width - end_x
                if end_height == -1:
                    end_height = frame_height - end_y

                alpha = (frame_idx - start_frame) / (end_frame - start_frame)
                start_x = int(start_x + alpha * (end_x - start_x))
                start_y = int(start_y + alpha * (end_y - start_y))
                start_width = int(start_width + alpha * (end_width - start_width))
                start_height = int(start_height + alpha * (end_height - start_height))
                break
            
            # Initialize occlusion mask
            occlusion_mask = np.ones((frame_height, frame_width), dtype=np.uint8)*255
            # Add occlusion to mask
            occlusion_mask[start_y:start_y+start_height, start_x:start_x+start_width] = 0


        # Apply occlusion to frame
        frame = cv2.bitwise_and(frame, frame, mask=occlusion_mask)
        frame_mask = cv2.bitwise_and(mask, mask, mask=occlusion_mask)

        # Save occluded frame
        output_path = os.path.join(output_dir, rgb_files[frame_idx])
        cv2.imwrite(output_path, frame)

        # Save occluded mask
        output_mask_path = os.path.join(output_masks_dir, masks_files[frame_idx])
        cv2.imwrite(output_mask_path, frame_mask)

def find_mask_center_average_across_frames(vision_asset: str):
    video_path = file_utils.video_dir(vision_asset)
    masks_folder = os.path.join(video_path, "masks")

    # Get list of image files
    masks_files = sorted(os.listdir(masks_folder))

    # Get video properties
    frame_height, frame_width = cv2.imread(os.path.join(
        masks_folder, masks_files[0]), cv2.IMREAD_GRAYSCALE).shape[:2]
    total_frames = len(masks_files)

    # Initialize center of geometry
    cX = 0
    cY = 0
    count = 0

    # Iterate through frames
    for frame_idx in tqdm(range(total_frames)):
        # Read frame
        masks_path = os.path.join(masks_folder, masks_files[frame_idx])
        mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)

        # Use the mask to determine the occlusion region
        # Calculate the center of geometry of the mask
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        M = cv2.moments(mask)
        cX += int(M["m10"] / M["m00"])
        cY += int(M["m01"] / M["m00"])
        count += 1

    cX /= count
    cY /= count
    cX = int(cX)
    cY = int(cY)

    return cX, cY

def write_static_mask_config_to_file(vision_asset: str, cX: float, cY: float, occlude_side: str):
    """Write a static mask configuration to a file. 
    The configuration specifies a single occlusion that covers a certain side of the image.
    Write two keyframes (0 and 1). 
    The format is (frame_idx, start_x, start_y, width, height).
    """
    # Calculate the occlusion region
    if occlude_side == "bottom":
        keyframes = [
            [0, 0, cY, -1, -1],
            [1, 0, cY, -1, -1],
        ]
    elif occlude_side == "top":
        keyframes = [
            [0, 0, 0, -1, cY],
            [1, 0, 0, -1, cY],
        ]
    elif occlude_side == "left":
        keyframes = [
            [0, 0, 0, cX, -1],
            [1, 0, 0, cX, -1],
        ]
    elif occlude_side == "right":
        keyframes = [
            [0, cX, 0, -1, -1],
            [1, cX, 0, -1, -1],
        ]
    else:
        raise ValueError(f"Invalid occlude_side: {occlude_side}")
    config_path = os.path.join(
        "/mnt/data0/minghz/repos/bundlenets/cnets-data-generation/manual_mask_config", 
        f"{vision_asset}_{occlude_side}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(keyframes, f)

def write_static_mask_config_to_file_for_all_assets(occlude_side: str):
    objs = ["bakingbox", "cardboard", "crushedcan", "gallon", "greencan", "oatly", "pinkcan", \
            "stapler", "styrofoam", "egg", "napkin", "cube", "bottle", "half", "milk"]
    toss_ids = ["1", "2", "3", "4", "5"]
    for obj in objs:
        for toss_id in toss_ids:
            vision_asset = obj + "_" + toss_id
            cX, cY = find_mask_center_average_across_frames(vision_asset)
            write_static_mask_config_to_file(vision_asset, cX, cY, occlude_side)

@click.group()
def cli():
    pass

@cli.command('calc-static-mask-config')
@click.argument("occlude-side", type=str)
@click.option("--vision-asset", type=str, default=None)
def calc_static_mask_config(occlude_side: str, vision_asset: str):
    if vision_asset is not None:
        cX, cY = find_mask_center_average_across_frames(vision_asset)
        write_static_mask_config_to_file(vision_asset, cX, cY, occlude_side)
    else:
        write_static_mask_config_to_file_for_all_assets(occlude_side)

@cli.command('create-occlusions')
@click.argument('vision-asset',
              type=str)
# @click.argument("video_path", type=click.Path(exists=True))
@click.option("--use-config", is_flag=True, default=False)
@click.option("--use-template", is_flag=True, default=False)
@click.option("--occlude-side", type=str, default=None)
def create_occlusions_cmd(vision_asset: str, use_config: bool, use_template: bool, occlude_side: str):
    # Get video path
    keyframes_config_root = "/mnt/data0/minghz/repos/bundlenets/cnets-data-generation/manual_mask_config"
    templates_root = "/mnt/data0/minghz/repos/bundlenets/cnets-data-generation/manual_mask_template"
    assert not (use_config and use_template), "Cannot use both config and template."
    assert use_template or occlude_side is not None, "Must provide occlude_side if not using template."
    if use_config:
        keyframes_path = os.path.join(keyframes_config_root, f"{vision_asset}_{occlude_side}.yaml")
        # Load keyframes
        with open(keyframes_path, "r") as f:
            keyframes = yaml.safe_load(f)
        template = None
    elif use_template:
        template_path = os.path.join(templates_root, f"{vision_asset}.png")
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        keyframes = []
    else:
        keyframes = []
        template = None

    # Create occlusions
    create_occlusions(vision_asset, keyframes, occlude_side, template)

if __name__ == "__main__":
    cli()