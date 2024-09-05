### This script is to create white background images with 
### object foreground from the original images defined by the masks. 
import os
import cv2
import numpy as np
import pdb
import tqdm

data_root = '/mnt/data0/minghz/repos/bundlenets/data'

for vision_asset in tqdm.tqdm(os.listdir(data_root)):
    asset_root = os.path.join(data_root, vision_asset)
    if os.path.isdir(asset_root):
        # create a new folder called 'rgb_masked' to store the white background images
        rgb_root = os.path.join(asset_root, 'rgb')
        mask_root = os.path.join(asset_root, 'masks')

        if not os.path.exists(mask_root):
            print(f'{mask_root} does not exist, skipping')
            continue

        rgb_masked_path = os.path.join(asset_root, 'rgb_masked')
        if os.path.exists(rgb_masked_path):
            list_of_files = os.listdir(rgb_masked_path)
            if len(list_of_files) > 0:
                continue

        os.makedirs(rgb_masked_path, exist_ok=True)

        for img_name in tqdm.tqdm(os.listdir(rgb_root)):
            img_path = os.path.join(rgb_root, img_name)
            mask_path = os.path.join(mask_root, img_name)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_bg = cv2.bitwise_not(mask)

            # create a white background image
            white_bg = np.ones_like(img) * 255

            img1_bg = cv2.bitwise_and(white_bg, white_bg, mask=mask_bg)
            img2_fg = cv2.bitwise_and(img, img, mask=mask)

            img_masked = cv2.add(img1_bg, img2_fg)
            cv2.imwrite(os.path.join(rgb_masked_path, img_name), img_masked)
            

