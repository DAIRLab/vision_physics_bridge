"""Script to visualize and possibly create manual mask annotations for XMem to
generate the remaining masks for BundleSDF datasets."""

import click
import cv2
import os
import os.path as op
import matplotlib.pyplot as plt
import pdb

import file_utils


def inspect_mask_on_image(rgb_img, mask_img):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(rgb_img)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(rgb_img)
    ax2.imshow(mask_img, alpha=0.5, cmap='gray')
    return fig



@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2-3; encodes " + \
                   "system and tosses.")
@click.option('--clear-data/--keep-data',
              default=False,
              help="whether to clear data folder before regenerating.")

def main_command(vision_asset: str, clear_data: bool):
    print(f'Viewing masks for {vision_asset}.')

    # Get the RGB and mask directories -- use the new RGB directory and the old
    # masks directory.
    rgb_dir = file_utils.bundlesdf_video_rgb_dir(vision_asset)
    mask_dir = op.join(
        file_utils.bundlesdf_video_dir(f'{vision_asset}_ORIGINAL'),
        'Annotations'
    )

    # Iterate over the existing mask files so they can be inspected.
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    for mask_file in mask_files:
        mask_path = op.join(mask_dir, mask_file)
        rgb_path = op.join(rgb_dir, mask_file)
        assert op.exists(rgb_path), f'Missing RGB file: {rgb_path}.'
        frame_num = mask_file.split('.')[0]
        print(f'Inspecting image {frame_num}.')

        # Load the RGB and mask images.
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Show the RGB and mask images.
        fig = inspect_mask_on_image(rgb_img, mask_img)
        fig.suptitle(f'{vision_asset} frame {frame_num}')
        plt.show()

    print(f'Finished inspecting masks for {vision_asset}.')


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
