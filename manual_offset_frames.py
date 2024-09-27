import os
import shutil
import argparse

"""
This script is used to manually offset the starting frame of the dataset. 
Giving a starting frame, create new folders for rgb, depth, mask, and annotated_poses. 
"""
def create_new_folders(base_path, start_frame):
    folders = ['rgb_occluded_template', 'depth', 'masks_occluded_template', 'annotated_poses']
    for folder in folders:
        new_folder_path = os.path.join(base_path, f"{folder}_offset_{start_frame}")
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Created folder: {new_folder_path}")

def copy_files_with_offset(base_path, start_frame):
    folders = ['rgb_occluded_template', 'depth', 'masks_occluded_template', 'annotated_poses']
    for folder in folders:
        original_folder_path = os.path.join(base_path, folder)
        new_folder_path = os.path.join(base_path, f"{folder}_offset_{start_frame}")
        
        for filename in sorted(os.listdir(original_folder_path)):
            frame_number = int(filename.split('.')[0])
            ext = filename.split('.')[-1]
            start_id = start_frame -1 if folder == 'annotated_poses' else start_frame
            if frame_number >= start_id:
                new_frame_number = frame_number - start_frame + 1
                new_filename = f"{new_frame_number:04d}.{ext}"
                shutil.copy(os.path.join(original_folder_path, filename), os.path.join(new_folder_path, new_filename))
                print(f"Copied {filename} to {new_filename}")

def main():
    parser = argparse.ArgumentParser(description="Manually offset the starting frame of the dataset.")
    parser.add_argument('base_path', type=str, help='Base path of the dataset')
    parser.add_argument('start_frame', type=int, help='Starting frame to offset from')
    args = parser.parse_args()

    create_new_folders(args.base_path, args.start_frame)
    copy_files_with_offset(args.base_path, args.start_frame)

if __name__ == "__main__":
    main()