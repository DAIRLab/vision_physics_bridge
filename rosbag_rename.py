"""The collected rosbags are in paths like: 
rosbags/january_2025_robot_bags/test_2025-01-08-16-41-10/0_2025-01-08-16-42-01.bag.
Rename and move them all to the rosbags directory with names like raw_300.bag, raw_301.bag, etc. 
Copy the renaming information to the google doc (Robot experiments) for reference."""

import os
import shutil
from datetime import datetime

def get_size_and_date(file_path):
    stat = os.stat(file_path)
    size = stat.st_size
    mtime = stat.st_mtime
    return size, mtime

def format_size(size):
    for unit in ['B', 'K', 'M', 'G', 'T', 'P']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024

def format_date(mtime):
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%b %d %H:%M")

def main():
    base_dir = 'rosbags/january_2025_robot_bags'
    start_index = 300
    index = start_index

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.bag'):
                old_path = os.path.join(root, file)
                new_name = f"raw_{index}.bag"
                new_path = os.path.join('rosbags', new_name)

                size, mtime = get_size_and_date(old_path)
                size_str = format_size(size)
                date_str = format_date(mtime)

                shutil.move(old_path, new_path)
                print(f"{size_str} {date_str} {file} -> {new_name}")

                index += 1

if __name__ == "__main__":
    main()