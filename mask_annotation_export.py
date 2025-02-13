"""This script downloads the mask annotations from Labelbox and saves them to a folder.
"""

import labelbox as lb
import urllib.request
from PIL import Image
import json
import os
import tqdm
import pdb

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHpqNmxlcGEwMGUyMDcyaTYzbm9ncTc2Iiwib3JnYW5pemF0aW9uSWQiOiJjbHpqNmxlcDEwMGUxMDcyaTJzNDY5Zm9hIiwiYXBpS2V5SWQiOiJjbHpqOTFzcHEwMHFuMDcwOWJvcTI1NmduIiwic2VjcmV0IjoiMjYzOTY1YjkyMGQzNTlhN2YwY2JkNDI0NWM0YjY2ZGQiLCJpYXQiOjE3MjI5OTg4ODUsImV4cCI6MjM1NDE1MDg4NX0.2DafjmOibD0svqaHPepTKuQLlO0JGvkbhBO0dqQBC0M'
client = lb.Client(api_key = API_KEY)
# project_key = 'clzj6q7mf00go07zlgjss7oj3'
project_key = 'cm5ylvf5904e7070g8y5dgd7d'   # search for 'projects' in the json file

mask_folder = 'mask_annotations_robotocc'
os.makedirs(mask_folder, exist_ok=True)
print(f'Saving masks to {mask_folder}')

# json_file = 'Export v2 project - Label_robot_object - 8_6_2024.ndjson'
# json_file = 'Export  project - Label_robotocc_object - 1_15_2025.ndjson'
# json_file = 'Export  project - Label_robotocc_object - 1_20_2025.ndjson'
json_file = 'Export  project - Label_robotocc_object - 1_27_2025.ndjson'
n_files = 0
with open(json_file) as f:
    for line in tqdm.tqdm(f):
        data = json.loads(line)
        image_name = data['data_row']['external_id']
        list_of_files = os.listdir(mask_folder)
        if image_name not in list_of_files and 'eggs' not in image_name:
            print(f'Processing {image_name}')
            n_files += 1
            mask_url = data['projects'][project_key]['labels'][0]['annotations']['objects'][0]['mask']['url']
            
            req = urllib.request.Request(mask_url, headers=client.headers)
            response = urllib.request.urlopen(req)
            mask = Image.open(response)
            mask.save(os.path.join(mask_folder, image_name))
            tqdm.tqdm.write(f'Saved mask for {image_name}')

print(f'Processed {n_files} files')