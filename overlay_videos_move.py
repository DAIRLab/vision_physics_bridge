"""This script moves the mesh overlay videos from the output directory to the
input directory of the contactnets pipeline.
"""

import file_utils
import shutil
import os.path as op

assets = [
    'oatly_2', 'oatly_3', 'oatly_4', 'oatly_5', 'oatly_6', 'cardboard_1', 
    'cardboard_2', 'cardboard_3', 'cardboard_4', 'cardboard_5', 'cardboard_6', 
    'cardboard_7', 'milk_1', 'milk_2', 'milk_3', 'milk_4', 'milk_5', 'milk_6', 'milk_7', 
    'styrofoam_1', 'styrofoam_2', 'styrofoam_3', 'styrofoam_4', 'styrofoam_5', 
    'styrofoam_6', 'toblerone_1', 'toblerone_2', 'toblerone_3', 'toblerone_4', 
    'toblerone_5', 'egg_1', 'egg_2', 'egg_3', 'egg_4', 'egg_5', 'egg_6', 
    'bakingbox_1'
]
assets = ['robotocc_'+ asset for asset in assets]
tracking_bundlesdf_id ='00'
nerf_bundlesdf_id = '00'
cycle_iteration = 1
gt_mesh = False
pll_id = '00'
for asset in assets:
    output_file = file_utils.inspection_overlay_video_filepath(
        dataset=asset, tracking_bundlesdf_id=tracking_bundlesdf_id,
        nerf_bundlesdf_id=nerf_bundlesdf_id, pll_id=None,
        cycle_iteration=cycle_iteration, gt_mesh=gt_mesh
    )
    # if not op.exists(op.dirname(self.output_file)):
    #     os.makedirs(op.dirname(self.output_file))

    # Put the output video in the same directory as the pll inputs though 
    # pll does not need the videos. It is for easier visualization. 
    output_file_to_pll_input_dir = file_utils.contactnets_input_bsdf_overlay_video_path(
        asset, cycle_iteration, tracking_bundlesdf_id,
        nerf_bundlesdf_id, gt_mesh=gt_mesh)
    
    print(f'from {output_file}')
    print(f'to {output_file_to_pll_input_dir}')
    if not op.exists(output_file):
        print(f'file does not exist: {output_file}')
        continue
    shutil.move(output_file, output_file_to_pll_input_dir)