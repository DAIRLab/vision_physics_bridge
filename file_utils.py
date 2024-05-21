from copy import deepcopy
import datetime
from typing import Tuple, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from PIL import Image
import glob
import cv2
import glob
import json
import shutil
import os
import os.path as op
import re
import yaml
import rospy
import sys

import math_utils


DATA_GEN_DIR = op.dirname(op.realpath(__file__))
REPO_DIR = op.dirname(DATA_GEN_DIR)
PLL_DIR = op.join(REPO_DIR, 'dair_pll')

if PLL_DIR not in sys.path:
    sys.path.append(PLL_DIR)    # For importing dair_pll.
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)   # For importing bundlenets.

from dair_pll import file_utils as pll_file_utils
from bundlenets import file_utils as bsdf_file_utils


"""Perform some checks on the file structure."""
assert pll_file_utils.MAIN_DIR == PLL_DIR, f"Unexpected file structure; " \
    f"{PLL_DIR} and {pll_file_utils.MAIN_DIR} don't match."
assert bsdf_file_utils.BUNDLENETS_REPO_DIR == REPO_DIR, f"Unexpected file " \
    f"structure; {REPO_DIR} and {bsdf_file_utils.BUNDLENETS_REPO_DIR} don't " \
    f"match."


REALSENSE_CAMERA_NAME = 'cam0'
PROCESSING_YAML_FILE = op.join(DATA_GEN_DIR, 'assets', 'config.yaml')
TAGLESS_OBJECTS = ['bakingbox', 'burger', 'cardboard', 'chocolate', 'cream',
                   'croc', 'crushedcan', 'duck', 'gallon', 'greencan', 'hotdog',
                   'icetray', 'mug', 'oatly', 'pinkcan', 'stapler', 'styrofoam',
                   'toothpaste']


"""Directory utilities."""
def assure_created(directory: str) -> str:
    """Wrapper to put around directory paths which ensure their existence.
    Reuses this implementation in PLL repo."""
    return pll_file_utils.assure_created(directory)

def get_pll_geometry_output_dir(system: str, cycle_iteration: int,
                                run_name: str):
    """Directory with PLL run's geometry outputs for BundleSDF training."""
    # Reconstruct the PLL storage name.
    data_asset = f'vision_{system}'
    pose_source = pose_source = 'tagslam_toss' if cycle_iteration == 0 else \
        f'bundlesdf_toss_iteration_{cycle_iteration}'
    storage_name = op.join(pll_file_utils.RESULTS_DIR, data_asset, pose_source)

    # Get the geometry output directory from the PLL storage.
    pll_geom_output_dir = pll_file_utils.geom_for_bsdf_dir(
        storage_name, run_name)
    assert os.listdir(pll_geom_output_dir), \
        f'No output found at {pll_geom_output_dir}'
    
    return pll_geom_output_dir


"""Directories."""
def bundlesdf_run_results_dir(dataset: str, cycle_iteration: int,
                              bundlesdf_id: str) -> str:
    """BundleSDF's results directory for a particular run and dataset."""
    bundlesdf_result_dir = bsdf_file_utils.results_dir(
        dataset=dataset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id
    )
    assert op.exists(bundlesdf_result_dir), f'Requires ' + \
        f'{bundlesdf_result_dir} to exist but not found.'
    return bundlesdf_result_dir

def bundlesdf_pose_dir(dataset: str, cycle_iteration: int, bundlesdf_id: str
                       ) -> str:
    """BundleSDF's output pose directory for a particular dataset.  Contains
    XXXX.txt files for every pose."""
    bundlesdf_result_dir = bsdf_file_utils.results_dir(
        dataset=dataset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id
    )
    path = op.join(bundlesdf_result_dir, 'ob_in_cam')
    assert op.exists(path), f'Requires {path} to exist but not found.'
    return path

def bundlesdf_nerf_results_dir(
        dataset: str, cycle_iteration: int, tracking_bundlesdf_id: str,
        nerf_bundlesdf_id: str) -> str:
    """BundleSDF's NeRF results directory for a particular run and dataset.
    Gets the NeRF results from the NeRF run associated with a tracking run."""
    bundlesdf_result_dir = bundlesdf_run_results_dir(
        dataset=dataset, cycle_iteration=cycle_iteration,
        bundlesdf_id=tracking_bundlesdf_id
    )
    nerf_results_dir = bsdf_file_utils.nerf_results_subdir(
        out_folder=bundlesdf_result_dir, bundlesdf_run_id=nerf_bundlesdf_id,
        create=False
    )
    assert op.exists(nerf_results_dir), f'Requires {nerf_results_dir} to ' + \
        f'exist but not found.'
    return nerf_results_dir

def bundlesdf_annotated_poses_dir(dataset: str, create: bool = False) -> str:
    """BundleSDF's input annotated pose directory for a particular dataset.
    Contains 0000.txt file with the first TagSLAM origin's pose represented in
    camera frame."""
    bundlesdf_video_dir = bsdf_file_utils.top_video_dir()
    path = op.join(bundlesdf_video_dir, dataset, 'annotated_poses')
    if create:
        return assure_created(path)
    assert op.exists(path), f'Requires {path} to exist but not found.'
    return path

def bundlesdf_run_associated_pll_run_id(dataset: str, cycle_iteration: int,
                                        bundlesdf_id: str) -> str:
    """Get the PLL run ID associated with a particular BundleSDF run."""
    return bsdf_file_utils.pll_run_id_from_bundlesdf_id(
        dataset=dataset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id)

def bundlesdf_geometry_dir(dataset: str, cycle_iteration: int, pll_run_id: str,
                           create: bool = True) -> str:
    """BundleSDF's geometry directory for a particular experiment."""
    geom_dir = bsdf_file_utils.geometry_dir(dataset, cycle_iteration,
                                            pll_run_id, check_exists=False)
    if create:
        return assure_created(geom_dir)
    return geom_dir

def bundlesdf_video_dir(dataset: str, check_exists: bool = False) -> str:
    """BundleSDF's video directory for a particular experiment."""
    return bsdf_file_utils.video_dir(dataset, check_exists=check_exists)

def cnets_data_gen_dataset_dir(dataset: str, check_exists: bool = False) -> str:
    """Directory for a particular dataset's data generation."""
    path = op.join(DATA_GEN_DIR, 'dataset', dataset)
    if check_exists:
        assert op.exists(path), f'Requires {path} to exist but not found.'
    return path

def tagslam_pose_dir(dataset: str, check_exists: bool = False) -> str:
    """TagSLAM's pose directory for a particular dataset.  Contains tagslam.txt
    file and XXXX.txt files for every timestamp corresponding to the times in
    the first column of tagslam.txt."""
    dataset_dir = cnets_data_gen_dataset_dir(
        dataset=dataset, check_exists=check_exists)
    path = op.join(dataset_dir, 'tagslam_poses')
    if check_exists:
        assert op.exists(path), f'Requires {path} to exist but not found.'
    return path

def synchronized_tagslam_pose_dir(dataset: str, check_exists: bool = False
                                  ) -> str:
    """TagSLAM's synchronized pose directory for a particular dataset.  Contains
    synced_tagslam.txt file and XXXX.txt files for every timestamp corresponding
    to bundlesdf_timestamps.txt in this directory's parent directory."""
    path = op.join(
        DATA_GEN_DIR, 'dataset', dataset, 'synchronized_tagslam_poses')
    if check_exists:
        assert op.exists(path), f'Requires {path} to exist but not found.'
    return path

def contactnets_input_dir(object: str) -> str:
    """ContactNets' input directory for a particular experiment."""
    return assure_created(
        op.join(pll_file_utils.ASSETS_DIR, f'vision_{object}')
    )

def contactnets_input_dir_tagslam(
        dataset: str, full: bool = True, create: bool = True) -> str:
    """ContactNets' input directory for a particular experiment from TagSLAM."""
    traj_subdir = 'full' if full else 'toss'
    track_subdir = '' if full else 'tagslam'
    object = dataset.split('_')[0]
    cn_input_dir = op.join(
        contactnets_input_dir(object), dataset, traj_subdir, track_subdir)

    if create:
        return assure_created(cn_input_dir)
    return cn_input_dir

def contactnets_input_dir_bundlesdf(
        dataset: str, iteration: int, bundlesdf_id: str, full: bool = True,
        create: bool = True
) -> str:
    """ContactNets' input directory for a particular experiment from a
    particular iteration of BundleSDF."""
    traj_subdir = 'full' if full else 'toss'
    iteration_subdir = f'bundlesdf_iteration_{iteration}'
    id_subdir = '' if full else bundlesdf_id
    object = dataset.split('_')[0]

    cn_input_dir = op.join(
        contactnets_input_dir(object), dataset, traj_subdir, iteration_subdir,
        id_subdir)

    if create:
        return assure_created(cn_input_dir)
    return cn_input_dir

def contactnets_input_geometry_dir(
        dataset: str, iteration: int, bundlesdf_id: str,
        create: bool = True) -> str:
    """ContactNets' input directory for geometry information from BundleSDF."""
    object = dataset.split('_')[0]
    pll_asset_subdirs = op.join(f'vision_{object}', dataset)
    geom_for_pll_dir = pll_file_utils.geom_for_pll_dir(
        pll_asset_subdirs, bundlesdf_id, iteration, check_exists=False)
    if create:
        return assure_created(geom_for_pll_dir)
    return geom_for_pll_dir

def contactnets_output_dir(dataset: str, cycle_iteration: int, pll_id: str
                           ) -> str:
    """PLL's geometry output directory for a particular experiment."""
    object = dataset.split('_')[0]
    results_dir = op.join(
        pll_file_utils.RESULTS_DIR, f'vision_{object}', dataset)
    subdir = 'tagslam' if cycle_iteration <= 0 else \
        f'bundlesdf_iteration_{cycle_iteration}'
    output_dir = op.join(results_dir, subdir, pll_id)
    assert op.exists(output_dir), f'PLL run results folder {output_dir} ' + \
        f'not exist.'
    return output_dir

def contactnets_output_geometry_dir(dataset: str, cycle_iteration: int,
                                    pll_id: str) -> str:
    """PLL's geometry output directory for a particular experiment."""
    run_results_dir = contactnets_output_dir(dataset, cycle_iteration, pll_id)
    output_geom_dir = op.join(run_results_dir,
                              pll_file_utils.BSDF_SUBFOLDER_NAME)

    # Known at this point that the parent results folder for this particular
    # run exists, since that is checked in the contactnets_output_dir function.
    assert op.exists(output_geom_dir), f'Looking for PLL geometry outputs ' + \
        f'in {output_geom_dir} but does not exist (however the parent ' + \
        f'directory {op.dirname(output_geom_dir)} does).'

    return output_geom_dir

def table_height_calibration_dir() -> str:
    """Directory for the point cloud processing output plots."""
    return assure_created(op.join(DATA_GEN_DIR, 'table_calibration'))

def bundlesdf_video_rgb_dir(dataset: str, check_exists: bool = True) -> str:
    """The BundleSDF input directory for RGB images for a particular dataset."""
    return bsdf_file_utils.video_rgb_dir(
        dataset, check_parent_exists=check_exists)

def bundlesdf_video_depth_dir(dataset: str, check_exists: bool = True) -> str:
    """The BundleSDF input directory for RGB images for a particular dataset."""
    return bsdf_file_utils.video_depth_dir(
        dataset, check_parent_exists=check_exists)

def bundlesdf_video_mask_dir(dataset: str, check_exists: bool = True) -> str:
    """The BundleSDF input directory for RGB images for a particular dataset."""
    return bsdf_file_utils.video_mask_dir(
        dataset, check_parent_exists=check_exists)


"""Manual inspection directories."""
def inspection_dir() -> str:
    """When running BundleSDF-to-PLL conversion, there are several result
    inspection files that can be created to manually inspect the performance.
    Put all of these files in a head directory at cnets-data-generation/
    consolidated_results/."""
    return assure_created(op.join(DATA_GEN_DIR, 'consolidated_results'))

def inspection_overlay_video_filepath(
        dataset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> str:
    """The directory for all overlay videos."""
    overlay_video_dir = op.join(inspection_dir(), 'overlay_videos')

    if tracking_bundlesdf_id.startswith('bundlesdf_id_'):
        tracking_bundlesdf_id = tracking_bundlesdf_id[13:]
    if nerf_bundlesdf_id.startswith('bundlesdf_id_'):
        nerf_bundlesdf_id = nerf_bundlesdf_id[13:]

    now = datetime.datetime.now()
    date_str = now.strftime('%m%d')
    filename = f'{date_str}_{dataset}_{tracking_bundlesdf_id}_' + \
        f'{nerf_bundlesdf_id}_{cycle_iteration}.mp4'

    return op.join(overlay_video_dir, filename)

def inspection_keyframe_overlay_image_dir(
        dataset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> str:
    """The directory for a NeRF run's optimized keyframe pose overlays."""
    parent_dir = op.join(inspection_dir(), 'keyframe_overlays')

    if tracking_bundlesdf_id.startswith('bundlesdf_id_'):
        tracking_bundlesdf_id = tracking_bundlesdf_id[13:]
    if nerf_bundlesdf_id.startswith('bundlesdf_id_'):
        nerf_bundlesdf_id = nerf_bundlesdf_id[13:]

    now = datetime.datetime.now()
    date_str = now.strftime('%m%d')
    child_dir_name = f'{date_str}_{dataset}_{tracking_bundlesdf_id}_' + \
        f'{nerf_bundlesdf_id}_{cycle_iteration}'

    return assure_created(op.join(parent_dir, child_dir_name))

def inspection_trajectory_plots_dir_and_prefix(
        dataset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> str:
    """The directory for all trajectory plots."""
    traj_dir = assure_created(op.join(inspection_dir(), 'trajectories'))

    if tracking_bundlesdf_id.startswith('bundlesdf_id_'):
        tracking_bundlesdf_id = tracking_bundlesdf_id[13:]
    if nerf_bundlesdf_id.startswith('bundlesdf_id_'):
        nerf_bundlesdf_id = nerf_bundlesdf_id[13:]

    now = datetime.datetime.now()
    date_str = now.strftime('%m%d')
    prefix = f'{date_str}_{dataset}_{tracking_bundlesdf_id}_' + \
        f'{nerf_bundlesdf_id}_{cycle_iteration}'

    return traj_dir, prefix

def inspection_mesh_filepath(
        dataset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> str:
    """The directory for all meshes."""
    mesh_dir = assure_created(op.join(inspection_dir(), 'meshes'))

    if tracking_bundlesdf_id.startswith('bundlesdf_id_'):
        tracking_bundlesdf_id = tracking_bundlesdf_id[13:]
    if nerf_bundlesdf_id.startswith('bundlesdf_id_'):
        nerf_bundlesdf_id = nerf_bundlesdf_id[13:]

    now = datetime.datetime.now()
    date_str = now.strftime('%m%d')
    filename = \
        f'{date_str}_{dataset}_{tracking_bundlesdf_id}_' + \
            f'{nerf_bundlesdf_id}_{cycle_iteration}_cleaned.obj'

    return op.join(mesh_dir, filename)

def inspection_3d_slice_video_filepath(
        dataset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> str:
    """The directory for all 3D slice plots."""
    slice_plot_dir = assure_created(
        op.join(inspection_dir(), '3d_slice_videos'))

    if tracking_bundlesdf_id.startswith('bundlesdf_id_'):
        tracking_bundlesdf_id = tracking_bundlesdf_id[13:]
    if nerf_bundlesdf_id.startswith('bundlesdf_id_'):
        nerf_bundlesdf_id = nerf_bundlesdf_id[13:]

    now = datetime.datetime.now()
    date_str = now.strftime('%m%d')
    filename = f'{date_str}_{dataset}_{tracking_bundlesdf_id}_' + \
        f'{nerf_bundlesdf_id}_{cycle_iteration}.mp4'

    return op.join(slice_plot_dir, filename)

def inspection_3d_slice_figure_filepath(
        dataset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> str:
    """The directory for all 3D slice plots."""
    slice_plot_dir = assure_created(
        op.join(inspection_dir(), '3d_slice_figures'))

    if tracking_bundlesdf_id.startswith('bundlesdf_id_'):
        tracking_bundlesdf_id = tracking_bundlesdf_id[13:]
    if nerf_bundlesdf_id.startswith('bundlesdf_id_'):
        nerf_bundlesdf_id = nerf_bundlesdf_id[13:]

    now = datetime.datetime.now()
    date_str = now.strftime('%m%d')
    filename = \
        f'{date_str}_{dataset}_{tracking_bundlesdf_id}_' + \
            f'{nerf_bundlesdf_id}_{cycle_iteration}.fig.pickle'

    return op.join(slice_plot_dir, filename)


"""Evaluation directories and utilities."""
def evaluation_dir() -> str:
    """Directory for all evaluation results."""
    return assure_created(op.join(DATA_GEN_DIR, 'evaluation'))

def evaluation_subdir(
        dataset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> str:
    """Subdirectory for a particular experiment's evaluation."""
    eval_dir = evaluation_dir()

    if tracking_bundlesdf_id.startswith('bundlesdf_id_'):
        tracking_bundlesdf_id = tracking_bundlesdf_id[13:]
    if nerf_bundlesdf_id.startswith('bundlesdf_id_'):
        nerf_bundlesdf_id = nerf_bundlesdf_id[13:]

    now = datetime.datetime.now()
    date_str = now.strftime('%m%d')
    subdir = \
        f'{date_str}_{dataset}_{tracking_bundlesdf_id}_' + \
            f'{nerf_bundlesdf_id}_{cycle_iteration}'

    return assure_created(op.join(eval_dir, subdir))


"""Yaml file parsing utilities."""
def load_camera_extrinsics(object: str) -> Tuple[np.ndarray, np.ndarray]:
    """Given the toss type, return the camera extrinsics in terms of a camera
    translation and rotation."""

    # Camera extrinsics file depends on the toss type.
    camera_extrinsics_filename = f'realsense_pose_{object}.yaml'
    if object in ['milk', 'prism']:
        camera_extrinsics_filename = 'realsense_pose_milk_prism.yaml'
    if object == 'cube_hand':
        camera_extrinsics_filename = 'realsense_pose_cube_hand_60_3.yaml'
    if object in TAGLESS_OBJECTS:
        camera_extrinsics_filename = 'realsense_pose_tagless.yaml'
    camera_extrinsics_file = op.join(DATA_GEN_DIR, 'assets',
                                     camera_extrinsics_filename)

    # Load the data from the extrinsics file.
    with open(camera_extrinsics_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # Gather the position and rotation information.
    cam_pos_dict = data_loaded[REALSENSE_CAMERA_NAME]['pose']['position']
    cam_trans = np.array(
        [cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]
    ).reshape(-1, 1)
    
    cam_rot_dict = data_loaded[REALSENSE_CAMERA_NAME]['pose']['rotation']
    cam_rot_axis_angle = np.array(
        [cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']]
    ).reshape(-1, 1)
    
    return cam_trans, cam_rot_axis_angle

def get_camera_intrinsics_filepath(object: str) -> str:
    """Get the filepath for the camera intrinsics file."""
    filename = 'cam_K.txt'
    if object in TAGLESS_OBJECTS:
        filename = 'cam_K_tagless.txt'
    return op.join(DATA_GEN_DIR, 'assets', filename)

def load_camera_intrinsics(object: str, as_matrix: bool = False):
    """The camera intrinsics appear to be the same for every experiment.  They
    are stored in cnets-data-generation/cam_K.txt."""
    cam_K_file = get_camera_intrinsics_filepath(object=object)
    if as_matrix:
        return np.loadtxt(cam_K_file)

    with open(cam_K_file, 'r') as f:
        lines = f.readlines()

    lines = math_utils.extract_floats_from_camk(lines)
    fx, fy = float(lines[0][0]), float(lines[1][1])
    cx, cy = float(lines[0][2]), float(lines[1][2])

    return fx, fy, cx, cy

def load_toss_objects_from_yaml() -> List[str]:
    """Load the objects that have been tossed in the dataset."""
    with open(PROCESSING_YAML_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data['dataset'].keys()

def load_toss_numbers_from_object_in_yaml(object: str) -> dict:
    """Load the toss numbers for a particular object from the dataset."""
    with open(PROCESSING_YAML_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data['dataset'][object].keys()

def load_toss_time_from_yaml(object, toss_number, key, as_ros_time=True):
    start_time_data = load_field_from_yaml(object, toss_number, key)
    if as_ros_time:
        time = rospy.rostime.Time(secs=start_time_data['secs'],
                                  nsecs=start_time_data['nsecs'])
    else:
        time = start_time_data['secs'] + start_time_data['nsecs'] * 1e-9
    return time

def load_field_from_yaml(object, toss_number, key):
    with open(PROCESSING_YAML_FILE, 'r') as f:
        data = yaml.safe_load(f)
    toss_data = data['tosses'][object][toss_number]
    return toss_data[key]

def load_rosbag_number_from_yaml(object, toss_number, second_toss_number=None):
    with open(PROCESSING_YAML_FILE, 'r') as f:
        data = yaml.safe_load(f)
    toss_data = data['dataset'][object][toss_number]
    if second_toss_number is not None:
        assert toss_data == data['dataset'][object][second_toss_number], \
            f"Inconsistent data entries for tosses {toss_number} and " + \
            f"{second_toss_number}:  got {toss_data} and " + \
            f"{data['dataset'][object][second_toss_number]}."
    return toss_data

def load_body_frame_pos_from_yaml(object):
    with open(PROCESSING_YAML_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data['body_frame'][object]['pose']['position']

def load_body_frame_rot_from_yaml(object):
    with open(PROCESSING_YAML_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data['body_frame'][object]['pose']['rotation']

def load_bundlesdf_id_from_pll_json(pll_output_dir: str) -> str:
    """Load the BundleSDF ID associated with a PLL run from its configuration
    file stored in its output directory."""
    json_file = op.join(pll_output_dir, 'config.json')
    assert op.exists(json_file), f'Did not find {json_file}.'

    with open(json_file, 'r') as f:
        json_object = json.loads(f.read())

    bundlesdf_id = json_object['data_config']['bundlesdf_id']
    if bundlesdf_id[:13] != 'bundlesdf_id_':
        return f'bundlesdf_id_{bundlesdf_id}'
    return bundlesdf_id

def load_pll_id_from_bundlesdf_yml(bundlesdf_output_dir: str) -> str:
    """Load the PLL ID associated with a BundleSDF run from its NeRF
    configuration file stored in its output directory."""
    yml_file = op.join(bundlesdf_output_dir, 'config_nerf.yml')
    assert op.exists(yml_file), f'Did not find {yml_file}.'

    with open(yml_file, 'r') as f:
        data = yaml.safe_load(f)

    geometry_dir = data['geometry_dir']
    if geometry_dir == None:
        return None

    pll_id = op.basename(geometry_dir)
    assert pll_id.startswith('pll_id_'), f'Unexpected PLL ID {pll_id=}.'

    return pll_id

def load_table_z_height(object, toss_number) -> float:
    """Load the table height associated with an object and toss."""
    with open(table_calibration_yaml_filepath(), 'r') as f:
        data = yaml.safe_load(f)
    return data[object][toss_number]

def load_keyframe_indices_from_nerf_results_yml(
        dataset: str, cycle_iteration: int, bundlesdf_id: str) -> list:
    """Load the indices of all keyframes from a NeRF run."""
    run_results_dir = bundlesdf_run_results_dir(
        dataset, cycle_iteration, bundlesdf_id)

    # Get the highest numbered frame folder.
    highest_frame = 0
    for frame_folder in os.listdir(run_results_dir):
        if frame_folder.isdigit():
            highest_frame = max(highest_frame, int(frame_folder))

    # Load keyframes.yml in this highest numbered frame folder.
    keyframes_yml = op.join(run_results_dir, f'{highest_frame:04d}',
                            'keyframes.yml')
    with open(keyframes_yml, 'r') as f:
        data = yaml.safe_load(f)

    # Get a list of all the keyframe indices.
    keyframe_indices = []
    for keyframe_str in data.keys():
        keyframe_indices.append(int(keyframe_str.split('keyframe_')[-1]))

    return keyframe_indices

def load_optimized_keyframe_poses_from_nerf_results(
        dataset: str, cycle_iteration: int, tracking_bundlesdf_id: str,
        nerf_bundlesdf_id: str) -> np.ndarray:
    """While BundleSDF trains the NeRF model, it produces optimized poses for
    all of the keyframes.  This function loads these optimized poses from the
    NeRF results directory, converting them to the standard ob_in_cam format."""
    nerf_results_dir = bundlesdf_nerf_results_dir(
        dataset=dataset, cycle_iteration=cycle_iteration,
        tracking_bundlesdf_id=tracking_bundlesdf_id,
        nerf_bundlesdf_id=nerf_bundlesdf_id
    )
    keyframe_data = np.loadtxt(
        op.join(nerf_results_dir, 'poses_after_nerf.txt'))

    # The keyframe data is of size (4*n_keyframes, 4), where each group of 4
    # rows is a 4x4 homogeneous transform.  Convert this to (n_keyframes, 4, 4).
    keyframe_tfs = keyframe_data.reshape(-1, 4, 4)

    # This directory is stored as camera in object frame, so do the conversion
    # to object in camera frame.
    keyframes_ob_in_cam = np.zeros_like(keyframe_tfs)
    for i, keyframe_tf in enumerate(keyframe_tfs):
        keyframes_ob_in_cam[i] = math_utils.inverse_homogeneous_transformation(
            keyframe_tf)

    return keyframes_ob_in_cam


"""ROS Bag utilities."""
def get_depth_bag_filename(rosbag_number: int) -> str:
    """Get the filename of the ROS bag with the raw depth data."""
    bag_filename = op.join(DATA_GEN_DIR, 'rosbags', f'raw_{rosbag_number}.bag')
    assert op.exists(bag_filename), f'Did not find expected {bag_filename}'
    return bag_filename

def get_odom_bag_filename(rosbag_number: int) -> str:
    """Get the filename of the ROS bag with the TagSLAM pose data."""
    bag_filename = op.join(DATA_GEN_DIR, 'rosbags', f'odom_{rosbag_number}.bag')
    assert op.exists(bag_filename), f'Did not find expected {bag_filename}'
    return bag_filename


"""Filtering/visualization."""
def get_tagslam_offset() -> np.ndarray:
    """Get the TagSLAM offset from the data generation assets directory."""
    filename = op.join(DATA_GEN_DIR, 'assets', 'tagslam_offset.txt')
    return np.loadtxt(filename)

def save_tagslam_offset(offset: np.ndarray) -> None:
    """Save the TagSLAM offset to a file in the data generation assets
    directory."""
    filename = op.join(DATA_GEN_DIR, 'assets', 'tagslam_offset.txt')
    np.savetxt(filename, offset)

def point_cloud_processing_plot_filepath(dataset: str, eps: bool = False
                                         ) -> str:
    """Get the filepath for the point cloud processing plot."""
    filename = f'{dataset}.png' if not eps else f'{dataset}_eps.png'
    return op.join(table_height_calibration_dir(), filename)

def point_cloud_processing_log_filepath(dataset: str) -> str:
    """Get the filepath for the point cloud processing log."""
    return op.join(table_height_calibration_dir(), f'{dataset}.txt')

def table_calibration_yaml_filepath() -> str:
    """Get the filepath for the table calibration yaml file that stores all of
    the table heights for each toss."""
    return op.join(table_height_calibration_dir(), 'table_heights.yaml')

def filter(real_img, sim_img):
    """
    Filter the simulated image from the real depth image.
    """
    masked_img = np.subtract(real_img, sim_img)
    return masked_img


def generate_depth_img_without_robot(depth_dir, mask_dir, filtered_depth_dir):
    """
    Filter out robot from depth image with the generated mask.
    :param str depth_dir: Directory to the original depth image from rosbag.
    :param str mask_dir: Directory to the mask generated by urdf_filter.
    :param str filtered_depth_dir: Directory to save the filtered depth image.
    """
    depth_img = np.loadtxt(depth_dir)
    # depth_img = Image.open(depth_dir)
    mask_img = Image.open(mask_dir)
    depth_array = np.array(depth_img)
    mask_array = np.array(mask_img)
    filtered_depth = deepcopy(depth_array * 1000)
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            if mask_array[i][j] == 255:
                filtered_depth[i][j] = 0
    plt.imshow(filtered_depth)
    plt.colorbar()
    # plt.show()
    imageio.imwrite(filtered_depth_dir, filtered_depth.astype(np.uint16))
    return filtered_depth


def generate_rgb_image_without_robot(rgb_dir, mask_dir, filtered_rgb_dir):
    """
    Filter out robot from rgb image with the generated mask.
    :param str rgb_dir: Directory to the original rgb image from rosbag.
    :param str mask_dir: Directory to the mask generated by urdf_filter.
    :param str filtered_rgb_dir: Directory to save the filtered rgb image.
    """
    rgb_img = Image.open(rgb_dir)
    mask_img = Image.open(mask_dir)
    rgb_array = np.array(rgb_img)
    mask_array = np.array(mask_img)
    filtered_rgb = deepcopy(rgb_array)
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            if mask_array[i][j] == 255:
                filtered_rgb[i][j][0] = 0
                filtered_rgb[i][j][1] = 0
                filtered_rgb[i][j][2] = 0
    # plt.imshow(filtered_rgb)
    # plt.show()
    imageio.imwrite(filtered_rgb_dir, filtered_rgb)
    return filtered_rgb


def import_data(position_file):
    """
    Load data generated from rosbag.
    """
    positions = np.loadtxt(position_file)
    print("joint position imported!")
    return positions


def write_real_depth_as_txt(start_frame, end_frame, img_dir, real_depth_dir):
    """
    Save real depth txt at once since image I/O is very slow.
    """
    loaded_arr = np.loadtxt(img_dir)
    print(loaded_arr.shape)
    load_original_arr = loaded_arr.reshape(
        loaded_arr.shape[0], loaded_arr.shape[1] // 640, 640
    )
    print("Done loading images.")
    for frame_id in tqdm(range(start_frame, end_frame + 1)):
        depth_dir = real_depth_dir % frame_id
        print("frame_id", frame_id)
        real = load_original_arr[frame_id - 1] * 0.001
        np.savetxt(depth_dir, real)


def check_empty_img(masks_dir):
    """
    Since Bundletrack loses tracking if any of the masks are empty, we need to check emptyness for masks.
    """
    empty_list = []
    nums = len([name for name in os.listdir(masks_dir)])
    for frame_id in range(1, nums+1):
        mask_file = (
            os.path.join(masks_dir, "%04i.png"% frame_id)
        )
        image = cv2.imread(mask_file)
        if np.sum(image) == 0:
            empty_list.append(frame_id)
        else:
            continue
    print(f'Empty masks: {empty_list}')
    return empty_list


def denoise(
    frame_id,
    img_dir,
    denoise_mask_dir,
    region=(10, 10),
    show=False,
):
    """
    Filter out small pieces of noise from depth images.
    """
    # img_dir = "./cube_data/depth_image_frame%04i.png" % frame_id
    img_dir = img_dir % frame_id
    # denoise_mask_dir = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets/masks/%04i.png"%frame_id
    denoise_mask_dir = os.path.join(denoise_mask_dir, "%04i.png" % frame_id)
    # Load image, convert to grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(img_dir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Filter using contour area and remove small noise
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Morph close and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, region)
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('close', close)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    im = Image.fromarray(close)
    if im.mode != "L":
        im = im.convert("L")
    im.save(denoise_mask_dir)
    if show:
        plt.imshow(im)
        plt.show()


def render_gif():
    """
    Make pngs into a gif.
    """
    frames = []
    for frame_id in range(1, 4432):
        name = "./cube_data/screen_image_frame00000{}.png".format(frame_id)
        new_frame = Image.open(name)
        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(
        "./png_to_gif.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=300,
        loop=0,
    )


def render_video():
    """
    Make pngs into a video for easy view.
    """
    fileList = []
    for frame_id in tqdm(range(1, 4432)):
        filename = "./denoise_cube_data/frame00000{}.png".format(frame_id)
        fileList.append(filename)

    writer = imageio.get_writer("new_depth.mp4", fps=20)

    for im in fileList:
        writer.append_data(imageio.imread(im))
    writer.close()


"""Data preparation for running BundleTrack on our own RGBD data."""
def copy():
    """
    For copying images from robot_filter folder to BundleTrack folder.
    """
    src_dir = "/home/cnets-vision/mengti_ws/robot_filter/depth_data"
    dst_dir = (
        "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets/depth"
    )
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.png")):
        shutil.copy(jpgfile, dst_dir)


def create_annotated_poses(output_dir, frame_id):
    """Create annotated_poses folder. First txt is the transformation matrix
    from camera to object.  Others are identity matrices solely for evaluation.
    """
    filename = os.path.join(output_dir, "%04i.txt" % frame_id)
    pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.savetxt(filename, pose)


def rename():
    """
    Currently, all data are named as frame00000{frame_id}.png. Need to rename to {frame_id}.png
    to match the required format of BundleTrack.
    Go to the directory and run rename().
    """
    for filename in os.listdir("."):
        print(filename)
        if filename.startswith("frame"):
            os.rename(filename, filename[-8:])


def removed_files(num):
    """Remove the first num frames from data folder."""
    for filename in os.listdir("."):
        name = re.findall(r"\d+", filename)[0]
        new_name = int(name)
        new_name = new_name - num
        new_filename = "%04i.txt" % new_name
        print(filename, new_filename)
        os.rename(filename, new_filename)

def replace_comma_with_space(file_path):
    """Replace every comma with a space in the specified file."""
    
    # Read the content of the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace comma with space
    modified_content = content.replace(',', ' ')
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(modified_content)

    print(f"Replaced commas with spaces in {file_path}")

def process_directory(directory_path):
    """Apply the replace_comma_with_space function to all .txt files in the specified directory."""
    
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                replace_comma_with_space(file_path)
    print('Done!')

def formulate_dataset(dataset):
    source_root = "/home/cnets-vision/mengti_ws/BundleSDF/data/"
    target_folder = f"/home/cnets-vision/mengti_ws/BundleSDF/data/{dataset}_0"
    subfolders = ["rgb", "depth", "masks"]
    total_folders = 10
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for sf in subfolders:
        target_subfolder = os.path.join(target_folder, sf)

        # Create the subfolder in the target directory
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        current_counter = 1  # Reset counter for each subfolder type

        # Loop through each old_toss_X folder for the current subfolder type
        for i in range(1, total_folders + 1):
            source_subfolder = os.path.join(source_root, f"{dataset}_{i}", sf)

            # List all files in the current subfolder
            files = sorted(os.listdir(source_subfolder))

            # Copy each file to the target subfolder and rename it
            for file in files:
                source_file_path = os.path.join(source_subfolder, file)
                target_file_name = f"{current_counter:04}.png"  # Format as 000X.png
                target_file_path = os.path.join(target_subfolder, target_file_name)

                shutil.copy2(source_file_path, target_file_path)
                # print(source_file_path, target_file_path)
                current_counter += 1
    
def main():
    # process_directory('/home/cnets-vision/mengti_ws/BundleSDF/results/ob_in_cam_projected_icp_transformed')
    formulate_dataset()
    # for frame_id in range(1, 3813):
    #     create_annotated_poses(output_dir="/home/cnets-vision/mengti_ws/BundleSDF/data/old_toss_10_tosses/annotated_poses", frame_id=frame_id)

if __name__ == "__main__":
    main()
