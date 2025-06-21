"""Get GT aligned geometry from experiments starting with toss 1 for experiments
that start with other toss numbers.  This allows us to avoid needing to re-run
ICP by reusing the previously aligned geometry, even if the initial toss
numbers are not the same."""

import click
import numpy as np
import open3d as o3d
import os
import os.path as op
import pdb

import file_utils, icp


def get_keyframe_poses_indices_times(
        vision_asset: str, cycle_iteration: int, bundlesdf_id: str):
    """Get the poses (as 4x4 ob in cam matrices), indices, and timestamps
    associated with a BundleSDF NeRF run's results."""
    # Get the keyframe poses from the appropriate experiment.
    keyframe_idx = file_utils.load_keyframe_indices_from_nerf_results_yml(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id)
    keyframe_tfs = file_utils.load_optimized_keyframe_poses_from_nerf_results(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        tracking_bundlesdf_id=bundlesdf_id,
        nerf_bundlesdf_id=bundlesdf_id)
    
    # Get the corresponding timestamps.
    dataset_dir = file_utils.cnets_data_gen_dataset_dir(
        dataset=vision_asset, check_exists=True)
    dataset_times_filepath = op.join(
        dataset_dir, 'bundlesdf_timestamps.txt')
    dataset_timestamps = np.loadtxt(dataset_times_filepath)
    keyframe_ts = dataset_timestamps[keyframe_idx]
    
    return keyframe_tfs, keyframe_idx, keyframe_ts

def get_tracking_pose_at_timestamp_or_idx(
        vision_asset: str, cycle_iteration: int, bundlesdf_id: str,
        timestamp: float = None, timestamp_idx_1: int = None):
    """Returns the (4,4) transformation matrix ob_in_cam."""
    if timestamp_idx_1 is None:
        # Figure out the index of the timestamp in the dataset.
        dataset_dir = file_utils.cnets_data_gen_dataset_dir(
            dataset=vision_asset, check_exists=True)
        dataset_times_filepath = op.join(
            dataset_dir, 'bundlesdf_timestamps.txt')
        dataset_timestamps = np.loadtxt(dataset_times_filepath)

        if timestamp not in dataset_timestamps:
            pdb.set_trace()
            raise ValueError(f'Timestamp {timestamp} not found in dataset.')
        
        timestamp_idx_0 = np.where(dataset_timestamps == timestamp)[0][0]
        timestamp_idx_1 = timestamp_idx_0 + 1

    # Get the poses.
    bundlesdf_pose_dir = file_utils.bundlesdf_pose_dir(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id)
    pose_at_t = np.loadtxt(
        op.join(bundlesdf_pose_dir, f'{timestamp_idx_1:04d}.txt'))

    return pose_at_t

def get_A_geom_to_A_track(
        vision_asset: str, cycle_iteration: int, bundlesdf_id: str):
    # Use a keyframe to get the transform from the geometry origin to the
    # tracking origin.
    keyframe_tfs, keyframe_idx, keyframe_ts = get_keyframe_poses_indices_times(
        vision_asset=vision_asset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id)
    tf_cam_to_geom = keyframe_tfs[-1]

    # Get the corresponding tracking pose.
    tf_cam_to_tracking = get_tracking_pose_at_timestamp_or_idx(
        vision_asset=vision_asset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id, timestamp_idx_1=keyframe_idx[-1])
   
    return np.linalg.inv(tf_cam_to_geom) @ tf_cam_to_tracking

def get_A_track_to_B_track(
        A_vision_asset: str, A_cycle_iteration: int, A_bundlesdf_id: str,
        B_vision_asset: str, B_cycle_iteration: int, B_bundlesdf_id: str):
    # Get a single timestamp that works for both of the datasets.
    assert '1-' in A_vision_asset and '-' not in B_vision_asset, \
        f'Expecting for this to work from 1-X to single tosses.'
    assert '1' not in B_vision_asset, 'Already have GT aligned geometry for ' +\
        f'toss 1.'
    
    # Use the first timestamp of single toss as the common timestamp.
    B_dataset_dir = file_utils.cnets_data_gen_dataset_dir(
        dataset=B_vision_asset, check_exists=True)
    B_dataset_times_filepath = op.join(
        B_dataset_dir, 'bundlesdf_timestamps.txt')
    B_dataset_timestamps = np.loadtxt(B_dataset_times_filepath)

    common_timestamp = B_dataset_timestamps[0]

    # Get the tracking poses of each at the timestamp.
    tf_cam_to_A_tracking = get_tracking_pose_at_timestamp_or_idx(
        vision_asset=A_vision_asset, cycle_iteration=A_cycle_iteration,
        bundlesdf_id=A_bundlesdf_id, timestamp=common_timestamp)
    tf_cam_to_B_tracking = get_tracking_pose_at_timestamp_or_idx(
        vision_asset=B_vision_asset, cycle_iteration=B_cycle_iteration,
        bundlesdf_id=B_bundlesdf_id, timestamp=common_timestamp)
    
    return np.linalg.inv(tf_cam_to_A_tracking) @ tf_cam_to_B_tracking

def get_B_track_to_B_geom(
        vision_asset: str, cycle_iteration: int, bundlesdf_id: str):
    # Can reuse get_A_geom_to_A_track.
    geom_to_tracking = get_A_geom_to_A_track(
        vision_asset=vision_asset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id)
    return np.linalg.inv(geom_to_tracking)

def get_A_geom_to_B_geom(
        A_vision_asset: str, A_cycle_iteration: int, A_bundlesdf_id: str,
        B_vision_asset: str, B_cycle_iteration: int, B_bundlesdf_id: str):
    A_geom_to_A_track = get_A_geom_to_A_track(
        vision_asset=A_vision_asset, cycle_iteration=A_cycle_iteration,
        bundlesdf_id=A_bundlesdf_id)
    A_track_to_B_track = get_A_track_to_B_track(
        A_vision_asset=A_vision_asset, A_cycle_iteration=A_cycle_iteration,
        A_bundlesdf_id=A_bundlesdf_id, B_vision_asset=B_vision_asset,
        B_cycle_iteration=B_cycle_iteration, B_bundlesdf_id=B_bundlesdf_id)
    B_track_to_B_geom = get_B_track_to_B_geom(
        vision_asset=B_vision_asset, cycle_iteration=B_cycle_iteration,
        bundlesdf_id=B_bundlesdf_id)
    
    return A_geom_to_A_track @ A_track_to_B_track @ B_track_to_B_geom

def get_gt_aligned_geometry(
        vision_asset: str, cycle_iteration: int, bundlesdf_id: str):
    assert '-' not in vision_asset and '1' not in vision_asset, \
        f'Expecting for this to work for single toss, and not toss 1.'
    
    new_eval_dir = file_utils.evaluation_subdir(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        tracking_bundlesdf_id=bundlesdf_id, nerf_bundlesdf_id=bundlesdf_id
    )

    # Get another experiment that already has aligned geometry.
    object = vision_asset.split('_')[:-1]
    object = '_'.join(object)
    other_vision_asset = object + '_1-5'
    other_cycle_iteration = 1
    other_bundlesdf_id = 'bundlesdf_id_00'

    # Get the transform from the geometry of the other experiment to the
    # geometry of this experiment.
    A_geom_to_B_geom = get_A_geom_to_B_geom(
        A_vision_asset=other_vision_asset, A_cycle_iteration=other_cycle_iteration,
        A_bundlesdf_id=other_bundlesdf_id,
        B_vision_asset=vision_asset, B_cycle_iteration=cycle_iteration,
        B_bundlesdf_id=bundlesdf_id
    )

    # Load the other experiment's aligned geometry.
    other_eval_dir = file_utils.evaluation_subdir(
        dataset=other_vision_asset, cycle_iteration=other_cycle_iteration,
        tracking_bundlesdf_id=other_bundlesdf_id,
        nerf_bundlesdf_id=other_bundlesdf_id
    )
    other_bsdf_true_mesh_path = op.join(other_eval_dir, 'true_geom_aligned.obj')
    mesh_A = icp.load_mesh_from_obj(other_bsdf_true_mesh_path)

    # Apply the transform.  After testing, needs to be inverse.
    mesh_B = mesh_A.transform(np.linalg.inv(A_geom_to_B_geom))

    # Write the mesh file.
    o3d.io.write_triangle_mesh(
        op.join(new_eval_dir, 'true_geom_aligned.obj'), mesh_B,
        write_triangle_uvs=False, write_vertex_colors=False
    )
    print(f'Wrote newly aligned gometry for {vision_asset}.')


#######################################################################
#@click.command()
#@click.option('--vision-asset',
#              type=str,
#              default=None,
#              help="directory of the asset folder e.g. cube_2; encodes " + \
#                "system and tosses.")
#@click.option('--bundlesdf-id',
#              type=str,
#              default=None,
#              help="what BundleSDF run ID associated with pose outputs to use.")
#@click.option('--nerf-bundlesdf-id',
#              type=str,
#              default=None,
#              help="what BundleSDF run ID associated with NeRF outputs to use.")
#@click.option('--cycle-iteration',
#              type=int,
#              default=1,
#              help="BundleSDF iteration number (can't choose 0 since that " + \
#                "means use TagSLAM poses).")

def main_command(vision_asset: str, bundlesdf_id: str, nerf_bundlesdf_id: str,
                 cycle_iteration: int):
    assert '-' not in vision_asset, f'This script was only written for ' + \
        f'single-toss experiments in mind.'
    toss_num = int(vision_asset.split('_')[-1])
    assert toss_num != 1, f'Already have GT aligned geometry for toss 1.'
    obj = vision_asset.split('_')[:-1]
    obj = '_'.join(obj)

    # Decode the BundleSDF run ID.
    if bundlesdf_id[:13] != 'bundlesdf_id_':
        bundlesdf_id = f'bundlesdf_id_{bundlesdf_id}'
    if nerf_bundlesdf_id is None:
        nerf_bundlesdf_id = bundlesdf_id
    elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
        nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'

    # Check if results already exist.
    eval_dir = file_utils.evaluation_subdir(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        tracking_bundlesdf_id=bundlesdf_id,
        nerf_bundlesdf_id=nerf_bundlesdf_id
    )
    if op.exists(op.join(eval_dir, 'true_geom_aligned.obj')):
        #print(f'Overwriting GT aligned geometry for {vision_asset}.')
        #os.system(f'rm -rf {eval_dir}/*')
        print(f'Already found GT aligned geometry for {vision_asset}.')
        return
    
    # Otherwise, get the GT aligned geometry.
    get_gt_aligned_geometry(
        vision_asset=vision_asset, cycle_iteration=cycle_iteration,
        bundlesdf_id=bundlesdf_id
    )



if __name__ == "__main__":
    NUMBERS = ['2', '3', '4', '5']
    OBJECTS = ['bakingbox', 'cardboard', 'crushedcan', 'gallon', 'greencan',
            'oatly', 'pinkcan', 'stapler', 'styrofoam', 'cube', 'bottle', 'half',
            'milk', 'egg', 'napkin']
    for obj in OBJECTS:
        for num in NUMBERS:
            vision_asset = f'{obj}_{num}'
            try:
                main_command(
                    vision_asset=vision_asset, bundlesdf_id='00',
                    cycle_iteration=1, nerf_bundlesdf_id='00')
            except Exception as e:
                print(f'Issue with {vision_asset}_00_1: {e}')
            try:
                main_command(
                    vision_asset=vision_asset, bundlesdf_id='02',
                    cycle_iteration=2, nerf_bundlesdf_id='02')
            except Exception as e:
                print(f'Issue with {vision_asset}_02_2: {e}')


    # main_command()  # pylint: disable=no-value-for-parameter
