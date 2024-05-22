"""Utilities for evaluations."""

import numpy as np
import os
import os.path as op
import pickle
from scipy.spatial import cKDTree
import sys
import torch
from torch import Tensor
from typing import Tuple

import file_utils, math_utils

DATA_GEN_DIR = op.dirname(op.realpath(__file__))
REPO_DIR = op.dirname(DATA_GEN_DIR)
PLL_DIR = op.join(REPO_DIR, 'dair_pll')

if PLL_DIR not in sys.path:
    sys.path.append(PLL_DIR)    # For importing dair_pll.

from dair_pll import inertia as pll_inertia
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem



#================================== METRICS ===================================#
"""ADD and ADD-S (/ADI) metrics:  We found the most information about these at:
https://www.sciencedirect.com/science/article/pii/B978032385787100021X

ADD (average distance of model points) computes the average distance of 1-to-1
mapped object surface points from one pose to another pose.  This is useful/
meaningful for asymmetric objects with distinguishable views.

ADD-S (average closest point distance), sometimes called ADI, computes the
average distance of one pose's surface points to the closest surface point from
another pose.  This handles ambiguous poses, and thus is useful for symmetric
objects where multiple "right" answers may exist.

These both require a mesh, which is likely to be ground truth, but perhaps we
could use the learned mesh instead.
"""
def compute_add_tracking_error(predicted_pose, true_pose, surface_vertices):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    """
    pred_pts = (predicted_pose@math_utils.to_homogeneous(surface_vertices).T).T[:,:3]
    gt_pts = (true_pose@math_utils.to_homogeneous(surface_vertices).T).T[:,:3]
    e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    return e

def compute_adds_tracking_error(predicted_pose, true_pose, surface_vertices):
    """
    @predicted_pose: 4x4 mat
    @true_pose: 4x4 mat
    @surface_vertices: (N,3)
    """
    pred_pts = (predicted_pose@math_utils.to_homogeneous(surface_vertices).T).T[:,:3]
    gt_pts = (true_pose@math_utils.to_homogeneous(surface_vertices).T).T[:,:3]
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e


#============================= RESULTS GATHERING ==============================#
def get_pll_config_stats_checkpoint(
        pll_run_results_dir: str) -> Tuple[dict, dict, dict]:
    """Get PLL run configuration, statistics, and checkpoint objects.  Returns
    None for any that don't exist."""
    config, stats, checkpoint = None, None, None

    config_file = op.join(pll_run_results_dir, 'config.pkl')
    if op.exists(config_file):
        with open(config_file, 'rb') as file:
            config = pickle.load(file)

    stats_file = op.join(pll_run_results_dir, 'statistics.pkl')
    if op.exists(stats_file):
        with open(stats_file, 'rb') as file:
            stats = pickle.load(file)

    checkpoint_file = op.join(pll_run_results_dir, 'checkpoint.pt')
    if op.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)

    return config, stats, checkpoint

def convert_inertia_theta_to_pi_cm(theta: Tensor) -> Tensor:
    """Convert inertia theta to pi-cm units."""
    return pll_inertia.InertialParameterConverter.theta_to_pi_cm(theta)

def get_mass_from_urdf(urdf_path: str) -> float:
    """Get mass from URDF file."""
    with open(urdf_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if 'mass' in line:
            mass = float(line.split('value="')[1].split('"')[0])
            return mass
    raise RuntimeError(f'Could not find mass in URDF file: {urdf_path}')

def overwrite_mesh_name_in_urdf(urdf_path: str) -> None:
    """Overwrite the mesh obj filename in the URDF path to match the expected
    BundleSDF-derived filename."""
    with open(urdf_path+'.tmp', 'w') as write_file:
        with open(urdf_path, 'r') as read_file:
            line = read_file.read().replace('"test.obj"', '"bsdf_mesh.obj"')
            write_file.write(line)

    os.system(f'mv {urdf_path}.tmp {urdf_path}')

def create_multibody_learnable_system(
        urdf_path: str) -> MultibodyLearnableSystem:
    """Create a MultibodyLearnableSystem from a URDF path.  Note that this
    hardcodes the frame rate to be 30.  The loss weights are all given as zeros,
    but this should be inconsequential since the system will not be trained."""
    output_urdf_dir = file_utils.assure_created(
        op.join(op.dirname(urdf_path), 'pll_urdf'))

    return MultibodyLearnableSystem(
        init_urdfs = {'learned_object': urdf_path},
        dt = 1.0/30,
        loss_weights_dict = {
            'w_pred': 0, 'w_comp': 0, 'w_pen': 0, 'w_diss': 0, 'w_bsdf': 0},
        force_mesh_to_be_polygon = True,
        output_urdfs_dir = output_urdf_dir
    ).eval()


#============================ DYNAMICS PREDICTIONS ============================#
def get_pll_tagslam_trajectories_pll_format(object: str) -> dict:
    """Load TagSLAM trajectories from the PLL assets directory so they are
    already in PLL format.  Returns a dictionary with toss numbers as keys and
    torch tensors (N, 13) as values.  Returns None if the object is tagless and
    thus there are no TagSLAM trajectories."""
    if object in file_utils.TAGLESS_OBJECTS:
        print(f'Object {object} is tagless; no TagSLAM trajectories.')
        return None
    
    # Get all of the trajectories, tosses 1 through 10 (or up until created).
    trajectories = {}

    for toss_i in range(1, 11):
        vision_asset = f'{object}_{toss_i}'
        tagslam_dir = file_utils.contactnets_input_dir_tagslam(
            vision_asset, full=False, create=False)
        
        if op.exists(tagslam_dir):
            trajectory = torch.load(op.join(tagslam_dir, f'{toss_i}.pt'))
            trajectories[toss_i] = trajectory
        else:
            print(f'No trajectory found for {vision_asset}; skipping.')

    return trajectories
    
def get_synced_bsdf_tagslam_toss_poses(
        vision_asset: str, bundlesdf_id: str, cycle_iteration: int,
        desired_toss_num: int) -> Tensor:
    """Get a time-synchronized pair of TagSLAM and BundleSDF poses for a given
    toss number.  If the desired toss number is within the range of the vision
    asset, then the first frame of that toss is used.  Otherwise, the first
    frame of the full BundleSDF trajectory is used.  Returns both poses as 4x4
    transformation matrices."""
    start_toss = int(vision_asset.split('_')[1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    
    # First try to use the first frame of the exact toss.
    if desired_toss_num in range(start_toss, end_toss+1):
        bsdf_dir = file_utils.contactnets_input_dir_bundlesdf(
            vision_asset, iteration=cycle_iteration, bundlesdf_id=bundlesdf_id,
            full=False, create=False)
        tagslam_dir = file_utils.contactnets_input_dir_tagslam(
            vision_asset, full=False, create=False)

        assert op.exists(bsdf_dir), f'Missing {bsdf_dir=}'
        assert op.exists(tagslam_dir), f'Missing {tagslam_dir=}'

        b_traj = torch.load(op.join(bsdf_dir, f'{desired_toss_num}.pt'))
        t_traj = torch.load(op.join(tagslam_dir, f'{desired_toss_num}.pt'))

    # Otherwise, use the beginning of the full BundleSDF tracking trajectory.
    else:
        bsdf_dir = file_utils.contactnets_input_dir_bundlesdf(
            vision_asset, iteration=cycle_iteration, bundlesdf_id=bundlesdf_id,
            full=True, create=False)
        tagslam_dir = file_utils.contactnets_input_dir_tagslam(
            vision_asset, full=True, create=False)

        assert op.exists(tagslam_dir), f'Missing {bsdf_dir=}'
        assert op.exists(tagslam_dir), f'Missing {tagslam_dir=}'

        b_traj = torch.load(op.join(bsdf_dir, f'{bundlesdf_id}.pt'))
        t_traj = torch.load(op.join(tagslam_dir, 'tagslam.pt'))
        
    # Get the first poses.
    b_pll_state = b_traj[0]
    t_pll_state = t_traj[0]

    # Convert to transformation matrices.
    b_trans_mat = math_utils.pll_format_to_trans_mat(b_pll_state)
    t_trans_mat = math_utils.pll_format_to_trans_mat(t_pll_state)

    return b_trans_mat, t_trans_mat

def get_pll_rollout_trajectory(
        system: MultibodyLearnableSystem, target_traj: Tensor) -> Tensor:
    """Get a rollout trajectory by simulating a PLL system from the first state
    of a provided target trajectory."""
    # Use the input argument structure of system.simulate().
    x_0 = target_traj[..., :1, :]
    carry_0 = system.carry_callback()
    steps = target_traj.shape[0] - 1

    prediction, carry = system.simulate(x_0, carry_0, steps)
    del carry

    return prediction.detach().clone()

