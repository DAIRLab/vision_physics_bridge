"""Utilities for evaluations."""

from PIL import Image
import numpy as np
import os
import os.path as op
import pdb
import pickle
from scipy.spatial import cKDTree
import sys
import torch
from torch import Tensor
from typing import Tuple

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import file_utils, math_utils, overlay_videos

from overlay_videos import OverlayVideoGenerator

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
    """Average Distance of Model Points for objects with no indistinguishable
    views by Hinterstoisser et al. (ACCV 2012).  Takes a single pose for each
    the predicted and true trajectories, each as a 4x4 homogeneous transform.
    """
    assert predicted_pose.shape == true_pose.shape == (4, 4), \
        f'Invalid {predicted_pose.shape=} or {true_pose.shape=}.'
    homogeneous_vertices = math_utils.to_homogeneous(surface_vertices)

    pred_pts = (predicted_pose @ homogeneous_vertices.T).T[:, :3]
    gt_pts = (true_pose @ homogeneous_vertices.T).T[:, :3]

    e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    return e

def compute_adds_tracking_error(predicted_pose, true_pose, surface_vertices):
    """
    @predicted_pose: 4x4 mat
    @true_pose: 4x4 mat
    @surface_vertices: (N,3)
    """
    assert predicted_pose.shape == true_pose.shape == (4, 4), \
        f'Invalid {predicted_pose.shape=} or {true_pose.shape=}.'
    homogeneous_vertices = math_utils.to_homogeneous(surface_vertices)

    pred_pts = (predicted_pose @ homogeneous_vertices.T).T[:, :3]
    gt_pts = (true_pose @ homogeneous_vertices.T).T[:, :3]

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

def overwrite_mesh_name_in_urdf(urdf_path: str, new_obj_name: str) -> None:
    """Overwrite the mesh obj filename in the URDF path to match the expected
    BundleSDF-derived filename."""
    with open(urdf_path+'.tmp', 'w') as write_file:
        with open(urdf_path, 'r') as read_file:
            line = read_file.read().replace('"test.obj"', f'"{new_obj_name}"')
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

def get_bundlesdf_trajectories_pll_format(
        vision_asset: str, cycle_iteration: int, bundlesdf_id: str) -> dict:
    """Load BundleSDF trajectories from the PLL assets directory so they are
    already in PLL format.  Returns a dictionary with toss numbers as keys and
    torch tensors (N, 13) as values.  Gets all toss numbers included in the
    vision asset."""
    trajectories = {}

    bsdf_pose_dir = file_utils.contactnets_input_dir_bundlesdf(
        vision_asset, cycle_iteration, bundlesdf_id, full=False, create=False)

    for file in os.listdir(bsdf_pose_dir):
        if file.endswith('.pt'):
            toss_i = int(file.split('.')[0])
            trajectory = torch.load(op.join(bsdf_pose_dir, file))
            trajectories[toss_i] = trajectory

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


class PredictionOverlayGenerator(OverlayVideoGenerator):
    """Make an overlay video showing the tracked BundleSDF poses and the
    dynamics predictions during the tosses."""
    def __init__(self, vision_asset: str, tracking_bundlesdf_id: str,
                 nerf_bundlesdf_id: str, cycle_iteration: int,
                 prediction_tosses: list, bsdf_only: bool = False,
                 remote: bool = False):
        super().__init__(
            vision_asset=vision_asset,
            tracking_bundlesdf_id=tracking_bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id,
            cycle_iteration=cycle_iteration,
            bsdf_only=bsdf_only, remote=remote
        )

        # Overwrite the output file so it gets written to evaluation directory.
        self.output_file = file_utils.evaluation_toss_prediction_video_filepath(
            dataset=vision_asset, tracking_bundlesdf_id=tracking_bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id, cycle_iteration=cycle_iteration
        )

        # Temporary restriction that we can only support rendering toss
        # predictions that are within the vision asset's range.
        prediction_tosses_to_render = []
        for toss in prediction_tosses:
            if toss < self.start_toss or toss > self.end_toss:
                print(f'Cannot render toss {toss} for {vision_asset}.')
            else:
                prediction_tosses_to_render.append(toss)
        self.prediction_tosses = prediction_tosses_to_render

        # Get the path to the evaluation directory.
        self.evaluation_dir = file_utils.evaluation_subdir(
            dataset=self.vision_asset,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id,
            cycle_iteration=self.cycle_iteration
        )

        # From the evaluation directory, get the predicted tosses.
        self._get_predicted_poses_in_world()

    def _get_predicted_poses_in_world(self):
        """Get the target and predicted trajectories for every toss in
        self.prediction_tosses.  These are in 4x4 transformation matrix form and
        stored as a dictionary with keys as the prediction toss number and the
        entries an (Ni, 4, 4) numpy array."""
        # The stored trajectories used PLL's space, which has the table height
        # at z=0.  Need to add the table height back in to make consistent with
        # world coordinates.
        table_heights = np.array([
            file_utils.load_table_z_height(self.object, toss) for toss in
            range(self.start_toss, self.end_toss+1)
        ])
        z_table = np.mean(table_heights)

        target_trajs = {}
        predicted_trajs = {}
        for toss in self.prediction_tosses:
            target_name = op.join(self.evaluation_dir, f'target_toss_{toss}.pt')
            pred_name = op.join(
                self.evaluation_dir, f'predicted_toss_{toss}.pt')

            assert op.exists(target_name), f'Cannot find {target_name=}.'
            assert op.exists(pred_name), f'Cannot find {pred_name=}.'

            target_traj = np.array(torch.load(target_name))
            predicted_traj = np.array(torch.load(pred_name))
            assert target_traj.shape == predicted_traj.shape, f'Cannot ' + \
                f'handle different shapes {target_traj.shape=}, ' + \
                f'{predicted_traj.shape=}.'

            # Convert to 4x4 transformation matrices.
            target_traj_trans = np.zeros((target_traj.shape[0], 4, 4))
            predicted_traj_trans = np.zeros((predicted_traj.shape[0], 4, 4))
            for i in range(target_traj.shape[0]):
                target_pose = target_traj[i]
                predicted_pose = predicted_traj[i]
                target_trans_mat = math_utils.pll_format_to_trans_mat(
                    target_pose)
                predicted_trans_mat = math_utils.pll_format_to_trans_mat(
                    predicted_pose)

                # Adjust for the table height.
                target_trans_mat[2, 3] += z_table
                predicted_trans_mat[2, 3] += z_table

                # Store to tensor.
                target_traj_trans[i] = target_trans_mat
                predicted_traj_trans[i] = predicted_trans_mat

            # Store the whole (N, 4, 4) trajectory to dictionary.
            target_trajs[toss] = target_traj_trans
            predicted_trajs[toss] = predicted_traj_trans

        self.target_trajs = target_trajs
        self.predicted_trajs = predicted_trajs

    def _add_meshcat_objects(self, vis: meshcat.Visualizer) -> None:
        super()._add_meshcat_objects(vis)
        vis["dynamics_triad"].set_object(g.triad(scale=0.1))
        vis["dynamics_mesh"].set_object(
            g.ObjMeshGeometry.from_file(self.mesh_file),
            g.MeshLambertMaterial(
                color=overlay_videos.PREDICTION_COLOR,
                reflectivity=0.0, transparent=0, opacity=.4)
        )

    def _set_meshcat_object_poses(self, frame_i: int, T_WA: np.ndarray,
                                  T_CB: np.ndarray) -> None:
        super()._set_meshcat_object_poses(frame_i, T_WA, T_CB)

        # First determine if the frame is within a toss.
        toss_i = self._within_which_toss(image_frame_i=frame_i+1)

        if (toss_i is not None) and (toss_i in self.prediction_tosses):
            # Get the predicted pose.
            toss_frame = frame_i+1 - self.start_frames[toss_i - self.start_toss]
            T_WP = self.predicted_trajs[toss_i][toss_frame]
            # TODO can debug the TagSLAM to BundleSDF transformation by looking
            # at the target trajectories here^

            self.vis["dynamics_triad"].set_transform(self.T_MW @ T_WP)
            self.vis["dynamics_mesh"].set_transform(self.T_MW @ T_WP)

        # If not showing a prediction, move the predicted geometry out of view.
        else:
            out_of_view_tf = tf.translation_matrix([0, 0, -1])

            self.vis["dynamics_triad"].set_transform(self.T_MC @ out_of_view_tf)
            self.vis["dynamics_mesh"].set_transform(self.T_MC @ out_of_view_tf)

# if __name__ == '__main__':
#     pdb.set_trace()
#     pgen = PredictionOverlayGenerator('bakingbox_1-2', '00', '00', 2, [1, 2])
#     # pgen = PredictionOverlayGenerator('cube_1', '00', '00', 2, [1])
#     pgen.make_overlay_video()
#     pdb.set_trace()
