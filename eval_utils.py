"""Utilities for evaluations."""

from PIL import Image
import numpy as np
import os
import os.path as op
import pdb
import pickle
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
import sys
import torch
from torch import Tensor
from typing import Tuple

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import file_utils, math_utils, overlay_videos

from conversion_bsdf_to_pll import TrajectoryConverterBundleSDFToPLL
from overlay_videos import OverlayVideoGenerator

DATA_GEN_DIR = op.dirname(op.realpath(__file__))
REPO_DIR = op.dirname(DATA_GEN_DIR)
PLL_DIR = op.join(REPO_DIR, 'dair_pll')

if PLL_DIR not in sys.path:
    sys.path.append(PLL_DIR)    # For importing dair_pll.

from dair_pll import inertia as pll_inertia
from dair_pll import deep_support_function as pll_dsf
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.system import MeshSummary


METRICS_BY_TOSS = ['dynamics_rollout_metrics', 'dynamics_single_step_metrics',
                   'tracking_metrics']

POSITION_AUC_THRESHOLD = 0.1
ORIENTATION_AUC_THRESHOLD = np.pi / 2
PENETRATION_AUC_THRESHOLD = 0.02    # TODO BIBIT the results seem really low



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

def chamfer_distance(point_cloud_1, point_cloud_2):
    """Chamfer distance between two point clouds, computed identically to how
    BundleSDF computed it.

    NOTE should not be mean of all, see:
    https://pdal.io/en/stable/apps/chamfer.html
    """
    kdtree1 = cKDTree(point_cloud_1)
    dists1, _indices1 = kdtree1.query(point_cloud_2)
    kdtree2 = cKDTree(point_cloud_2)
    dists2, _indices2 = kdtree2.query(point_cloud_1)
    return 0.5*(dists1.mean()+dists2.mean())

def extract_mesh_from_support_points(support_points: Tensor):
    """Given a set of convex polytope vertices, extracts a vertex/face mesh.

    Args:
        support_points: ``(*, 3)`` polytope vertices.

    Returns:
        Object vertices and face indices.
    """
    support_point_hashes = set()
    unique_support_points = []

    # remove duplicate vertices
    for vertex in support_points:
        vertex_hash = hash(vertex.numpy().tobytes())
        if vertex_hash in support_point_hashes:
            continue
        support_point_hashes.add(vertex_hash)
        unique_support_points.append(vertex)

    vertices = torch.stack(unique_support_points)
    hull = ConvexHull(vertices.numpy())
    faces = Tensor(hull.simplices).to(torch.long)  # type: ignore

    _, backwards, _ = pll_dsf.extract_outward_normal_hyperplanes(
        vertices.unsqueeze(0), faces.unsqueeze(0))
    backwards = backwards.squeeze(0)
    faces[backwards] = faces[backwards].flip(-1)

    return MeshSummary(vertices=support_points, faces=faces)

def _get_mesh_interior_point(halfspaces: np.ndarray) -> Tuple[np.ndarray,float]:
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                             (halfspaces.shape[0], 1))
    objective_coefficients = np.zeros((halfspaces.shape[1],))
    objective_coefficients[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = linprog(objective_coefficients, A_ub=A, b_ub=b, bounds=(None, None))
    interior_point = res.x[:-1]
    interior_point_gap = res.x[-1]
    return interior_point, interior_point_gap

def convex_volume_error(vertices_learned: Tensor,
                        vertices_true: Tensor) -> Tensor:
    """Relative error between two convex hulls of provided vertices.  This
    definition comes from:
    https://github.com/ebianchi/dair_pll/blob/main/helpers/corl_plot.py#L428

    Use the identity that the area of the non-overlapping region is the sum of
    the areas of the two polygons minus twice the area of their intersection.

    Args:
        vertices_learned: (N, 3) tensor of vertices of the learned geometry.
        vertices_true: (N, 3) tensor of vertices of the true geometry.
    """
    # pylint: disable=too-many-locals
    true_volume = ConvexHull(vertices_true.numpy()).volume
    sum_volume = ConvexHull(vertices_learned.numpy()).volume + true_volume

    mesh_learned = extract_mesh_from_support_points(vertices_learned)
    mesh_true = extract_mesh_from_support_points(vertices_true)

    normal_learned, _, extent_learned = \
        pll_dsf.extract_outward_normal_hyperplanes(
            mesh_learned.vertices.unsqueeze(0), mesh_learned.faces.unsqueeze(0))
    normal_true, _, extent_true = pll_dsf.extract_outward_normal_hyperplanes(
        mesh_true.vertices.unsqueeze(0), mesh_true.faces.unsqueeze(0))

    halfspaces_true = torch.cat(
        [normal_true.squeeze(), -extent_true.squeeze().unsqueeze(-1)],
        dim=1)

    halfspaces_learned = torch.cat(
        [normal_learned.squeeze(), -extent_learned.squeeze().unsqueeze(-1)],
        dim=1)

    intersection_halfspaces = torch.cat(
        [halfspaces_true, halfspaces_learned], dim=0).numpy()

    # find interior point of intersection
    interior_point, interior_point_gap = _get_mesh_interior_point(
        intersection_halfspaces)

    intersection_volume = 0.

    if interior_point_gap > 0.:
        # intersection is non-empty
        intersection_halfspace_convex = HalfspaceIntersection(
            intersection_halfspaces, interior_point)

        intersection_volume = ConvexHull(
            intersection_halfspace_convex.intersections).volume

    return Tensor([sum_volume - 2 * intersection_volume]).abs() / true_volume

def compute_auc(array_of_errors, max_val):
    """Compute the Area Under the Curve (AUC) error for a given trajectory with
    respect to a provided maximum value of tolerated error."""
    if len(array_of_errors) == 0:
        return 0

    array_of_errors = np.sort(np.array(array_of_errors))
    n = len(array_of_errors)

    prec = np.arange(1,n+1) / float(n)
    array_of_errors = array_of_errors.reshape(-1)
    prec = prec.reshape(-1)

    index = np.where(array_of_errors<max_val)[0]
    array_of_errors = array_of_errors[index]
    prec = prec[index]

    mrec=[0, *list(array_of_errors), max_val]
    mpre=[0, *list(prec), prec[-1]]

    for i in range(1,len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    mpre = np.array(mpre)
    mrec = np.array(mrec)
    i = np.where(mrec[1:]!=mrec[0:len(mrec)-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) / max_val

    return ap.item()

def compute_position_auc_error(array_of_errors):
    """Compute the Area Under the Curve (AUC) error for a given trajectory.
    Computed identically to BundleSDF's implementation."""
    return compute_auc(array_of_errors, max_val=POSITION_AUC_THRESHOLD)

def compute_orientation_auc_error(array_of_errors):
    """This is the same as position AUC error, but with a different max_val
    corresponding to 90 degrees of orientation error."""
    return compute_auc(array_of_errors, max_val=ORIENTATION_AUC_THRESHOLD)

def compute_penetration_auc_error(array_of_errors):
    """Compute the Area Under the Curve (AUC) error for a given trajectory."""
    return compute_auc(array_of_errors, max_val=PENETRATION_AUC_THRESHOLD)


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
            line = read_file.read(
                ).replace('"test.obj"', f'"{new_obj_name}"'
                ).replace('"bundlesdf_mesh.obj"', f'"{new_obj_name}"'
                ).replace('"test_best.obj"', f'"{new_obj_name}"')
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

def create_empty_results_dict(
        vision_asset: str, cycle_iteration: int, last_run_was_bsdf: bool = True
) -> dict:
    """Create an empty results dictionary for a given vision asset.  It contains
    the structure of the results yaml file, modified to only include metric keys
    for metrics relevant to the vision asset -- i.e. TagSLAM-related metrics
    only if the object is tagged, and toss numbers only for those included in
    the vision asset.  All entries are set to None."""
    # First load the empty results yaml file.
    empty_results = file_utils.load_empty_results_yaml()

    # First modification:  Add the appropriate toss numbers.
    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])

    for category_key in METRICS_BY_TOSS:
        sub_results = empty_results[category_key]['against_bundlesdf']
        for metric_key in sub_results.keys():
            sub_results[metric_key] = {
                f'toss_{i}': {'traj': None, 'mean': None} for i in range(
                    start_toss, end_toss+1)}
    sub_results = empty_results['tracking_metrics']['against_tagslam']
    for metric_key in sub_results.keys():
        sub_results[metric_key] = {
            f'toss_{i}': {'traj': None, 'mean': None} for i in range(
                start_toss, end_toss+1)}
        sub_results[metric_key]['full'] = {'traj': None, 'mean': None}
    sub_results = empty_results['tracking_metrics']['against_bundlesdf']
    for metric_key in sub_results.keys():
        sub_results[metric_key]['full'] = {'traj': None, 'mean': None}

    # Second modification:  Get rid of any against_tagslam entries if the object
    # is tagless.
    object = '_'.join(vision_asset.split('_')[:-1])
    if object in file_utils.TAGLESS_OBJECTS:
        for category_key in METRICS_BY_TOSS:
            del empty_results[category_key]['against_tagslam']

    # Third modification:  Get rid of any dynamics predictions if PLL has never
    # been run.
    if cycle_iteration <= 1 and last_run_was_bsdf:
        del empty_results['dynamics_rollout_metrics']
        del empty_results['dynamics_single_step_metrics']

    # Fourth modification:  Get rid of any tracking metrics if the last run was
    # PLL instead of BundleSDF.
    if not last_run_was_bsdf:
        del empty_results['tracking_metrics']

    return empty_results


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

    print(f'First checking for TagSLAM trajectories for {object}_1-10.')
    vision_asset = f'{object}_1-10'
    tagslam_dir = file_utils.contactnets_input_dir_tagslam(
        vision_asset, full=False, create=False)
    if op.exists(tagslam_dir):
        for toss_i in range(1, 11):
            trajectory = torch.load(op.join(tagslam_dir, f'{toss_i}.pt'))
            trajectories[toss_i] = trajectory
        return trajectories
    
    print(f'{object}_1-10 not found; trying individual tosses instead.')
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
        system: MultibodyLearnableSystem, target_traj: Tensor,
        start_adjust: int = 0) -> Tensor:
    """Get a rollout trajectory by simulating a PLL system from the first state
    of a provided target trajectory.  A start adjust can be given to start the
    rollout from a different beginning.  A start adjust of i means preserve the
    first i steps of the trajectory, then simulate starting from the ith index.
    """
    assert target_traj.ndim == 2, f'Invalid {target_traj.shape=}.'
    assert target_traj.shape[1] == 13, f'Invalid {target_traj.shape=}.'

    # Use the input argument structure of system.simulate().
    x_pre = target_traj[..., :start_adjust, :]
    x_0 = target_traj[..., start_adjust:start_adjust+1, :]
    carry_0 = system.carry_callback()
    steps = target_traj.shape[0] - 1 - start_adjust

    prediction, carry = system.simulate(x_0, carry_0, steps)
    del carry

    full_traj = torch.concatenate((x_pre, prediction), dim=0)

    return full_traj.detach().clone()

def get_pll_single_step_predictions_and_targets(
        system: MultibodyLearnableSystem, full_traj: Tensor,
        start_adjust: int = 0) -> Tensor:
    """Get single step predictions and targets from a ground truth trajectory.
    A start adjust can be given to ignore some of the starting states."""
    assert full_traj.ndim == 2, f'Invalid {full_traj.shape=}.'
    assert full_traj.shape[1] == 13, f'Invalid {full_traj.shape=}.'

    # Use the input argument structure of system.simulate() -- this can be
    # batched as (n_batch, n_step, n_state).
    initial_states = full_traj[start_adjust:-1, :].reshape(-1, 1, 13)
    end_states = full_traj[start_adjust+1:, :].reshape(-1, 1, 13)

    carry_0 = system.carry_callback()
    simulated_states, _ = system.simulate(initial_states, carry_0, 1)

    # Return as (N-1-start_adjust, 13) tensors.  Need to drop the initial
    # condition from the simulated states.
    simulated_states = simulated_states[:, 0].detach().clone().squeeze()
    end_states = end_states.squeeze()
    return simulated_states, end_states


class PredictionOverlayGenerator(OverlayVideoGenerator):
    """Make an overlay video showing the tracked BundleSDF poses and the
    dynamics predictions during the tosses."""
    def __init__(self, vision_asset: str, history: dict, nerf_bundlesdf_id: str,
                 bsdf_only: bool = False, remote: bool = False):
        # Extract the relevant tracking, NeRF, and PLL IDs.
        last_bsdf_iteration = 1
        for cycle in history.keys():
            cycle_num = cycle.split('_')[-1]
            if int(cycle_num) > last_bsdf_iteration:
                last_bsdf_iteration = int(cycle_num)
        last_tracking_bsdf_id = history[
            f'cycle_iteration_{last_bsdf_iteration}']['bundlesdf_id']
        last_pll_id = history[
            f'cycle_iteration_{last_bsdf_iteration}']['pll_id']

        super().__init__(
            vision_asset=vision_asset,
            tracking_bundlesdf_id=last_tracking_bsdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id,
            cycle_iteration=last_bsdf_iteration,
            bsdf_only=bsdf_only, remote=remote
        )

        # Overwrite the output file so it gets written to evaluation directory.
        if last_pll_id is not None:
            last_tracking_bsdf_id = None
            nerf_bundlesdf_id = None
        else:
            last_tracking_bsdf_id = self.tracking_bundlesdf_id
            nerf_bundlesdf_id = self.nerf_bundlesdf_id
        self.output_file = file_utils.evaluation_toss_prediction_video_filepath(
            dataset=vision_asset, tracking_bundlesdf_id=last_tracking_bsdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id, pll_id=last_pll_id,
            cycle_iteration=last_bsdf_iteration
        )

        # Overwrite the mesh file if last run was PLL.
        if last_pll_id is not None:
            pll_output_dir = file_utils.contactnets_output_dir(
                self.vision_asset, self.cycle_iteration, last_pll_id)
            self.mesh_file = op.join(pll_output_dir, 'urdfs', 'test.obj')

        # Get the path to the evaluation directory.
        self.evaluation_dir = file_utils.evaluation_subdir(
            dataset=self.vision_asset,
            tracking_bundlesdf_id=last_tracking_bsdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id,
            pll_id=last_pll_id,
            cycle_iteration=self.cycle_iteration
        )

        # From the evaluation directory, get the predicted tosses.
        self._get_predicted_poses_in_world()

        # If there are more predicted tosses than there are tracked tosses, then
        # get the longer video length.
        self._adjust_for_longer_video()

    def _adjust_for_longer_video(self):
        """If more predictions are generated than are included in the training
        video, making overlays requires the object's 1-10 dataset to exist."""
        # Check if the prediction tosses include some beyond the tracking.
        min_prediction_toss = 11
        max_prediction_toss = 0
        for toss in self.prediction_tosses:
            if toss > max_prediction_toss:
                max_prediction_toss = toss
            if toss < min_prediction_toss:
                min_prediction_toss = toss
        if max_prediction_toss == self.end_toss and \
            min_prediction_toss == self.start_toss:
            print(f'Predictions are the same as {self.start_toss=} and ' + \
                  f'{self.end_toss=}; no need to extend video.')
            return

        print(f'Extending video to include tosses {min_prediction_toss} ' + \
              f'to {max_prediction_toss}: from {self.rgb_images.shape[0]}' + \
              f' images to ', end='')

        # Compute the new start/end frames.
        relative_start_frames = np.array([file_utils.load_field_from_yaml(
            self.object, toss_i, 'start_frame') for toss_i in range(
                min_prediction_toss, max_prediction_toss+1)])
        relative_end_frames = np.array([file_utils.load_field_from_yaml(
            self.object, toss_i, 'end_frame') for toss_i in range(
                min_prediction_toss, max_prediction_toss+1)])
        start_ros_times = np.array([file_utils.load_toss_time_from_yaml(
            self.object, toss_i, 'start_time', as_ros_time=True) for toss_i \
                in range(min_prediction_toss, max_prediction_toss+1)])
        cnets_data_gen_dir = file_utils.cnets_data_gen_dataset_dir(
            f'{self.object}_1-10', check_exists=True)
        bundlesdf_times = np.loadtxt(
            op.join(cnets_data_gen_dir, 'bundlesdf_timestamps.txt'))

        self.start_frames = math_utils.convert_relative_frames_to_absolute(
            relative_start_frames, bundlesdf_times, start_ros_times)
        self.end_frames = math_utils.convert_relative_frames_to_absolute(
            relative_end_frames, bundlesdf_times, start_ros_times)

        # Need to extend the video.  Only extend to the end of the last toss
        # that has a prediction.
        full_dataset_dir = file_utils.cnets_data_gen_dataset_dir(
            dataset=f'{self.object}_1-10', check_exists=True)
        full_dataset_times_filepath = op.join(
            full_dataset_dir, 'bundlesdf_timestamps.txt')
        assert op.exists(full_dataset_times_filepath), f'Missing ' + \
            f'{full_dataset_times_filepath=}.'
        full_dataset_times = np.loadtxt(full_dataset_times_filepath)

        last_toss_end = file_utils.load_toss_time_from_yaml(
            self.object, max_prediction_toss, 'end_time', as_ros_time=False)

        rgb_images = []
        full_rgb_dir = file_utils.bundlesdf_video_rgb_dir(f'{self.object}_1-10')
        for filename in sorted(os.listdir(full_rgb_dir)):
            frame_num = int(filename.split('.')[0])
            if full_dataset_times[frame_num-1] <= last_toss_end:
                im = Image.open(op.join(full_rgb_dir, filename))
                rgb_images.append(np.array(im))
                del im
            else:
                break

        self.rgb_images = np.array(rgb_images)
        print(f'{self.rgb_images.shape[0]} images.')

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

        predicted_trajs = {}
        prediction_tosses = []
        for toss in range(1, 11):
            pred_name = op.join(
                self.evaluation_dir, f'predicted_toss_{toss}.pt')
            if not op.exists(pred_name):
                print(f'Did not find predictions for toss {toss}; skipping.')
                continue
            prediction_tosses.append(toss)

            predicted_traj = np.array(torch.load(pred_name))

            # Convert to 4x4 transformation matrices.
            predicted_traj_trans = np.zeros((predicted_traj.shape[0], 4, 4))
            for i in range(predicted_traj_trans.shape[0]):
                predicted_pose = predicted_traj[i]
                predicted_trans_mat = math_utils.pll_format_to_trans_mat(
                    predicted_pose)

                # Adjust for the table height.
                predicted_trans_mat[2, 3] += z_table

                # Store to tensor.
                predicted_traj_trans[i] = predicted_trans_mat

            # Store the whole (N, 4, 4) trajectory to dictionary.
            predicted_trajs[toss] = predicted_traj_trans

        self.predicted_trajs = predicted_trajs
        self.prediction_tosses = prediction_tosses

        # Store the same trajectories as reported by BundleSDF and/or TagSLAM.
        tagslam_b_trajs = {}
        bundlesdf_trajs = {}
        for toss in self.prediction_tosses:
            tagslam_b_name = op.join(
                self.evaluation_dir, f'tagslam_b_toss_{toss}.pt')
            bundlesdf_name = op.join(
                self.evaluation_dir, f'bundlesdf_toss_{toss}.pt')

            assert op.exists(tagslam_b_name) or op.exists(bundlesdf_name), \
                f'No comparison trajectories found for toss {toss} at ' + \
                f'{tagslam_b_name=} or {bundlesdf_name=}.'

            for filename, storage in zip([tagslam_b_name, bundlesdf_name],
                                         [tagslam_b_trajs, bundlesdf_trajs]):
                if not op.exists(filename):
                    continue
                comp_traj = np.array(torch.load(filename))

                assert comp_traj.shape[0] == \
                    self.predicted_trajs[toss].shape[0], \
                    f'Cannot handle different shapes {comp_traj.shape=}, ' + \
                    f'{self.predicted_trajs[toss].shape=}.'

                # Convert to 4x4 transformation matrices.
                comp_traj_trans = np.zeros((comp_traj.shape[0], 4, 4))
                for i in range(comp_traj_trans.shape[0]):
                    trans_mat = math_utils.pll_format_to_trans_mat(comp_traj[i])

                    # Adjust for the table height.
                    trans_mat[2, 3] += z_table

                    # Store to tensor.
                    comp_traj_trans[i] = trans_mat

                # Store the whole (N, 4, 4) trajectory to dictionary.
                storage[toss] = comp_traj_trans

        self.bundlesdf_trajs = bundlesdf_trajs
        self.tagslam_b_trajs = tagslam_b_trajs

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

            self.vis["dynamics_triad"].set_transform(self.T_MW @ T_WP)
            self.vis["dynamics_mesh"].set_transform(self.T_MW @ T_WP)

        # If not showing a prediction, move the predicted geometry out of view.
        else:
            out_of_view_tf = self.T_MC @ tf.translation_matrix([0, 0, -1])

            self.vis["dynamics_triad"].set_transform(out_of_view_tf)
            self.vis["dynamics_mesh"].set_transform(out_of_view_tf)


class TagSLAMTrajectoryConverter(TrajectoryConverterBundleSDFToPLL):
    """Create dair_pll/assets/vision_{object}/{vision_asset}/{full or toss}/
    tagslam/ entries, without needing to have run a BundleSDF experiment on
    that asset.

    An example of how this would get called:

        tagslam_converter = TagSLAMTrajectoryConverter('bottle_1-10')
        tagslam_converter.do_process()
        tagslam_converter.save_data()
    """
    def __init__(self, vision_asset: str):
        object = vision_asset.split('_')[0]

        start_toss = int(vision_asset.split('_')[1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
                f'-{end_toss} inferred from {vision_asset=}.'

        # Get the camera extrinsics.
        cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(
            object)

        # Get the table height.  Use the average if using multiple tosses.
        table_heights = np.array([
            file_utils.load_table_z_height(object, toss) for toss in
            range(start_toss, end_toss+1)
        ])
        z_table = np.mean(table_heights)

        # Get the relevant timings.
        start_ros_times = np.array([file_utils.load_toss_time_from_yaml(
            object, toss_i, 'start_time', as_ros_time=True) for toss_i in range(
                start_toss, end_toss+1)])
        relative_start_frames = np.array([file_utils.load_field_from_yaml(
            object, toss_i, 'start_frame') for toss_i in range(
                start_toss, end_toss+1)])
        relative_end_frames = np.array([file_utils.load_field_from_yaml(
            object, toss_i, 'end_frame') for toss_i in range(
                start_toss, end_toss+1)])

        # Store class attributes.
        self.start_toss = start_toss
        self.end_toss = end_toss
        self.object = object
        self.cam_trans = cam_trans
        self.cam_rot_axis_angle = cam_rot_axis_angle
        self.frame_rate = 30
        self.z_table = z_table
        self.dataset = vision_asset
        self.has_full_keyframe_tosses = False

        self.tagslam_dir = file_utils.synchronized_tagslam_pose_dir(
            vision_asset, check_exists=True)

        self._load_tagslam_poses()
        self._get_absolute_frames(start_ros_times, relative_start_frames,
                                  relative_end_frames, self.tagslam_full_times)

    def _load_tagslam_poses(self):
        super()._load_tagslam_poses()
        tagslam_full_dts = np.mean(
            self.tagslam_full_times[1:] - self.tagslam_full_times[:-1]
        )
        print(f'TagSLAM full trajectory information:' + \
            f'\n\t{self.tagslam_t_full_poses.shape=}' + \
            f'\n\t{self.tagslam_full_times[0]=}' + \
            f'\n\tAverage frame rate (full): {1/tagslam_full_dts}\n')

    def _trim_processed_trajectories(self):
        self.tagslam_toss_processed_states = []
        self.tagslam_toss_times = []

        for i in range(len(self.start_frames)):
            # Need to do one less than provided start and end frames because
            # loaded data in 1-indexed directory but provided 0-indexed
            # start_frame and end_frame.
            t_start = self.start_frames[i] - 1
            t_end = self.end_frames[i] - 1
            self.tagslam_toss_processed_states.append(
                self.tagslam_full_processed_states[t_start:t_end])
            self.tagslam_toss_times.append(
                self.tagslam_full_times[t_start:t_end])

    def do_process(self):
        q_ts, p_ts, w_ts, v_ts = self._process_poses(
            self.tagslam_t_full_poses, self.tagslam_full_times)
        self.tagslam_full_processed_states = np.concatenate(
            (q_ts, p_ts, w_ts, v_ts), axis=1)

        self._trim_processed_trajectories()

        # Print information about the trimmed trajectories.
        for i in range(len(self.start_frames)):
            toss_i = i + self.start_toss
            print(f'\n=================== TOSS {toss_i} ===================')
            tagslam_toss_dts = np.mean(self.tagslam_toss_times[i][1:] - \
                                    self.tagslam_toss_times[i][:-1])
            print(f'TagSLAM toss {toss_i} trajectory information:' + \
                f'\n\t{self.tagslam_toss_processed_states[i].shape=}' + \
                f'\n\t{self.tagslam_toss_times[i][0]=}' + \
                f'\n\tAverage frame rate (toss): {1/tagslam_toss_dts}\n')

    def save_data(self):
        self.bsdf_only = False
        super().save_data(save_bundlesdf=False, save_tagslam=True)



if __name__ == '__main__':
    # pdb.set_trace()
    # pgen = PredictionOverlayGenerator('bakingbox_1-2', '00', '00', 2, [1, 2])
    # # pgen = PredictionOverlayGenerator('cube_1', '00', '00', 2, [1])
    # pgen.make_overlay_video()
    # pdb.set_trace()
    # pdb.set_trace()
    tagslam_converter = TagSLAMTrajectoryConverter('bottle_1-10')
    tagslam_converter.do_process()
    tagslam_converter.save_data()
    pdb.set_trace()
