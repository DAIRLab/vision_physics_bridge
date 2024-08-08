"""Utilities for evaluations."""

import numpy as np
import os
import os.path as op
import pdb
import pickle
from PIL import Image, ImageDraw
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
import sys
import torch
from torch import Tensor
from typing import Tuple
import click

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

# Triad trails.
TRAIL_LENGTH = 20
BUNDLESDF_TRIAD_TRAIL_NAMES = [
    f'bundlesdf_triad_{i}' for i in range(TRAIL_LENGTH)]
DYNAMICS_X_AXIS_TRAIL_NAMES = [
    f'dynamics_x_triad_{i}' for i in range(TRAIL_LENGTH)]
DYNAMICS_Y_AXIS_TRAIL_NAMES = [
    f'dynamics_y_triad_{i}' for i in range(TRAIL_LENGTH)]
DYNAMICS_Z_AXIS_TRAIL_NAMES = [
    f'dynamics_z_triad_{i}' for i in range(TRAIL_LENGTH)]
# TODO Can figure out later if we think we need to how to get the trail to be of
# fading opacities.
OPACITY_OVER_TIME = [(i+1)/TRAIL_LENGTH for i in range(TRAIL_LENGTH)]

PHYSICS_GREEN = '#7dab54'
VYSICS_PURPLE = '#68379A'
TOSS_BORDER = 10


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

def point_wise_chamfer_distance(point_cloud_1, point_cloud_2):
    """Point-wise chamfer distance between two point clouds."""
    kdtree1 = cKDTree(point_cloud_1)
    dists_from_2_to_1, _indices1 = kdtree1.query(point_cloud_2)
    kdtree2 = cKDTree(point_cloud_2)
    dists_from_1_to_2, _indices2 = kdtree2.query(point_cloud_1)
    return dists_from_2_to_1, dists_from_1_to_2

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

    if len(index) == 0:
        return 0.0

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

def get_mass_from_urdf(urdf_dir: str) -> float:
    """Get mass from the URDF file in the provided directory."""
    # First find the path to the URDF.
    urdf_path = None
    for file in os.listdir(urdf_dir):
        if file.endswith('.urdf'):
            assert urdf_path is None, f'Multiple URDF files in {urdf_dir}.'
            urdf_path = op.join(urdf_dir, file)
            break

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
                ).replace('"true_geom_aligned.obj"', f'"{new_obj_name}"'
                ).replace('"true_geom_aligned_assist.obj"', f'"{new_obj_name}"'
                ).replace('"test_best.obj"', f'"{new_obj_name}"'
                ).replace('"bsdf_mesh.obj"', f'"{new_obj_name}"')
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
        output_urdfs_dir = output_urdf_dir,
        represent_geometry_as = 'polygon',
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
    toss_key = vision_asset.split('_')[-1]
    start_toss = int(toss_key.split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(toss_key.split('-')[1])

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
    if object in file_utils.TAGLESS_OBJECTS or object.startswith('robot'):
        for category_key in METRICS_BY_TOSS:
            del empty_results[category_key]['against_tagslam']

    # Third modification:  Get rid of any dynamics predictions if PLL has never
    # been run and there are no TagSLAM trajectories to use.  If there are
    # TagSLAM trajectories to use, then dynamics predictions can be run using
    # the BundleSDF mesh and average dynamics parameters.
    if object in file_utils.TAGLESS_OBJECTS or object.startswith('robot'):
        if cycle_iteration <= 1 and last_run_was_bsdf:
            del empty_results['dynamics_rollout_metrics']
            del empty_results['dynamics_single_step_metrics']

    # Fourth modification:  Get rid of any tracking metrics if the last run was
    # PLL instead of BundleSDF.
    if not last_run_was_bsdf:
        del empty_results['tracking_metrics']

    # Fifth modification:  Get rid of any metrics against BundleSDF if it was
    # never run in this experiment's history.
    if cycle_iteration == 0:
        del empty_results['dynamics_rollout_metrics']['against_bundlesdf']
        del empty_results['dynamics_single_step_metrics']['against_bundlesdf']

    return empty_results


#============================ DYNAMICS PREDICTIONS ============================#
def get_pll_tagslam_trajectories_pll_format(object: str) -> dict:
    """Load TagSLAM trajectories from the PLL assets directory so they are
    already in PLL format.  Returns a dictionary with toss numbers as keys and
    torch tensors (N, 13) as values.  Returns None if the object is tagless and
    thus there are no TagSLAM trajectories."""
    if object in file_utils.TAGLESS_OBJECTS or object.startswith('robot'):
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

def get_tagslam_t_trajectories_pll_format(object: str) -> dict:
    """Load TagSLAM trajectories from the cnets-data-gen dataset directory,
    where TagSLAM trajectories wrt the TagSLAM origin are stored in PLL format.
    Returns a dictionary with toss numbers as keys and torch tensors (N, 13) as
    values.  Returns None if the object is tagless and thus there are no TagSLAM
    trajectories."""
    if object in file_utils.TAGLESS_OBJECTS or object.startswith('robot'):
        print(f'Object {object} is tagless; no TagSLAM trajectories.')
        return None

    # Get all of the trajectories, tosses 1 through 10 (or up until created).
    trajectories = {}

    print(f'First checking for TagSLAM trajectories for {object}_1-10.')
    vision_asset = f'{object}_1-10'
    tagslam_t_dir = file_utils.synchronized_tagslam_t_state_dir(
        vision_asset, create=False)
    if op.exists(tagslam_t_dir):
        for toss_i in range(1, 11):
            trajectory = torch.load(op.join(tagslam_t_dir, f'{toss_i}.pt'))
            trajectories[toss_i] = trajectory
        return trajectories

    print(f'{object}_1-10 not found; trying individual tosses instead.')
    for toss_i in range(1, 11):
        vision_asset = f'{object}_{toss_i}'
        tagslam_t_dir = file_utils.synchronized_tagslam_t_state_dir(
            vision_asset, create=False)

        if op.exists(tagslam_t_dir):
            trajectory = torch.load(op.join(tagslam_t_dir, f'{toss_i}.pt'))
            trajectories[toss_i] = trajectory
        else:
            print(f'No trajectory found for {vision_asset}; skipping.')

    return trajectories

def get_tagslam_b_trajectories_pll_format(object: str) -> dict:
    """Load TagSLAM trajectories from the cnets-data-gen dataset directory,
    where TagSLAM trajectories wrt BundleSDF origin are stored in PLL format.
    Returns a dictionary with toss numbers as keys and torch tensors (N, 13) as
    values.  Returns None if the object is tagless and thus there are no TagSLAM
    trajectories."""
    if object in file_utils.TAGLESS_OBJECTS or object.startswith('robot'):
        print(f'Object {object} is tagless; no TagSLAM trajectories.')
        return None

    # Get all of the trajectories, tosses 1 through 10 (or up until created).
    trajectories = {}

    print(f'First checking for TagSLAM trajectories for {object}_1-10.')
    vision_asset = f'{object}_1-10'
    tagslam_b_dir = file_utils.synchronized_tagslam_b_state_dir(
        vision_asset, create=False)
    if op.exists(tagslam_b_dir):
        for toss_i in range(1, 11):
            trajectory = torch.load(op.join(tagslam_b_dir, f'{toss_i}.pt'))
            trajectories[toss_i] = trajectory
        return trajectories

    print(f'{object}_1-10 not found; trying individual tosses instead.')
    for toss_i in range(1, 11):
        vision_asset = f'{object}_{toss_i}'
        tagslam_b_dir = file_utils.synchronized_tagslam_b_state_dir(
            vision_asset, create=False)

        if op.exists(tagslam_b_dir):
            trajectory = torch.load(op.join(tagslam_b_dir, f'{toss_i}.pt'))
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
    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
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

def get_synced_bsdf_keyframe_tagslam_toss_poses(
        vision_asset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> Tensor:
    """Get a time-synchronized pair of TagSLAM and BundleSDF keyframe poses for
    a given BundleSDF experiment.  Uses the last keyframe in the BundleSDF
    experiment's results.  Returns both poses as 4x4 transformation matrices."""
    # First load the keyframes:  get the adjusted keyframe poses from the
    # BundleSDF NeRF results' poses_after_nerf.txt.
    keyframe_tfs = \
        file_utils.load_optimized_keyframe_poses_from_nerf_results(
            dataset=vision_asset, cycle_iteration=cycle_iteration,
            tracking_bundlesdf_id=tracking_bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id
        )

    # Use the last keyframe.
    b_trans_mat_camera = keyframe_tfs[-1]

    # Convert to world frame.
    object = '_'.join(vision_asset.split('_')[:-1])
    cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(object)
    b_trans_mat = math_utils.camera_to_world(
        b_trans_mat_camera, translation=cam_trans, axis_vec=cam_rot_axis_angle)

    # Get the keyframe indices from the BundleSDF tracking results' last frame
    # directory, in keyframes.yml.
    keyframe_idx_1_indexed = \
        file_utils.load_keyframe_indices_from_nerf_results_yml(
            vision_asset, cycle_iteration, tracking_bundlesdf_id)
    keyframe_idx = [i-1 for i in keyframe_idx_1_indexed]
    keyframe_i = keyframe_idx[-1]

    # Lastly get the corresponding TagSLAM pose.
    """Load all the poses reported by TagSLAM.  These are in world coordinates
    of the TagSLAM body origin with the following ordering:
        [x, y, z, qx, qy, qz, qw]
    """
    tagslam_dir = file_utils.synchronized_tagslam_pose_dir(
        vision_asset, check_exists=True)
    tagslam_data = np.loadtxt(op.join(tagslam_dir, 'synced_tagslam.txt'))
    tagslam_poses = tagslam_data[:, 1:]
    t_trans_mat = math_utils.pos_quat_to_trans_mat(tagslam_poses[keyframe_i,:7])

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

    full_traj = torch.cat((x_pre, prediction), dim=0)

    return full_traj.detach().clone()

def get_pll_single_step_predictions_and_targets(
        system: MultibodyLearnableSystem, full_traj: Tensor) -> Tensor:
    """Get single step predictions and targets from a ground truth trajectory.
    No start adjust since only doing single-step predictions."""
    assert full_traj.ndim == 2, f'Invalid {full_traj.shape=}.'
    assert full_traj.shape[1] == 13, f'Invalid {full_traj.shape=}.'

    # Use the input argument structure of system.simulate() -- this can be
    # batched as (n_batch, n_step, n_state).
    initial_states = full_traj[:-1, :].reshape(-1, 1, 13)
    end_states = full_traj[1:, :].reshape(-1, 1, 13)

    carry_0 = system.carry_callback()
    simulated_states, _ = system.simulate(initial_states, carry_0, 1)

    # Return as (N-1, 13) tensors.  Need to drop the initial condition from the
    # simulated states.
    simulated_states = simulated_states[:, 1].detach().clone().squeeze()
    end_states = end_states.squeeze()
    return simulated_states, end_states

def get_first_trans_mat(
        vision_asset: str, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
        cycle_iteration: int) -> Tensor:
    # First load the keyframes:  get the adjusted keyframe poses from the
    # BundleSDF NeRF results' poses_after_nerf.txt.
    keyframe_tfs = \
        file_utils.load_optimized_keyframe_poses_from_nerf_results(
            dataset=vision_asset, cycle_iteration=cycle_iteration,
            tracking_bundlesdf_id=tracking_bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id
        )

    # Use the first keyframe.
    b_trans_mat_camera = keyframe_tfs[0]

    # Get the keyframe indices from the BundleSDF tracking results' last frame
    # directory, in keyframes.yml.
    keyframe_idx_1_indexed = \
        file_utils.load_keyframe_indices_from_nerf_results_yml(
            vision_asset, cycle_iteration, tracking_bundlesdf_id)
    keyframe_idx = [i-1 for i in keyframe_idx_1_indexed]
    keyframe_i = keyframe_idx[0]
    assert keyframe_i == 0, f'Expected initial image to be in keyframe pool.'

    return b_trans_mat_camera


class PredictionOverlayGenerator(OverlayVideoGenerator):
    """Make an overlay video showing the tracked BundleSDF poses and the
    dynamics predictions during the tosses.

    TODO:  Currently this is not working for experiments that were trained on
    tosses starting after toss 1.
    """
    def __init__(self, vision_asset: str, history: dict, nerf_bundlesdf_id: str,
                 bsdf_only: bool = False, remote: bool = False):
        # Extract the relevant tracking, NeRF, and PLL IDs.
        last_bsdf_iteration = 0
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
            if not op.exists(self.mesh_file):
                self.mesh_file = op.join(
                    pll_output_dir, 'urdfs', 'test_best.obj')
                assert op.exists(self.mesh_file), f'Could not find test.obj' +\
                    f'or test_best.obj in {pll_output_dir}/urdfs.'

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
        self.relative_start_frames = relative_start_frames
        self.relative_end_frames = relative_end_frames

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


class OfficialVideoPredictionOverlayGenerator(PredictionOverlayGenerator):
    """Make an overlay video showing the tracked BundleSDF poses as a small pink
    triad and the dynamics predictions as a larger RGB triad.  Have both of
    these triads make a disappearing trail of their past positions.
    Additionally, give the image a green border to indicate when the video is
    part of an autonomous dynamics portion, and a purple border to indicate when
    the video is featuring unseen data."""
    def __init__(self, vision_asset: str, history: dict, nerf_bundlesdf_id: str,
                 bsdf_only: bool = False, remote: bool = False):
        super().__init__(
            vision_asset=vision_asset, history=history,
            nerf_bundlesdf_id=nerf_bundlesdf_id, bsdf_only=bsdf_only,
            remote=remote
        )

        # Keep track of what trail index to next update.
        self.next_trail_index_to_update = 0

        # Update the filepath so the evaluation directory can keep its debugging
        # overlay video.
        self.output_file = file_utils.inspection_presentable_video_filepath(
            dataset=vision_asset,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id,
            cycle_iteration=self.cycle_iteration
        )

    def _add_meshcat_objects(self, vis: meshcat.Visualizer) -> None:
        """Only visualize triads, no learned/true geometries."""
        out_of_view_tf = tf.translation_matrix([0, 0, -1]) @ \
            tf.rotation_matrix(np.pi, (0,0,1)) @ \
            tf.translation_matrix([0, 0, -1])
        for bsdf_triad_name in BUNDLESDF_TRIAD_TRAIL_NAMES:
            vis[bsdf_triad_name].set_object(
                g.ObjMeshGeometry.from_file(
                    file_utils.pink_triad_obj_filepath()),
                g.MeshLambertMaterial(
                    color=0xff80d5, reflectivity=0.0, transparent=0,
                    opacity=1.0))
            vis[bsdf_triad_name].set_transform(out_of_view_tf)
        for dynamics_x_name in DYNAMICS_X_AXIS_TRAIL_NAMES:
            vis[dynamics_x_name].set_object(
                g.ObjMeshGeometry.from_file(file_utils.x_axis_obj_filepath()),
                g.MeshLambertMaterial(
                    color=0xff0000, reflectivity=0.0, transparent=0,
                    opacity=1.0))
            vis[dynamics_x_name].set_transform(out_of_view_tf)
        for dynamics_y_name in DYNAMICS_Y_AXIS_TRAIL_NAMES:
            vis[dynamics_y_name].set_object(
                g.ObjMeshGeometry.from_file(file_utils.y_axis_obj_filepath()),
                g.MeshLambertMaterial(
                    color=0x00ff00, reflectivity=0.0, transparent=0,
                    opacity=1.0))
            vis[dynamics_y_name].set_transform(out_of_view_tf)
        for dynamics_z_name in DYNAMICS_Z_AXIS_TRAIL_NAMES:
            vis[dynamics_z_name].set_object(
                g.ObjMeshGeometry.from_file(file_utils.z_axis_obj_filepath()),
                g.MeshLambertMaterial(
                    color=0x0000ff, reflectivity=0.0, transparent=0,
                    opacity=1.0))
            vis[dynamics_z_name].set_transform(out_of_view_tf)

    def _set_meshcat_object_poses(self, frame_i: int, T_WA: np.ndarray,
                                  T_CB: np.ndarray) -> None:
        """Set the poses of the objects.  Since they are implemented as trails,
        Update the last trail item's pose."""
        # First handle the tracked BundleSDF pose.
        if T_CB is not None:
            self.vis[f'bundlesdf_triad_{self.next_trail_index_to_update}'
                ].set_transform(self.T_MC @ T_CB)
        else:
            out_of_view_tf = self.T_MC @ tf.translation_matrix([0, 0, -1])
            self.vis[f'bundlesdf_triad_{self.next_trail_index_to_update}'
                ].set_transform(out_of_view_tf)

        # Second handle the predicted pose:  starting by determining if the
        # frame is within a toss.
        toss_i = self._within_which_toss(image_frame_i=frame_i+1)

        if (toss_i is not None) and (toss_i in self.prediction_tosses):
            toss_frame = frame_i+1 - self.start_frames[toss_i - self.start_toss]
            T_WP = self.predicted_trajs[toss_i][toss_frame]

            self.vis[f'dynamics_x_triad_{self.next_trail_index_to_update}'
                ].set_transform(self.T_MW @ T_WP)
            self.vis[f'dynamics_y_triad_{self.next_trail_index_to_update}'
                ].set_transform(self.T_MW @ T_WP)
            self.vis[f'dynamics_z_triad_{self.next_trail_index_to_update}'
                ].set_transform(self.T_MW @ T_WP)

        # If not showing a prediction, move the predicted geometry out of view.
        else:
            out_of_view_tf = self.T_MC @ tf.translation_matrix([0, 0, -1])
            self.vis[f'dynamics_x_triad_{self.next_trail_index_to_update}'
                ].set_transform(out_of_view_tf)
            self.vis[f'dynamics_y_triad_{self.next_trail_index_to_update}'
                ].set_transform(out_of_view_tf)
            self.vis[f'dynamics_z_triad_{self.next_trail_index_to_update}'
                ].set_transform(out_of_view_tf)

        # Update the next index.
        self.next_trail_index_to_update += 1
        if self.next_trail_index_to_update == TRAIL_LENGTH:
            self.next_trail_index_to_update = 0

        # Update the opacities.
        self._update_meshcat_object_opacities()

    def _update_meshcat_object_opacities(self) -> None:
        """This already assumes the next trail index to update has been
        incremented.  Update the opacities of the trails so the most recently
        updated one is the most prominent."""
        for i in range(TRAIL_LENGTH):
            trail_index = (self.next_trail_index_to_update + i) % TRAIL_LENGTH
            self.vis[f'bundlesdf_triad_{trail_index}'].set_property(
                'color', [1, 0.5, 0.84, OPACITY_OVER_TIME[i]])
            self.vis[f'dynamics_x_triad_{trail_index}'].set_property(
                'color', [1, 0, 0, OPACITY_OVER_TIME[i]])
            self.vis[f'dynamics_y_triad_{trail_index}'].set_property(
                'color', [0, 1, 0, OPACITY_OVER_TIME[i]])
            self.vis[f'dynamics_z_triad_{trail_index}'].set_property(
                'color', [0, 0, 1, OPACITY_OVER_TIME[i]])

    def _determine_if_unseen_frame(self, image_frame_i: int) -> bool:
        if self.end_toss + 1 not in self.prediction_tosses:
            return False

        first_unseen_frame = self.start_frames[self.end_toss] - \
            self.relative_start_frames[self.end_toss]
        return image_frame_i >= first_unseen_frame

    def _add_watermark(self, im: Image, image_frame_i: int) -> Image:
        """Add a label to the image to specify whether the portion of the video
        is part of the PLL toss or not (green border) or if it is unseen data
        (purple border)."""
        # Determine if the frame index is within a PLL toss.
        toss_i = self._within_which_toss(image_frame_i)

        # If the system is outside of the training data, draw a purple frame.
        if self._determine_if_unseen_frame(image_frame_i):
            fill_color = VYSICS_PURPLE

        # If the system is within a training set toss, draw a green frame.
        elif toss_i is not None:
            fill_color = PHYSICS_GREEN

        else:
            return im

        # Add a border border to the image.
        draw = ImageDraw.Draw(im)
        draw.polygon([
            (0, 0), (0, self.image_height),
            (self.image_width, self.image_height), (self.image_width, 0),
            (TOSS_BORDER, 0), (TOSS_BORDER, TOSS_BORDER),
            (self.image_width-TOSS_BORDER, TOSS_BORDER),
            (self.image_width-TOSS_BORDER, self.image_height-TOSS_BORDER),
            (TOSS_BORDER, self.image_height-TOSS_BORDER),
            (TOSS_BORDER, 0), (0, 0)],
            fill=fill_color)

        return im


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
        object = '_'.join(vision_asset.split('_')[:-1])

        start_toss = int(vision_asset.split('_')[-1].split('-')[0])
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


class InputVideoGenerator(OverlayVideoGenerator):
    """Generate a video out of the input data for a particular vision asset.
    There is no overlay necessary, but inheriting from the OverlayVideoGenerator
    class eliminates the need to rewrite a good bit of code."""
    def __init__(self, vision_asset: str, remote: bool = False):
        # Initialize the parent class with some dummy values, since these won't
        # matter for generating the video.
        super().__init__(
            vision_asset,
            tracking_bundlesdf_id='bundlesdf_id_00',
            nerf_bundlesdf_id='bundlesdf_id_00',
            cycle_iteration=1,
            bsdf_only=False, remote=remote)
        
        # Overwrite the output file.
        self.output_file = file_utils.inspection_input_video_filepath(
            vision_asset)
        
    def _add_meshcat_objects(self, vis: meshcat.Visualizer) -> None:
        """No meshcat objects are needed for the input video."""
        pass

    def _set_up_meshcat(self) -> None:
        """No meshcat setup is needed for the input video."""
        pass

    def _clean_up_meshcat(self) -> None:
        """No meshcat cleanup is needed for the input video."""
        pass

    def _render_one_image(self, frame_i: int, T_WA: np.ndarray,
                          T_CB: np.ndarray) -> Image:
        """This can do nothing except load the camera image."""
        return Image.fromarray(self.rgb_images[frame_i]).convert('RGB')


#######################################################################
@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="asset name e.g. cube_2.")
@click.option('--remote/--local',
              type=bool,
              default=False,
              help="whether to generate the videos remotely or locally.")

def main_command(vision_asset: str, remote: bool):
    print(f'Processing {vision_asset}')
    # tagslam_converter = TagSLAMTrajectoryConverter(vision_asset)
    # tagslam_converter.do_process()
    # tagslam_converter.save_data()

    input_video_generator = InputVideoGenerator(vision_asset, remote=remote)
    input_video_generator.make_overlay_video()


if __name__ == '__main__':
    main_command()
    # # pdb.set_trace()
    # # pgen = PredictionOverlayGenerator('bakingbox_1-2', '00', '00', 2, [1, 2])
    # # # pgen = PredictionOverlayGenerator('cube_1', '00', '00', 2, [1])
    # # pgen.make_overlay_video()
    # # pdb.set_trace()
    # # pdb.set_trace()
    # tagslam_converter = TagSLAMTrajectoryConverter('bottle_1-10')
    # tagslam_converter.do_process()
    # tagslam_converter.save_data()
    # pdb.set_trace()
