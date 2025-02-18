"""This script generates dynamics predictions for robot experiments based on
recorded impedance controller commands stored in a rosbag.

To run on a specific vision asset and PLL ID, set the variables appropriately
and copy over:
    - PLL-format trajectories (dair_pll/assets/vision_{OBJ}/{VISION_ASSET})
    - the dataset (cnets-data-generation/dataset/{VISION_ASSET})
    - PLL results (dair_pll/results/vision_{OBJ}/{VISION_ASSET}/
        bundlesdf_iteration_{ITERATION}/{PLL_ID})
    - the rosbag (cnets-data-generation/rosbags/raw_{BAG_NUM}.bag)

TODO:
    [x] Export visualizations to video.
    [x] Export predicted trajectories to files.
    [x] Store the intermediate files somewhere per experiment.
    [x] Be able to swap out what URDF to simulate (PLL, BSDF, Vysics).
    [x] Be able to simulate ground truth mesh.
    [x] Make file callable with vision asset.
    [ ] Make file callable with run IDs.
    [x] Use more reasonable other inertia for BSDF and GT (use geometric mean).
    [x] Be able to generate videos from existing computed trajectories.
    [ ] Be able to use existing URDFs.
"""

import click
import os
import os.path as op
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import trimesh
from typing import List, Tuple

from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.geometry import HalfSpace, MeshcatVisualizer, StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, CoulombFriction, \
    MultibodyPlant, MultibodyPlantConfig
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import TrajectorySource
from pydrake.trajectories import PiecewisePolynomial, \
    PiecewiseQuaternionSlerp, StackedTrajectory
from pydrake.visualization import AddFrameTriadIllustration, VideoWriter

import file_utils
import rosbag_processor

from evaluate import TrajectoryMetrics


# Some settings on the plot generation.
rc('legend', fontsize=12)
plt.rc('axes', titlesize=16)    # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels


SIM_TIME_STEP = 5e-4
CONTROLLER_TIME_STEP = 1e-3
vis_TIME_STEP = 5e-3

CAM_FOV = np.pi/6
VIDEO_PIXELS = [480, 640]
FPS = 30

# Front video view.
SENSOR_RPY_FRONT = np.array([-np.pi / 2, 0, np.pi / 2])
SENSOR_POSITION_FRONT = np.array([2., 0., 0.2])
SENSOR_POSE_FRONT_VIEW = RigidTransform(
    RollPitchYaw(SENSOR_RPY_FRONT).ToQuaternion(), SENSOR_POSITION_FRONT)

# Side video view -- match the RealSense camera's perspective.
SENSOR_POSITION_CAMERA, cam_rot_axis_angle = file_utils.load_camera_extrinsics(
    'robotocc')
SENSOR_ANGLE_AXIS_CAMERA = AngleAxis(
    angle=np.linalg.norm(cam_rot_axis_angle),
    axis=cam_rot_axis_angle/np.linalg.norm(cam_rot_axis_angle))
SENSOR_POSE_CAMERA_VIEW = RigidTransform(
    theta_lambda=SENSOR_ANGLE_AXIS_CAMERA, p=SENSOR_POSITION_CAMERA)
_fx, fy, _cx, _cy = file_utils.load_camera_intrinsics('robotocc')
CAM_FOV_REALSENSE = 2*np.arctan(VIDEO_PIXELS[0]/(2*fy))

WORLD_TO_FRANKA_PLL_OFFSET = 0.0087

MODELS_TO_TEST = ['vysics', 'bsdf', 'pll', 'gt']
TRACKING_BUNDLESDF_ID = 'bundlesdf_id_00'
BSDF_ITERATION = 1

FILES_TO_EXPORT = ['joint_times.txt', 'joint_angles.txt',
                   'joint_velocities.txt', 'tau_J_ds.txt',
                   'pose_command_times.txt', 'des_pos.txt', 'des_quat_wxyz.txt',
                   'cartesian_stiffness.txt', 'cartesian_damping.txt']
FILES_TO_GENERATE = [f'pred_franka_states_{mod}.txt' for mod in MODELS_TO_TEST]
FILES_TO_GENERATE += [f'pred_object_states_{mod}.txt' for mod in MODELS_TO_TEST]
FILES_TO_GENERATE += [f'pred_forces_from_ee_{mod}.txt' for mod in \
                      MODELS_TO_TEST]

def t11_tuple_from_vision_asset(vision_asset: str) -> Tuple[str, str, str]:
    return (f'pll_id_t11_{vision_asset}',
            'bundlesdf_id_00', 'bundlesdf_id_00-t11')

PLL_BSDF_NERF_IDS_FROM_VISION_ASSET = {
    'robotocc_oatly_3':
        ('pll_id_t09d_robotocc_oatly_3',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_oatly_4': t11_tuple_from_vision_asset('robotocc_oatly_4'),
    'robotocc_oatly_5':
        ('pll_id_t09d_robotocc_oatly_5',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_oatly_6': t11_tuple_from_vision_asset('robotocc_oatly_6'),
    'robotocc_oatly_7': t11_tuple_from_vision_asset('robotocc_oatly_7'),
    'robotocc_oatly_8': t11_tuple_from_vision_asset('robotocc_oatly_8'),
    'robotocc_oatly_9': t11_tuple_from_vision_asset('robotocc_oatly_9'),
    'robotocc_oatly_10': t11_tuple_from_vision_asset('robotocc_oatly_10'),
    'robotocc_oatly_11': t11_tuple_from_vision_asset('robotocc_oatly_11'),
    'robotocc_oatly_12': t11_tuple_from_vision_asset('robotocc_oatly_12'),
    'robotocc_milk_2': t11_tuple_from_vision_asset('robotocc_milk_2'),
    'robotocc_milk_3': t11_tuple_from_vision_asset('robotocc_milk_3'),
    'robotocc_milk_4': t11_tuple_from_vision_asset('robotocc_milk_4'),
    'robotocc_milk_5':
        ('pll_id_t09d_robotocc_milk_5',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_milk_8': t11_tuple_from_vision_asset('robotocc_milk_8'),
    'robotocc_styrofoam_1':
        ('pll_id_t09d_robotocc_styrofoam_1',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_styrofoam_2': t11_tuple_from_vision_asset('robotocc_styrofoam_2'),
    'robotocc_styrofoam_3':
        ('pll_id_t09d_robotocc_styrofoam_3',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_styrofoam_4': t11_tuple_from_vision_asset('robotocc_styrofoam_4'),
    'robotocc_styrofoam_5':
        ('pll_id_t09d_robotocc_styrofoam_5',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_styrofoam_6':
        ('pll_id_t09d_robotocc_styrofoam_6',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_styrofoam_7': t11_tuple_from_vision_asset('robotocc_styrofoam_7'),
    'robotocc_styrofoam_8': t11_tuple_from_vision_asset('robotocc_styrofoam_8'),
    'robotocc_styrofoam_9': t11_tuple_from_vision_asset('robotocc_styrofoam_9'),
    'robotocc_styrofoam_10': t11_tuple_from_vision_asset('robotocc_styrofoam_10'),
    'robotocc_toblerone_1': t11_tuple_from_vision_asset('robotocc_toblerone_1'),
    'robotocc_toblerone_2':
        ('pll_id_t09d_robotocc_toblerone_2',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_toblerone_5':
        ('pll_id_t09d_robotocc_toblerone_5',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_toblerone_6': t11_tuple_from_vision_asset('robotocc_toblerone_6'),
    'robotocc_toblerone_11': t11_tuple_from_vision_asset('robotocc_toblerone_11'),
    'robotocc_egg_1':
        ('pll_id_t09d_robotocc_egg_1',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_egg_6': t11_tuple_from_vision_asset('robotocc_egg_6'),
    'robotocc_bakingbox_1': t11_tuple_from_vision_asset('robotocc_bakingbox_1'),
    'robotocc_bakingbox_2': t11_tuple_from_vision_asset('robotocc_bakingbox_2'),
    'robotocc_bakingbox_3': t11_tuple_from_vision_asset('robotocc_bakingbox_3'),
    'robotocc_bakingbox_4': t11_tuple_from_vision_asset('robotocc_bakingbox_4'),
    'robotocc_bakingbox_5':
        ('pll_id_t09d_robotocc_bakingbox_5',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_bakingbox_7':
        ('pll_id_t09d_robotocc_bakingbox_7',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_bakingbox_8':
        ('pll_id_t09d_robotocc_bakingbox_8',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_bakingbox_9': t11_tuple_from_vision_asset('robotocc_bakingbox_9'),
    'robotocc_bakingbox_10': t11_tuple_from_vision_asset('robotocc_bakingbox_10'),
    'robotocc_bakingbox_11': t11_tuple_from_vision_asset('robotocc_bakingbox_11'),
    'robotocc_bakingbox_12': t11_tuple_from_vision_asset('robotocc_bakingbox_12'),
    'robotocc_bakingbox_13': t11_tuple_from_vision_asset('robotocc_bakingbox_13'),
    'robotocc_bakingbox_14': t11_tuple_from_vision_asset('robotocc_bakingbox_14'),
    'robotocc_bakingbox_15': t11_tuple_from_vision_asset('robotocc_bakingbox_15'),
    'robotocc_bottle_1':
        ('pll_id_t09d_robotocc_bottle_1',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_bottle_2': t11_tuple_from_vision_asset('robotocc_bottle_2'),
    'robotocc_bottle_3':
        ('pll_id_t09d_robotocc_bottle_3',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_bottle_4': t11_tuple_from_vision_asset('robotocc_bottle_4'),
    'robotocc_bottle_5': t11_tuple_from_vision_asset('robotocc_bottle_5'),
    'robotocc_bottle_6':
        ('pll_id_t09d_robotocc_bottle_6',
         'bundlesdf_id_00',
         'bundlesdf_id_00-t09d'),
    'robotocc_bottle_7': t11_tuple_from_vision_asset('robotocc_bottle_7'),
}

## Code for generating command line for robot overlay videos.
# breakpoint()
# for vision_asset, items in PLL_BSDF_NERF_IDS_FROM_VISION_ASSET.items():
#     pll_id, bsdf_id, bsdf_nerf_id = items
#     pll_id = pll_id.replace('pll_id_', '')
#     bsdf_id = bsdf_id.replace('bundlesdf_id_', '')
#     bsdf_nerf_id = bsdf_nerf_id.replace('bundlesdf_id_', '')
#     print(f'python overlay_videos.py --vision-asset={vision_asset} ' + \
#           f'--cycle-iteration={BSDF_ITERATION} --bundlesdf-id={bsdf_id} ' + \
#           f'--nerf-bundlesdf-id={bsdf_nerf_id} --bsdf-only --show-robot ' + \
#           f'--remote')
# breakpoint()

def hex_to_rgba_format(hex: str, opacity: float):
    r, g, b = tuple(int(hex[i:i+2], 16) for i in (1, 3, 5))
    return f'{r/255} {g/255} {b/255} {opacity}'

PLL_MESH_HEX = '#70ad47'
BSDF_MESH_HEX = '#4472c4'
VYSICS_MESH_HEX = '#7030a0'
GT_MESH_HEX = '#990000'
MANUAL_ANNOTATION_HEX = '#999999'

PLL_MESH_RGBA = hex_to_rgba_format(PLL_MESH_HEX, 0.6)
BSDF_MESH_RGBA = hex_to_rgba_format(BSDF_MESH_HEX, 0.6)
VYSICS_MESH_RGBA = hex_to_rgba_format(VYSICS_MESH_HEX, 0.6)
GT_MESH_RGBA = hex_to_rgba_format(GT_MESH_HEX, 0.6)
COMPARISON_MESH_RGBA = '0.6 0.6 0.6 0.6'

DEFAULT_COM_STRING = '<inertial>\n            <origin xyz="0 0 0"'

SMALLEST_LINEWIDTH = 0.2
BIGGEST_LINEWIDTH = 3.0

POSITION_TOLERANCE = 0.1
ROTATION_TOLERANCE = np.pi/4


def obj_file_to_com_string(obj_path: str) -> str:
    obj = trimesh.load(obj_path, force='mesh')
    com = obj.center_mass
    return DEFAULT_COM_STRING.replace('0 0 0', f'{com[0]} {com[1]} {com[2]}')

def ids_from_vision_asset(vision_asset: str) -> str:
    try:
        pll_id, bsdf_id, bsdf_nerf_id = \
            PLL_BSDF_NERF_IDS_FROM_VISION_ASSET[vision_asset]
        return pll_id, bsdf_id, bsdf_nerf_id
    except KeyError:
        raise ValueError(f'Not prepared to analyze results for ' + \
            f'{vision_asset=}, add to PLL_BSDF_NERF_IDS_FROM_VISION_ASSET ' + \
            f'dictionary.')

def visualize_drake_systems(plant=None, diagram=None):
    plt.figure()
    if diagram is not None:
        plot_system_graphviz(diagram)
    elif plant is not None:
        plot_graphviz(plant.GetTopologyGraphvizString())
    else:
        raise ValueError('Need to provide either a plant or a diagram.')
    plt.plot(1)
    plt.show(block=False)

def write_obj_file_with_normals(original_obj_path: str, new_obj_path: str,
                                verbose: bool = False):
    original_obj = trimesh.load(original_obj_path, force='mesh')

    if verbose:  print(f'Converting obj:  ', end='')

    with open(new_obj_path, 'w') as f:
        f.write(f'# Vertices\n')
        for vertex in original_obj.vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        if verbose:  print(f'wrote vertices, ', end='')

        f.write(f'\n# Vertex normals\n')
        for normal in original_obj.vertex_normals:
            f.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')
        if verbose:  print(f'normals, ', end='')

        f.write(f'\n# Faces:  vertex index // vertex normal index\n')
        for face in original_obj.faces:
            # +1 because obj files use 1-indexing but trimesh uses 0.
            # This is of format v_i//vn_i, which for us are always the same.
            f.write(f'f {face[0]+1}//{face[0]+1} ' + \
                    f'{face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n')
        if verbose:  print(f'and faces.')

def make_gt_geometry_urdf(
        vision_asset: str, bsdf_iteration: int, pll_id: str, track_bsdf_id: str,
        nerf_bsdf_id: str, save_dir: str, verbose: bool = False,
        from_eval_folder: bool = True, for_comparison: bool = True) -> str:
    name = 'comparison' if for_comparison else 'gt'
    color_rgba = COMPARISON_MESH_RGBA if for_comparison else GT_MESH_RGBA

    # Get the GT-aligned mesh from the NeRF results directory.
    orig_urdf_path = op.join(
        file_utils.contactnets_output_dir(
            vision_asset, bsdf_iteration, pll_id),
        'with_bundlesdf_mesh.urdf'
    )
    new_urdf_path = op.join(save_dir, f'{name}.urdf')

    if from_eval_folder:
        evaluation_results_dir = file_utils.evaluation_subdir(
            dataset=vision_asset, cycle_iteration=bsdf_iteration,
            tracking_bundlesdf_id=track_bsdf_id, nerf_bundlesdf_id=nerf_bsdf_id
        )
        orig_obj_path = op.join(
            evaluation_results_dir, 'true_geom_aligned_assist.obj')
    else:
        nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
            dataset=vision_asset, cycle_iteration=bsdf_iteration,
            tracking_bundlesdf_id=track_bsdf_id, nerf_bundlesdf_id=nerf_bsdf_id
        )
        orig_obj_path = op.join(
            nerf_results_dir, 'true_geom_aligned_meshlab.obj')

    new_com_string = obj_file_to_com_string(orig_obj_path)
    new_obj_path = '/'.join(new_urdf_path.split('/')[:-1]) + '/gt_mesh.obj'

    # Change the URDF body name and color.
    with open(new_urdf_path, 'w') as write_file:
        with open(orig_urdf_path, 'r') as read_file:
            line = read_file.read(
                ).replace('name="vision_object"',
                          f'name="{name}"'
                ).replace('color rgba="0.6 0 0 1.0"',
                          f'color rgba="{color_rgba}"'
                ).replace('name="body"', f'name="{name}_body"'
                ).replace('mesh filename="bundlesdf_mesh.obj"',
                          'mesh filename="gt_mesh.obj"'
                ).replace(DEFAULT_COM_STRING, new_com_string)
            write_file.write(line)

    # Need to do something special for the obj file.  BundleSDF exports obj
    # files without any normals.
    write_obj_file_with_normals(orig_obj_path, new_obj_path, verbose=verbose)

    if verbose:
        print(f'Getting for comparison:\nURDF from {orig_urdf_path}\nOBJ ' + \
              f'from {orig_obj_path}\n')

    return new_urdf_path

def make_bsdf_geometry_urdf(
        vision_asset: str, bsdf_iteration: int, pll_id: str, track_bsdf_id: str,
        save_dir: str, verbose: bool = False,
        inertia_from: str = 'geometry') -> str:

    nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
        dataset=vision_asset, cycle_iteration=bsdf_iteration,
        tracking_bundlesdf_id=track_bsdf_id, nerf_bundlesdf_id=track_bsdf_id
    )
    orig_obj_path = op.join(nerf_results_dir, 'textured_mesh.obj')

    new_com_string = DEFAULT_COM_STRING

    # Determine where to get the original URDF, based on which inertial
    # parameters to use.
    if inertia_from == 'learned':
        orig_urdf_path = op.join(
            file_utils.get_pll_urdf_output_dir(
                vision_asset, bsdf_iteration, pll_id),
            'with_bundlesdf_mesh.urdf'
        )
        # The new CoM is the same as the PLL URDF, which is not the same as the
        # default, so no need to overwrite new_com_string.
    elif inertia_from == 'tracking':
        orig_urdf_path = op.join(
            file_utils.contactnets_output_dir(
                vision_asset, bsdf_iteration, pll_id),
            'with_bundlesdf_mesh.urdf'
        )
    elif inertia_from == 'geometry':
        orig_urdf_path = op.join(
            file_utils.contactnets_output_dir(
                vision_asset, bsdf_iteration, pll_id),
            'with_bundlesdf_mesh.urdf'
        )
        new_com_string = obj_file_to_com_string(orig_obj_path)
    else:
        raise NotImplementedError(
            f'Not prepared to get URDF inertia from {inertia_from=}')

    new_urdf_path = op.join(save_dir, 'bsdf.urdf')
    new_obj_path = '/'.join(new_urdf_path.split('/')[:-1]) + \
        '/bundlesdf_mesh.obj'

    # Change the URDF body name and color.
    with open(new_urdf_path, 'w') as write_file:
        with open(orig_urdf_path, 'r') as read_file:
            line = read_file.read(
                ).replace('name="vision_object"',
                          'name="from_bundlesdf"'
                ).replace('color rgba="0.6 0 0 1.0"',
                          f'color rgba="{BSDF_MESH_RGBA}"'
                ).replace('name="body"', 'name="bsdf_body"'
                ).replace('mesh filename="body_best.obj"',
                          'mesh filename="bundlesdf_mesh.obj"'
                ).replace(DEFAULT_COM_STRING, new_com_string)
            write_file.write(line)

    # Need to do something special for the obj file.  BundleSDF exports obj
    # files without any normals.
    write_obj_file_with_normals(orig_obj_path, new_obj_path, verbose=verbose)

    if verbose:
        print(f'Getting for BundleSDF:\nURDF from {orig_urdf_path}\nOBJ ' + \
              f'from {orig_obj_path}\n')

    return new_urdf_path

def make_pll_geometry_urdf(vision_asset: str, bsdf_iteration: int, pll_id: str,
                           save_dir: str, verbose: bool = False) -> str:
    orig_urdf_path = op.join(
        file_utils.get_pll_urdf_output_dir(
            vision_asset, bsdf_iteration, pll_id),
        'with_bundlesdf_mesh.urdf'
    )
    orig_obj_path = '/'.join(orig_urdf_path.split('/')[:-1]) + '/body_best.obj'

    new_urdf_path = op.join(save_dir, 'pll.urdf')
    new_obj_path = '/'.join(new_urdf_path.split('/')[:-1]) + '/pll_mesh.obj'

    # Change the URDF body name and color.
    with open(new_urdf_path, 'w') as write_file:
        with open(orig_urdf_path, 'r') as read_file:
            line = read_file.read(
                ).replace('name="vision_object"',
                          'name="from_pll"'
                ).replace('color rgba="0.6 0 0 1.0"',
                          f'color rgba="{PLL_MESH_RGBA}"',
                ).replace('name="body"', 'name="pll_body"'
                ).replace('mesh filename="body_best.obj"',
                          'mesh filename="pll_mesh.obj"')
            write_file.write(line)
    os.system(f'cp {orig_obj_path} {new_obj_path}')

    if verbose:
        print(f'Getting for PLL:\nURDF from {orig_urdf_path}\nOBJ from ' + \
              f'{orig_obj_path}\n')

    return new_urdf_path

def make_vysics_geometry_urdf(
        vision_asset: str, bsdf_iteration: int, pll_id: str, track_bsdf_id: str,
        nerf_bsdf_id: str, save_dir: str, verbose: bool = False) -> str:
    orig_urdf_path = op.join(
        file_utils.get_pll_urdf_output_dir(
            vision_asset, bsdf_iteration, pll_id),
        'with_bundlesdf_mesh.urdf'
    )
    nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
        dataset=vision_asset, cycle_iteration=bsdf_iteration,
        tracking_bundlesdf_id=track_bsdf_id, nerf_bundlesdf_id=nerf_bsdf_id
    )
    orig_obj_path = op.join(nerf_results_dir, 'textured_mesh.obj')

    new_urdf_path = op.join(save_dir, 'vysics.urdf')
    new_obj_path = '/'.join(new_urdf_path.split('/')[:-1]) + '/vysics_mesh.obj'

    # Change the URDF body name and color.
    with open(new_urdf_path, 'w') as write_file:
        with open(orig_urdf_path, 'r') as read_file:
            line = read_file.read(
                ).replace('name="vision_object"',
                          'name="from_vysics"'
                ).replace('color rgba="0.6 0 0 1.0"',
                          f'color rgba="{VYSICS_MESH_RGBA}"',
                ).replace('name="body"', 'name="vysics_body"'
                ).replace('mesh filename="body_best.obj"',
                          'mesh filename="vysics_mesh.obj"')
            write_file.write(line)

    # Need to do something special for the obj file.  BundleSDF exports obj
    # files without any normals.
    write_obj_file_with_normals(orig_obj_path, new_obj_path, verbose=verbose)

    if verbose:
        print(f'Getting for Vysics:\nURDF from {orig_urdf_path}\nOBJ from ' + \
              f'{orig_obj_path}\n')

    return new_urdf_path

def binary_iou(first_bools: np.array, second_bools: np.array) -> float:
    agreement = first_bools == second_bools
    return np.sum(agreement) / np.size(agreement)


### Inverse dynamics controller ###
class InverseDynamicsController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, controller_params: dict):
        LeafSystem.__init__(self)
        self.plant = plant

        # Store necessary controller parameters and hard-coded settings.
        self._set_param_settings(controller_params)
        self._set_default_settings()

        # Declare inputs and outputs.
        self.ee_pose_command_input_index = self.DeclareVectorInputPort(
            'ee_pose_command', 7).get_index()
        self.robot_state_input_index = self.DeclareVectorInputPort(
            'robot_state', plant.num_positions() + plant.num_velocities()
        ).get_index()
        self.joint_torque_output_index = self.DeclareVectorOutputPort(
            'joint_torques', plant.num_actuators(), self.CalcControl
        ).get_index()

    def get_ee_pose_command_input_port(self):
        return self.get_input_port(self.ee_pose_command_input_index)

    def get_robot_state_input_port(self):
        return self.get_input_port(self.robot_state_input_index)

    def get_joint_torque_output_port(self):
        return self.get_output_port(self.joint_torque_output_index)

    def _set_param_settings(self, controller_params):
        self.q_d_nullspace = np.array(controller_params[
            'hybrid_impedance_wrench_controller']['q_d_nullspace'])
        self.nullspace_stiffness = np.array(controller_params[
            'hybrid_impedance_wrench_controller']['nullspace_stiffness'])

        self.cartesian_trans_acc_norm_limit = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_trans_acc_norm_limit']
        self.cartesian_rot_acc_norm_limit = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_rot_acc_norm_limit']

    def _set_default_settings(self):
        self.delta_tau_max = 1000.0
        self.filter_param = 0.005

        self.position_d = np.zeros(3)
        self.position_d_target = np.zeros(3)
        self.quaternion_d = Quaternion()
        self.quaternion_d_target = Quaternion()

    def StartFrom(self, joint_angles: np.ndarray, joint_velocities: np.ndarray,
                  joint_torques: np.ndarray, cartesian_stiffness: np.ndarray,
                  cartesian_damping: np.ndarray):
        """Resets the controller to the initial state and sets the controller to
        the initial joint angles."""
        self.cartesian_stiffness = cartesian_stiffness
        self.cartesian_damping = cartesian_damping

        self._set_controller_plant_state_and_context(
            joint_angles, joint_velocities)

        # Store some initial set of joint torques.
        # TODO better practice to use Drake state instead of class variable, or
        # to pipe the tau_d output to a DiscreteTimeDelay that feeds into this
        # system's input ports.
        self.tau_J_d = joint_torques

        # Set the equilibrium point to the initial condition.
        ee_pose = self.plant.EvalBodyPoseInWorld(
            self.plant_context, self.plant.GetBodyByName('end_effector_tip'))
        self.position_d = ee_pose.translation()
        self.position_d_target = ee_pose.translation()
        self.orientation_d = ee_pose.rotation().ToQuaternion()
        self.orientation_d_target = ee_pose.rotation().ToQuaternion()

    def _set_controller_plant_state_and_context(
            self, joint_angles: np.ndarray, joint_velocities: np.ndarray):
        if not hasattr(self, 'plant_context'):
            self.plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(self.plant_context, joint_angles)
        self.plant.SetVelocities(self.plant_context, joint_velocities)

    def _update_desired_targets(self, context):
        """Based on the new commanded pose, update the desired targets.  The
        pose is in the form [qw qx qy qz x y z]."""
        pose_command = self.EvalVectorInput(
            context, self.ee_pose_command_input_index).get_value().copy()
        self.position_d_target = pose_command[4:7]

        # Ensure quaternion representation is consistent.
        last_orientation_d_target = self.orientation_d_target.wxyz()
        new_orientation_d_target = pose_command[:4]
        assert np.abs(np.linalg.norm(new_orientation_d_target) - 1) < 1e-3, \
            f'Expected normalized quaternion but got ' + \
            f'{np.linalg.norm(new_orientation_d_target)} from ' + \
            f'{new_orientation_d_target} -- is the pose command in the ' + \
            f'right order?  {pose_command=}'
        if np.dot(last_orientation_d_target, new_orientation_d_target) < 0:
            new_orientation_d_target *= -1
        self.orientation_d_target = Quaternion(new_orientation_d_target)

    def _saturate_torque_rate(self, tau_d_calculated):
        tau_d_saturated = np.zeros(7)
        for i in range(7):
            difference = tau_d_calculated[i] - self.tau_J_d[i]
            tau_d_saturated[i] = self.tau_J_d[i] + np.clip(
                difference, -self.delta_tau_max, self.delta_tau_max)
        return tau_d_saturated

    def _saturate_cartesian_acceleration(self, cart_acc_des):
        cart_acc_saturated = cart_acc_des
        if np.abs(cart_acc_des[2] > self.cartesian_trans_acc_norm_limit):
            cart_acc_saturated[2] = np.sign(cart_acc_des[2]) * \
                self.cartesian_trans_acc_norm_limit
        return cart_acc_saturated

    def CalcControl(self, context, output):
        """Compute the control law for the robot, emulating the controller used
        for the Franka teleop setup."""
        # First get the current commanded pose and robot state.
        self._update_desired_targets(context)

        x = self.EvalVectorInput(
            context, self.robot_state_input_index).get_value()
        q = x[:self.plant.num_positions()]
        v = x[self.plant.num_positions():]

        # Compute Drake's quantities like mass matrix, jacobian, coriolis,
        # gravity, etc.  Slower but more accurate option if needed:
        # CalcMassMatrixViaInverseDynamics
        self._set_controller_plant_state_and_context(q, v)
        mass_matrix = self.plant.CalcMassMatrix(self.plant_context)
        M_inv = np.linalg.inv(mass_matrix)
        coriolis = self.plant.CalcBiasTerm(self.plant_context)
        gravity = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        Jv_v_AB = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.plant.GetFrameByName('end_effector_tip'),
            p_BoBi_B=np.zeros(3),
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame()
        )  # (3, 7)
        Jv_w_AB = self.plant.CalcJacobianAngularVelocity(
            self.plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.plant.GetFrameByName('end_effector_tip'),
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame()
        )  # (3, 7)
        jacobian = np.concatenate((Jv_v_AB, Jv_w_AB), axis=0)  # (6, 7)

        # Get the current end effector pose.
        ee_pose = self.plant.EvalBodyPoseInWorld(
            self.plant_context, self.plant.GetBodyByName('end_effector_tip'))
        position = ee_pose.translation()
        orientation = ee_pose.rotation().ToQuaternion()

        # Compute errors.
        position_error = position - self.position_d
        quaternion_error = self.orientation_d.multiply(orientation.inverse())
        angle_axis_error = AngleAxis(quaternion=quaternion_error)
        orientation_error = -angle_axis_error.axis() * angle_axis_error.angle()
        error = np.concatenate((position_error, orientation_error))
        assert error.shape == (6,)

        # Compute control -- no feedforward term needed because desired wrench
        # is always zero.
        # tau_wrench = jacobian.T @ np.concatenate((force_d, torque_d))

        # Compute control -- second task.
        cart_acc_des = -self.cartesian_stiffness * error - \
            self.cartesian_damping * (jacobian @ v)
        cart_acc_des = self._saturate_cartesian_acceleration(cart_acc_des)
        delassus_inv = np.linalg.inv(jacobian @ M_inv @ jacobian.T)
        tau_task = jacobian.T @ delassus_inv @ cart_acc_des

        # Compute control -- third nullspace.
        tau_q_pd = self.nullspace_stiffness * (self.q_d_nullspace - q) - \
            (2.0 * np.sqrt(self.nullspace_stiffness) * v)
        jacobian_dyncost_pinv_transpose = delassus_inv @ jacobian @ M_inv
        tau_nullspace = (
            np.eye(7) - jacobian.T @ jacobian_dyncost_pinv_transpose) @ tau_q_pd

        # Desired torque.
        tau_d = tau_task + tau_nullspace + coriolis - gravity  # + tau_wrench
        tau_d = self._saturate_torque_rate(tau_d)
        # tau_d = coriolis - gravity  # sanity check can make robot not move

        # Set the output to the desired torques.
        assert tau_d.ndim == 1
        assert tau_d.shape[0] == self.plant.num_actuators() == 7
        output.SetFromVector(tau_d)

        # Do updates for the next iteration.
        self._end_of_calc_updates(tau_d)

    def _end_of_calc_updates(self, tau_d):
        self.tau_J_d = tau_d

        self.position_d = self.filter_param * \
            self.position_d_target + (1 - self.filter_param) * \
            self.position_d

        # Special case to handle the quaternion.
        wxyz_d = self.orientation_d.wxyz()
        wxyz_d_target = self.orientation_d_target.wxyz()
        wxyz_d = self.filter_param * \
            wxyz_d_target + (1 - self.filter_param) * wxyz_d
        self.orientation_d = Quaternion(wxyz_d/np.linalg.norm(wxyz_d))


class RobotDynamicsPredictor():
    """Generate dynamics predictions for robot experiments based on recorded
    end effector pose commands stored in a ROS bag, emulating the impedance
    controller used for hardware experiments in simulation.  Generate videos and
    trajectories of the results."""
    def __init__(self, vision_asset: str, pll_id: str, bundlesdf_id: str,
                 nerf_bundlesdf_id: str, bsdf_iteration: str,
                 debug: bool = False, open_meshcat: bool = True):
        self.vision_asset = vision_asset
        self.pll_id = pll_id
        self.bundlesdf_id = bundlesdf_id
        self.nerf_bundlesdf_id = nerf_bundlesdf_id
        self.bsdf_iteration = bsdf_iteration
        self.debug = debug
        self.open_meshcat = open_meshcat

        self._export_test_data()
        self._load_data()
        self._create_drake_trajectories()

    def run_simulation(self, model_to_test: str, overwrite: bool = False):
        if model_to_test == 'pll':
            self.learned_urdf_path = make_pll_geometry_urdf(
                vision_asset=self.vision_asset,
                bsdf_iteration=self.bsdf_iteration,
                pll_id=self.pll_id,
                save_dir=self.save_dir,
                verbose=self.debug
            )
        elif model_to_test == 'vysics':
            self.learned_urdf_path = make_vysics_geometry_urdf(
                vision_asset=self.vision_asset,
                bsdf_iteration=self.bsdf_iteration,
                pll_id=self.pll_id,
                track_bsdf_id=self.bundlesdf_id,
                nerf_bsdf_id=self.nerf_bundlesdf_id,
                save_dir=self.save_dir,
                verbose=self.debug
            )
        elif model_to_test == 'bsdf':
            self.learned_urdf_path = make_bsdf_geometry_urdf(
                vision_asset=self.vision_asset,
                bsdf_iteration=self.bsdf_iteration,
                pll_id=self.pll_id,
                track_bsdf_id=self.bundlesdf_id,
                save_dir=self.save_dir,
                verbose=self.debug
            )
        elif model_to_test == 'gt':
            self.learned_urdf_path = make_gt_geometry_urdf(
                vision_asset=self.vision_asset,
                bsdf_iteration=self.bsdf_iteration,
                pll_id=self.pll_id,
                track_bsdf_id=self.bundlesdf_id,
                nerf_bsdf_id=self.nerf_bundlesdf_id,
                save_dir=self.save_dir,
                verbose=self.debug,
                for_comparison=False
            )
        else:
            raise NotImplementedError(
                f'Not prepared to simulate {model_to_test=}')

        self.comparison_urdf_path = make_gt_geometry_urdf(
            vision_asset=self.vision_asset,
            bsdf_iteration=self.bsdf_iteration,
            pll_id=self.pll_id,
            track_bsdf_id=self.bundlesdf_id,
            nerf_bsdf_id=self.nerf_bundlesdf_id,
            save_dir=self.save_dir,
            verbose=self.debug,
            for_comparison=True
        )

        pred_object_states_filename = op.join(
            self.save_dir, f'pred_object_states_{model_to_test}.txt')
        pred_franka_states_filename = op.join(
            self.save_dir, f'pred_franka_states_{model_to_test}.txt')
        pred_forces_on_object_by_ee_filename = op.join(
            self.save_dir, f'pred_forces_from_ee_{model_to_test}.txt')

        do_simulation = False if op.exists(pred_object_states_filename) and \
            op.exists(pred_franka_states_filename) and \
            op.exists(pred_forces_on_object_by_ee_filename) else True
        if not do_simulation:
            pred_object_states = np.loadtxt(pred_object_states_filename)
            pred_franka_states = np.loadtxt(pred_franka_states_filename)

            if overwrite:
                print(f'Overwriting existing predictions: ', end=' ')
                do_simulation = True
            elif pred_object_states.shape[0] != self.object_pose_ts.shape[0]:
                print(f'Stored predictions do not match recorded data ' + \
                      f'length ({pred_object_states.shape[0]=} versus ' + \
                      f'{self.object_pose_ts.shape[0]=}) -- will simulate.')
                do_simulation = True

        if do_simulation:
            print(f'Simulating {model_to_test} model...')
            self._build_control_drake_plant()
            self._build_sim_drake_diagram()
        else:
            print(f'Found existing predictions; just generating video ' + \
                  f'for {model_to_test}...')

        if self.debug:
            print(f'Comparison URDF: {self.comparison_urdf_path}')
            print(f'Learned URDF: {self.learned_urdf_path}')

        # Generate a visualization Drake diagram whether simulating or not.
        self._build_vis_drake_diagram(model_to_test=model_to_test)

        # Prepare to run a simulation.
        if do_simulation:
            t0 = self.object_pose_ts[0]
            sim_plant_context = self.sim_plant.CreateDefaultContext()
            self.sim_plant.SetPositions(sim_plant_context, np.vstack((
                self.gt_joint_angle_traj.value(t0),
                self.recorded_object_quat_pos_traj.value(t0))))
            self.inv_dyn_controller.StartFrom(
                joint_angles=self.gt_joint_angle_traj.value(t0),
                joint_velocities=self.gt_joint_vel_traj.value(t0),
                joint_torques=self.gt_joint_torque_traj.value(t0),
                cartesian_stiffness=self.cartesian_stiffness,
                cartesian_damping=self.cartesian_damping
            )

            # Prepare to store results from the simulation.
            N = len(self.object_pose_ts)
            pred_object_states = np.zeros((N, 13))
            pred_franka_states = np.zeros((N, 14))
            pred_ee_forces = np.zeros((N, 3))

        for i, t in enumerate(self.object_pose_ts):
            if do_simulation:
                # Run the simulation.
                self.simulator.AdvanceTo(t)

                # Get the current states from the simulator.
                sim_context = self.simulator.get_context()
                sim_plant_context = self.sim_plant.GetMyContextFromRoot(
                    sim_context)
                sim_positions = self.sim_plant.GetPositions(sim_plant_context)
                sim_velocities = self.sim_plant.GetVelocities(sim_plant_context)

                franka_joint_angles = sim_positions[:7]
                franka_joint_velocity = sim_velocities[:7]
                object_quat_pos = sim_positions[7:]
                object_velocity = sim_velocities[7:]

                # Store the state results.
                pred_object_states[i] = np.hstack((
                    object_quat_pos, object_velocity
                ))
                pred_franka_states[i] = np.hstack((
                    franka_joint_angles, franka_joint_velocity
                ))

                # Store the contact results.
                contact_results = \
                    self.sim_plant.get_contact_results_output_port(
                    ).Eval(sim_plant_context)
                pred_ee_forces[i] = self._get_robot_contact_force(\
                    contact_results, model_to_test)

            else:
                object_quat_pos = pred_object_states[i, :7]
                franka_joint_angles = pred_franka_states[i, :7]

            # Update the visualization.
            vis_states = np.vstack((
                franka_joint_angles.reshape(-1, 1),
                self.commanded_quat_pos_traj.value(t),
                self.recorded_object_quat_pos_traj.value(t),
                object_quat_pos.reshape(-1, 1)
            ))
            vis_context = self.vis_simulator.get_context()
            vis_plant_context = self.vis_plant.GetMyMutableContextFromRoot(
                vis_context)
            self.vis_plant.SetPositions(vis_plant_context, vis_states)
            self.vis_diagram.ForcedPublish(vis_context)

            vw_context_front = self.video_writer_front.GetMyContextFromRoot(
                vis_context)
            self.video_writer_front._publish(vw_context_front)

            vw_context_camera = self.video_writer_camera.GetMyContextFromRoot(
                vis_context)
            self.video_writer_camera._publish(vw_context_camera)

        self.video_writer_front.Save()
        self.video_writer_camera.Save()

        if do_simulation:
            np.savetxt(pred_object_states_filename, pred_object_states)
            np.savetxt(pred_franka_states_filename, pred_franka_states)
            np.savetxt(pred_forces_on_object_by_ee_filename, pred_ee_forces)

    def _get_robot_contact_force(self, contact_results, model_to_test):
        """Given the simulated contact results, determine the world forces
        between the robot and the end effector."""
        body_model = self.sim_plant.GetBodyByName(
            f'{model_to_test}_body').index()
        body_ee = self.sim_plant.GetBodyByName(
            'end_effector_tip').index()

        contact_force = np.zeros(3)

        n_point_contacts = contact_results.num_point_pair_contacts()
        for i in range(n_point_contacts):
            point_pair_contact_info = contact_results.point_pair_contact_info(i)
            body_a = point_pair_contact_info.bodyA_index()
            body_b = point_pair_contact_info.bodyB_index()

            if (body_a == body_model and body_b == body_ee) or \
               (body_a == body_ee and body_b == body_model):
                scale = 1 if body_b == body_model else -1
                return scale*point_pair_contact_info.contact_force().reshape(3)

        # There should be no hydroelastic contacts expected, but check just in
        # case.
        n_hydroelastic_contacts = contact_results.num_hydroelastic_contacts()
        for i in range(n_hydroelastic_contacts):
            hydroelastic_contact_info = \
                contact_results.hydroelastic_contact_info(i)
            body_a = hydroelastic_contact_info.contact_surface().id_M()
            body_b = hydroelastic_contact_info.contact_surface().id_N()

            if (body_a == body_model and body_b == body_ee) or \
               (body_a == body_ee and body_b == body_model):
                # Note:  the scale convention is flipped between point contact
                # and hydroelastic contact.
                scale = 1 if body_a == body_model else -1
                return hydroelastic_contact_info.F_Ac_W().translational(
                    ).reshape(3)

        return contact_force

    def _export_test_data(self):
        self.save_dir = file_utils.robot_dynamics_subdir(
            self.vision_asset, overwrite=False)

        object = '_'.join(self.vision_asset.split('_')[:-1])
        toss_num = int(self.vision_asset.split('_')[-1])

        self.object = object
        self.toss_num = toss_num

        did_extract = False
        for file in FILES_TO_EXPORT:
            if not op.exists(op.join(self.save_dir, file)):
                self._extract_data_from_rosbag()
                did_extract = True
                break
        if not did_extract:
            print(f'All files already extracted for {object} toss {toss_num}.')

    def _extract_data_from_rosbag(self):
        print(f'Extracting data from ROS bag...')
        start = file_utils.load_toss_time_from_yaml(
            self.object, self.toss_num, 'start_time', as_ros_time=True)
        end = file_utils.load_toss_time_from_yaml(
            self.object, self.toss_num, 'end_time', as_ros_time=True)
        bag_number = file_utils.load_rosbag_number_from_yaml(
            self.object, self.toss_num)
        bag_file = file_utils.get_robot_bag_filename(bag_number)

        rosbag_processor.extract_franka_states(
            start_time=start, end_time=end, bag_file=bag_file,
            franka_states_output_dir=self.save_dir
        )
        rosbag_processor.extract_pose_commands(
            start_time=start, end_time=end, bag_file=bag_file,
            ee_pose_command_output_dir=self.save_dir
        )

    def _load_data(self):
        """Loads the following into class variables:
            - command_ts (N,)
            - des_pos (N, 3)
            - des_quat_wxyz (N, 4)
            - des_quats (N,) list of Quaternions
            - franka_joint_ts (M,)
            - joint_angles (M, 7)
            - joint_velocities (M, 7)
            - joint_torques (M, 7)
            - cartesian_stiffness (6,)
            - cartesian_damping (6,)
            - object_poses (L, 7)
            - object_pose_ts (L,)
        """
        save_dir = self.save_dir

        command_ts = np.loadtxt(op.join(save_dir, 'pose_command_times.txt'))
        des_pos = np.loadtxt(op.join(save_dir, 'des_pos.txt'))
        des_quat_wxyz = np.loadtxt(op.join(save_dir, 'des_quat_wxyz.txt'))

        des_quats = []
        for quat_wxyz in des_quat_wxyz:
            des_quats.append(Quaternion(quat_wxyz))

        franka_joint_ts = np.loadtxt(op.join(save_dir, 'joint_times.txt'))
        joint_angles = np.loadtxt(op.join(save_dir, 'joint_angles.txt'))
        joint_velocities = np.loadtxt(op.join(save_dir, 'joint_velocities.txt'))
        joint_torques = np.loadtxt(op.join(save_dir, 'tau_J_ds.txt'))

        cartesian_stiffness = np.loadtxt(op.join(
            save_dir, 'cartesian_stiffness.txt'))
        cartesian_damping = np.loadtxt(op.join(
            save_dir, 'cartesian_damping.txt'))

        # Zero out the trajectories.
        init_t = min(command_ts[0], franka_joint_ts[0])
        command_ts -= init_t
        franka_joint_ts -= init_t

        ### Load object-related data ###
        # Offset the commanded end effector poses from the table height.  The
        # Franka will also get offset by the same amount in the simulations.
        des_pos[:, 2] += WORLD_TO_FRANKA_PLL_OFFSET

        # Load the object poses.
        pll_traj_dir = file_utils.contactnets_input_dir_bundlesdf(
            self.vision_asset, self.bsdf_iteration, self.bundlesdf_id,
            full=False, create=False)
        toss_data_dict = torch.load(
            op.join(pll_traj_dir, f'{self.toss_num}.pt'), weights_only=False)
        object_states = np.array(toss_data_dict['object_state'])
        object_poses = object_states[:, :7]  # quat_wxyz, pos
        cnets_data_gen_dir = file_utils.cnets_data_gen_dataset_dir(
            self.vision_asset, check_exists=True)
        object_pose_ts = np.loadtxt(op.join(
            cnets_data_gen_dir, 'bundlesdf_timestamps.txt')) - init_t
        assert np.abs(object_pose_ts[0]) < 1e-1, f'Files may not be ' + \
            f'compatible, since {object_pose_ts[0]=}'

        self.command_ts = command_ts
        self.des_pos = des_pos
        self.des_quat_wxyz = des_quat_wxyz
        self.des_quats = des_quats
        self.franka_joint_ts = franka_joint_ts
        self.joint_angles = joint_angles
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.cartesian_stiffness = cartesian_stiffness
        self.cartesian_damping = cartesian_damping
        self.object_poses = object_poses
        self.object_pose_ts = object_pose_ts

    def _create_drake_trajectories(self):
        """Creates the following into class variables of Drake trajectories:
            - commanded_quat_pos_traj
            - recorded_object_quat_pos_traj
            - gt_joint_angle_traj
            - gt_joint_vel_traj
            - gt_joint_torque_traj
        """
        position_trajectory = \
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                breaks=self.command_ts,
                samples=self.des_pos.T,
                sample_dot_at_start=np.zeros(3),
                sample_dot_at_end=np.zeros(3)
            )
        orientation_trajectory = PiecewiseQuaternionSlerp(
            breaks=self.command_ts,
            quaternions=self.des_quats
        )
        # Sadly the more fool-proof PiecewisePose outputs 4x4 homogeneous
        # transform matrices, but TrajectorySource needs a column vector.  Use
        # StackedTrajectory instead, and use caution when interpreting the
        # output ordering.
        commanded_quat_pos_traj = StackedTrajectory()
        commanded_quat_pos_traj.Append(orientation_trajectory)
        commanded_quat_pos_traj.Append(position_trajectory)
        self.commanded_quat_pos_traj = commanded_quat_pos_traj

        # Build the object trajectory.
        recorded_object_quat_traj = PiecewiseQuaternionSlerp(
            breaks=self.object_pose_ts,
            quaternions=[Quaternion(q) for q in self.object_poses[:, :4]]
        )
        recorded_object_pos_traj = \
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                breaks=self.object_pose_ts,
                samples=self.object_poses[:, 4:].T,
                sample_dot_at_start=np.zeros(3),
                sample_dot_at_end=np.zeros(3)
            )
        recorded_object_quat_pos_traj = StackedTrajectory()
        recorded_object_quat_pos_traj.Append(recorded_object_quat_traj)
        recorded_object_quat_pos_traj.Append(recorded_object_pos_traj)
        self.recorded_object_quat_pos_traj = recorded_object_quat_pos_traj

        # TODO would it be better to combine angles and velocities into one
        # trajectory, since technically the derivatives affect each other?
        # Maybe torques too?
        self.gt_joint_angle_traj = \
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                breaks=self.franka_joint_ts,
                samples=self.joint_angles.T,
                sample_dot_at_start=np.zeros(7),
                sample_dot_at_end=np.zeros(7)
            )
        self.gt_joint_vel_traj = \
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                breaks=self.franka_joint_ts,
                samples=self.joint_velocities.T,
                sample_dot_at_start=np.zeros(7),
                sample_dot_at_end=np.zeros(7)
            )
        self.gt_joint_torque_traj = \
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                breaks=self.franka_joint_ts,
                samples=self.joint_torques.T,
                sample_dot_at_start=np.zeros(7),
                sample_dot_at_end=np.zeros(7)
            )

    def _build_control_drake_plant(self):
        """Defines control_plant class variable."""
        # Add the controller, which needs a separate plant for control with just
        # the robot.
        control_plant = MultibodyPlant(time_step=CONTROLLER_TIME_STEP)
        control_parser = Parser(control_plant)
        control_robot = control_parser.AddModels(
            file_utils.franka_filepath(with_collision_geometry=True))[0]
        control_plant.WeldFrames(control_plant.world_frame(),
                                control_plant.GetFrameByName('panda_link0'),
                                RigidTransform(p=np.array([
                                    0, 0, WORLD_TO_FRANKA_PLL_OFFSET])))
        control_plant.Finalize()
        control_plant.set_name('control_plant')
        control_plant.SetDefaultPositions(
            self.gt_joint_angle_traj.value(
                self.commanded_quat_pos_traj.start_time()))

        self.control_plant = control_plant

    def _build_sim_drake_diagram(self):
        ### Simulation Drake diagram ###
        builder = DiagramBuilder()

        # Add the robot to a simulation.
        sim_mbp_config = MultibodyPlantConfig(time_step=SIM_TIME_STEP)
        sim_plant, scene_graph = AddMultibodyPlant(sim_mbp_config, builder)
        sim_parser = Parser(sim_plant)
        sim_robot = sim_parser.AddModels(
            file_utils.franka_filepath(with_collision_geometry=True))[0]
        sim_plant.WeldFrames(sim_plant.world_frame(),
                            sim_plant.GetFrameByName('panda_link0'),
                            RigidTransform(p=np.array([
                                0, 0, WORLD_TO_FRANKA_PLL_OFFSET])))

        # Add the table (as a half space) to the simulation, located at z=0.
        # TODO:  meshcat doesn't display halfspace geometries, so may want to
        # use a box instead.
        friction = CoulombFriction(1.0, 1.0)
        sim_plant.RegisterCollisionGeometry(
            sim_plant.world_body(), RigidTransform(), HalfSpace(),
            'table', friction)
        sim_plant.RegisterVisualGeometry(
            sim_plant.world_body(), RigidTransform(), HalfSpace(),
            'table', np.array([0.5, 0.5, 0.5, 1]))

        # Add the object to the simulation.
        sim_object = sim_parser.AddModels(self.learned_urdf_path)[0]

        sim_plant.Finalize()
        sim_plant.set_name('sim_plant')
        sim_plant.SetDefaultPositions(np.vstack((
            self.gt_joint_angle_traj.value(
                self.commanded_quat_pos_traj.start_time()),
            self.recorded_object_quat_pos_traj.value(
                self.commanded_quat_pos_traj.start_time())
        )))
        # if self.debug:  visualize_drake_systems(plant=sim_plant)

        # Add a trajectory source for the desired end effector pose.
        traj_source = builder.AddSystem(TrajectorySource(
            self.commanded_quat_pos_traj))

        # Add the controller, which needs the control plant.
        control_params = file_utils.load_franka_control_params()
        inv_dyn_controller = builder.AddSystem(
            InverseDynamicsController(self.control_plant, control_params))
        inv_dyn_controller.set_name('inv_dyn_controller')

        # Wire the diagram.
        builder.Connect(traj_source.get_output_port(),
                        inv_dyn_controller.get_ee_pose_command_input_port())
        builder.Connect(sim_plant.get_state_output_port(sim_robot),
                        inv_dyn_controller.get_robot_state_input_port())
        builder.Connect(inv_dyn_controller.get_joint_torque_output_port(),
                        sim_plant.get_actuation_input_port())

        diagram = builder.Build()
        # if self.debug:  visualize_drake_systems(diagram=diagram); breakpoint()
        simulator = Simulator(diagram)
        simulator.Initialize()
        simulator.set_target_realtime_rate(1)

        self.sim_plant = sim_plant
        self.inv_dyn_controller = inv_dyn_controller
        self.simulator = simulator

    def _build_vis_drake_diagram(self, model_to_test: str):
        ### Visualization Drake diagram ###
        vis_builder = DiagramBuilder()

        # Add the robot and a floating end effector to the plant.
        vis_mbp_config = MultibodyPlantConfig(time_step=vis_TIME_STEP)
        vis_plant, vis_scene_graph = AddMultibodyPlant(
            vis_mbp_config, vis_builder)
        vis_parser = Parser(vis_plant)
        vis_robot = vis_parser.AddModels(
            file_utils.franka_filepath(with_collision_geometry=True))
        vis_plant.WeldFrames(vis_plant.world_frame(),
                            vis_plant.GetFrameByName('panda_link0'),
                            RigidTransform(p=np.array([
                                0, 0, WORLD_TO_FRANKA_PLL_OFFSET])))
        vis_commanded_ee = vis_parser.AddModels(file_utils.ee_urdf_filepath())
        vis_comp_object = vis_parser.AddModels(self.comparison_urdf_path)[0]
        vis_object = vis_parser.AddModels(self.learned_urdf_path)[0]
        vis_plant.RegisterVisualGeometry(
            vis_plant.world_body(), RigidTransform(), HalfSpace(),
            'table', np.array([0.5, 0.5, 0.5, 0.5]))
        vis_plant.Finalize()
        vis_plant.set_name('vis_plant')
        vis_plant.SetDefaultPositions(np.vstack((
            self.gt_joint_angle_traj.value(
                self.commanded_quat_pos_traj.start_time()),
            self.commanded_quat_pos_traj.value(
                self.commanded_quat_pos_traj.start_time()),
            self.recorded_object_quat_pos_traj.value(
                self.commanded_quat_pos_traj.start_time()),
            self.recorded_object_quat_pos_traj.value(
                self.commanded_quat_pos_traj.start_time())
        )))
        AddFrameTriadIllustration(
            scene_graph=vis_scene_graph,
            body=vis_plant.GetBodyByName('floating_end_effector_tip'),
            length=0.05, radius=0.008, opacity=0.5
        )
        AddFrameTriadIllustration(
            scene_graph=vis_scene_graph,
            body=vis_plant.GetBodyByName('end_effector_tip'),
            length=0.1, radius=0.005, opacity=0.8
        )
        AddFrameTriadIllustration(
            scene_graph=vis_scene_graph,
            body=vis_plant.GetBodyByName(f'comparison_body'),
            length=0.15, radius=0.01, opacity=0.5
        )
        AddFrameTriadIllustration(
            scene_graph=vis_scene_graph,
            body=vis_plant.GetBodyByName(f'{model_to_test}_body'),
            length=0.2, radius=0.008, opacity=0.8
        )

        # Add a meshcat visualizer.
        if self.open_meshcat:
            if hasattr(self, 'meshcat'):
                self.meshcat.Delete()
            else:
                self.meshcat = StartMeshcat()
            MeshcatVisualizer.AddToBuilder(
                vis_builder, vis_scene_graph, self.meshcat)

        video_writer_front = VideoWriter.AddToBuilder(
            filename=op.join(self.save_dir,
                             f'{self.vision_asset}_{model_to_test}_front.mp4'),
            builder=vis_builder,
            sensor_pose=SENSOR_POSE_FRONT_VIEW,
            fps=FPS,
            backend="cv2",
            width=VIDEO_PIXELS[1],
            height=VIDEO_PIXELS[0],
            fov_y=CAM_FOV
        )
        video_writer_camera = VideoWriter.AddToBuilder(
            filename=op.join(self.save_dir,
                             f'{self.vision_asset}_{model_to_test}_cam.mp4'),
            builder=vis_builder,
            sensor_pose=SENSOR_POSE_CAMERA_VIEW,
            fps=FPS,
            backend="cv2",
            width=VIDEO_PIXELS[1],
            height=VIDEO_PIXELS[0],
            fov_y=CAM_FOV_REALSENSE
        )

        # Build the diagram.
        vis_diagram = vis_builder.Build()
        #if self.debug:  visualize_drake_systems(diagram=vis_diagram)
        vis_simulator = Simulator(vis_diagram)
        vis_context = vis_simulator.get_context()
        vis_diagram.ForcedPublish(vis_context)

        self.vis_plant = vis_plant
        self.video_writer_front = video_writer_front
        self.video_writer_camera = video_writer_camera
        self.vis_simulator = vis_simulator
        self.vis_diagram = vis_diagram


class DynamicsPredictionQuantifier():
    """Creates an internal dictionary with the following structure:

        tracked:
            vysics:
                position_error: (N,)
                rotation_error: (N,)
                time_before_bad_pos: float
                time_before_bad_rot: float
            bsdf: ...
            pll: ...
            gt: ...
        gt_sim: ...
    """
    def __init__(self, vision_asset: str, interactive: bool = False):
        self.vision_asset = vision_asset
        self.interactive = interactive
        self.pred_dir = file_utils.robot_dynamics_subdir(vision_asset)
        self.save_dir = file_utils.robot_dynamics_subdir(
            vision_asset, overwrite=False)

        if interactive:
            plt.ion()

        self._load_data()
        self._load_recorded_object_poses_times_contacts()
        self._compute_time_series_errors()

    def _load_data(self):
        self.pred_files = {}
        for file in FILES_TO_GENERATE:
            # Skip loading the Franka states, only need object states.
            if 'object' in file or 'forces' in file:
                self.pred_files[file] = np.loadtxt(op.join(self.pred_dir, file))

    def _load_recorded_object_poses_times_contacts(self):
        pll_id, bsdf_id, nerf_bsdf_id = ids_from_vision_asset(self.vision_asset)
        toss_num = int(self.vision_asset.split('_')[-1])

        # Load the object poses from the PLL input data.
        pll_traj_dir = file_utils.contactnets_input_dir_bundlesdf(
            self.vision_asset, BSDF_ITERATION, bsdf_id,
            full=False, create=False)
        toss_data_dict = torch.load(
            op.join(pll_traj_dir, f'{toss_num}.pt'), weights_only=False)
        object_states = np.array(toss_data_dict['object_state'])
        self.object_poses = object_states[:, :7]  # quat_wxyz, pos

        # Load the timestamps from the generated dataset.
        cnets_data_gen_dir = file_utils.cnets_data_gen_dataset_dir(
            self.vision_asset, check_exists=True)
        self.times = np.loadtxt(op.join(
            cnets_data_gen_dir, 'bundlesdf_timestamps.txt'))
        self.times -= self.times[0]

        # Load the contact activations from the manually annotated yaml.
        contact_activation_dict = \
            file_utils.load_contact_activation_dict_from_yaml(self.vision_asset)
        starts = contact_activation_dict['start']
        starts = [starts] if type(starts) != list else starts
        ends = contact_activation_dict['end']
        ends = [ends] if type(ends) != list else ends
        contact_activations = np.zeros_like(self.times, dtype=bool)
        for start, end in zip(starts, ends):
            contact_activations[start:end+1] = np.ones(end+1-start)
        self.contact_activations = contact_activations

    def _compute_time_series_errors(self):
        """Create the time-series dictionary of errors, first keyed by the
        comparison trajectory (tracked or GT simulated), then by the model
        being tested (Vysics, BSDF, PLL, GT), and lastly by the error type
        (position error or rotation error).

        In addition to time-series errors, this also computes the time before
        position and rotation errors cross the error tolerances."""
        # Quantify w.r.t. tracked poses or GT simulated poses.
        compare_against = [
            self.object_poses, self.pred_files['pred_object_states_gt.txt']]
        contact_force_dict = {}
        errors = {'tracked': {}, 'gt_sim': {}}

        for compare_traj, error_dict in zip(compare_against, errors.values()):
            for key, pred_traj in self.pred_files.items():
                if 'franka' in key:
                    continue

                model = key.replace('.txt', '').split('_')[-1]
                if model not in error_dict.keys():
                    error_dict[model] = {}

                if 'forces' in key:
                    if model in contact_force_dict.keys():
                        continue

                    force_activation = np.linalg.norm(pred_traj, axis=1) > 0
                    contact_activation_iou = binary_iou(
                        force_activation, self.contact_activations)
                    contact_force_dict[model] = {
                        'pred_contact_activation': force_activation,
                        'contact_activation_iou': contact_activation_iou
                    }
                    continue

                # Compute the error.
                pos_error = TrajectoryMetrics.position_error(
                    compare_traj, pred_traj)
                rotation_error = TrajectoryMetrics.rotation_error(
                    compare_traj, pred_traj)

                # Compute the time before each error becomes too large.
                try:
                    pos_idx = np.where(pos_error > POSITION_TOLERANCE)[0][0]
                    pos_error_time = self.times[pos_idx]
                except IndexError:
                    pos_error_time = self.times[-1] + 1e-3

                try:
                    rad_idx = np.where(rotation_error>ROTATION_TOLERANCE)[0][0]
                    rad_error_time = self.times[rad_idx]
                except IndexError:
                    rad_error_time = self.times[-1] + 1e-3

                # Store the results.
                error_dict[model]['position_error'] = pos_error
                error_dict[model]['rotation_error'] = rotation_error
                error_dict[model]['time_before_bad_pos'] = pos_error_time
                error_dict[model]['time_before_bad_rot'] = rad_error_time

        self.errors = errors
        self.contact_force_dict = contact_force_dict

    def plot_errors(self, truncate: bool = False):
        for compare_against in ['tracked', 'gt_sim']:
            fig, axs = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
            title = f'{self.vision_asset} against {compare_against}'
            title += f', truncated ({POSITION_TOLERANCE:.2f}m, ' + \
                f'{ROTATION_TOLERANCE:.2f}rad)' if truncate else ''
            fig.suptitle(title)
            axs[0].set_xlabel('Time (s)')
            axs[1].set_xlabel('Time (s)')
            axs[0].set_ylabel('Error [m]')
            axs[1].set_ylabel('Error [rad]')
            axs[0].set_title('Position Error')
            axs[1].set_title('Rotation Error')

            models = ['vysics', 'bsdf', 'pll', 'gt']
            colors = [VYSICS_MESH_HEX, BSDF_MESH_HEX, PLL_MESH_HEX, GT_MESH_HEX]
            for model, color in zip(models, colors):
                pos_times = self.times if not truncate else \
                    self.times[
                        self.times < self.errors[compare_against][model][
                            'time_before_bad_pos']]
                pos_error = self.errors[compare_against][model][
                    'position_error'][:len(pos_times)]

                rot_times = self.times if not truncate else \
                    self.times[
                        self.times < self.errors[compare_against][model][
                            'time_before_bad_rot']]
                rot_error = self.errors[compare_against][model][
                    'rotation_error'][:len(rot_times)]

                axs[0].plot(pos_times, pos_error, color=color, label=model)
                axs[1].plot(rot_times, rot_error, color=color, label=model)

            plt.legend()
            plt.savefig(
                op.join(self.save_dir, f'{compare_against}_{model}_errors.png'))
            if not self.interactive:
                plt.close()

        # if self.interactive:
        #     breakpoint()

    def plot_time_to_failure(self):
        for compare_against in ['tracked', 'gt_sim']:
            fig, axs = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
            fig.suptitle(f'{self.vision_asset} Time to Dynamics Prediction ' + \
                         f'Divergence, compared against {compare_against}')
            axs[0].set_ylabel('Time [s]')
            axs[0].set_title(f'Time to {POSITION_TOLERANCE:.2f}m Error')
            axs[1].set_title(f'Time to {ROTATION_TOLERANCE:.2f}rad Error')

            models = ['vysics', 'bsdf', 'pll', 'gt']
            colors = [VYSICS_MESH_HEX, BSDF_MESH_HEX, PLL_MESH_HEX, GT_MESH_HEX]

            bar_width = 0.2
            offsets = np.linspace(-bar_width * (len(models) - 1) / 2,
                                  bar_width * (len(models) - 1) / 2,
                                  len(models))

            for model_i, (model, color) in enumerate(zip(models, colors)):
                t = self.errors[compare_against][model]['time_before_bad_pos']
                axs[0].bar(offsets[model_i], t, width=bar_width, color=color,
                           label=model)

                t = self.errors[compare_against][model]['time_before_bad_rot']
                axs[1].bar(offsets[model_i], t, width=bar_width, color=color,
                           label=model)

            plt.legend()
            plt.savefig(op.join(
                self.save_dir, f'{compare_against}_{model}_time_divergence.png'
            ))
            if not self.interactive:
                plt.close()

        if self.interactive:
            breakpoint()

    def plot_contact_activations(self):
        fig, axs = plt.subplots(1, 1, figsize=(6, 3))
        fig.suptitle(f'{self.vision_asset} Contact Activations')
        axs.set_xlabel('Time (s)')

        models = ['vysics', 'bsdf', 'pll', 'gt', 'manual']
        colors = [VYSICS_MESH_HEX, BSDF_MESH_HEX, PLL_MESH_HEX, GT_MESH_HEX,
                  MANUAL_ANNOTATION_HEX]

        for model_i, (model, color) in enumerate(zip(models, colors)):
            if model == 'manual':
                cs = self.contact_activations
            else:
                cs = self.contact_force_dict[model]['pred_contact_activation']
            true_indices = np.where(cs)[0]
            true_times = self.times[true_indices]
            try:
                widths = self.times[true_indices + 1] - true_times
            except IndexError:
                widths = self.times[true_indices[:-1] + 1] - true_times[:-1]
                widths = np.append(widths, self.times[-1] - true_times[-1])
            axs.barh(y=model_i, width=widths, left=true_times,
                     color=color, align='center', label=model)

        axs.set_xlim([0, self.times[-1]])
        axs.set_yticks([i for i in range(len(models))], models)
        plt.tight_layout()

        plt.savefig(op.join(self.save_dir, f'contact_activations.png'))
        if not self.interactive:
            plt.close()


class ConglomeratedDynamicsMetrics():
    def __init__(self, list_of_dpqs: List[DynamicsPredictionQuantifier],
                 interactive: bool = False):
        self.dpqs = list_of_dpqs
        self.interactive = interactive
        self.save_dir = file_utils.robot_dynamics_dir()

        self._get_ordered_dpqs()

    def _get_ordered_dpqs(self):
        """Goal is to define self.pos_ordered_dpqs as a dictionary keyed by
        ['tracked', 'gt_sim'], each key pointing to a list of dpqs, ordered by
        the GT geometry's time to position error divergence, and similarly for
        self.rot_ordered_dpqs."""
        dpq_pos_rot = {}
        for compare_against in ['tracked', 'gt_sim']:
            dpq_pos_rot[compare_against] = [
                self.dpqs,
                [dpq.errors[compare_against]['gt']['time_before_bad_pos'] \
                    for dpq in self.dpqs],
                [dpq.errors[compare_against]['gt']['time_before_bad_rot'] \
                    for dpq in self.dpqs],
            ]

        self.pos_ordered_dpqs = {}
        self.rot_ordered_dpqs = {}
        for compare_against in ['tracked', 'gt_sim']:
            self.pos_ordered_dpqs[compare_against] = [
                dpq for _, dpq in sorted(zip(
                    dpq_pos_rot[compare_against][1], self.dpqs))]
            self.rot_ordered_dpqs[compare_against] = [
                dpq for _, dpq in sorted(zip(
                    dpq_pos_rot[compare_against][2], self.dpqs))]

    def plot(self):
        if self.interactive:
            plt.ion()

        self._plot_traces(truncate=False)
        self._plot_traces(truncate=True)
        self._plot_time_to_failure()
        self._plot_time_to_failure_cdf(relative=False)
        self._plot_time_to_failure_cdf(relative=True)

        if self.interactive:
            breakpoint()

    def _plot_traces(self, truncate: bool = False):
        n_lines = len(self.dpqs)
        for compare_against in ['tracked', 'gt_sim']:
            fig, axs = plt.subplots(2, 4, figsize=(12, 9), sharex=True,
                                    sharey='row')
            title = f'Conglomerated Dynamics Predictions against ' + \
                    f'{compare_against}'
            title += f', truncated ({POSITION_TOLERANCE:.2f}m, ' + \
                f'{ROTATION_TOLERANCE:.2f}rad)' if truncate else ''
            fig.suptitle(title)
            axs[1, 0].set_xlabel('Time (s)')
            axs[1, 1].set_xlabel('Time (s)')
            axs[1, 2].set_xlabel('Time (s)')
            axs[1, 3].set_xlabel('Time (s)')
            axs[0, 0].set_ylabel('Position Error [m]')
            axs[1, 0].set_ylabel('Orientation Error [rad]')
            axs[0, 0].set_title('Vysics')
            axs[0, 1].set_title('BundleSDF')
            axs[0, 2].set_title('PLL')
            axs[0, 3].set_title('Ground Truth')

            models = ['vysics', 'bsdf', 'pll', 'gt']
            colors = [VYSICS_MESH_HEX, BSDF_MESH_HEX, PLL_MESH_HEX, GT_MESH_HEX]
            lines = []
            for col_i, (model, color) in enumerate(zip(models, colors)):
                for dpq_i, dpq in enumerate(self.dpqs):
                    pos_times = dpq.times if not truncate else \
                        dpq.times[
                            dpq.times < dpq.errors[compare_against][model][
                                'time_before_bad_pos']]
                    pos_error = dpq.errors[compare_against][model][
                        'position_error'][:len(pos_times)]

                    rot_times = dpq.times if not truncate else \
                        dpq.times[
                            dpq.times < dpq.errors[compare_against][model][
                                'time_before_bad_rot']]
                    rot_error = dpq.errors[compare_against][model][
                        'rotation_error'][:len(rot_times)]

                    linewidth = 2  #SMALLEST_LINEWIDTH #+ \
                        # (BIGGEST_LINEWIDTH-SMALLEST_LINEWIDTH)*dpq_i /
                        # (n_lines-1)
                    line = axs[0, col_i].plot(pos_times, pos_error, color=color,
                        linewidth=linewidth, alpha=0.2,
                        label=dpq.vision_asset.replace('robotocc_', ''))
                    axs[1, col_i].plot(rot_times, rot_error, color=color,
                        linewidth=linewidth, alpha=0.2,
                        label=dpq.vision_asset.replace('robotocc_', ''))
                    if col_i == 0:
                        lines.append(line[0])

            fig.legend(handles=lines, loc='center right')
            plt.tight_layout(rect=[0, 0, 0.88, 1])
            plt.savefig(op.join(
                self.save_dir, f'{compare_against}_combined_errors.png'))
            if not self.interactive:
                plt.close()

    def _plot_time_to_failure_cdf(self, relative: bool = False):
        # Some settings on the plot generation.
        rc('legend', fontsize=20)
        plt.rc('axes', titlesize=24)    # fontsize of the axes title
        plt.rc('axes', labelsize=24)    # fontsize of the x and y labels

        for compare_against in ['tracked', 'gt_sim']:
            # Generate a plot of cumulative distribution functions.
            fig, axs = plt.subplots(1, 2, figsize=(13,10), sharey=True)
            if relative:
                axs[0].set_ylabel(f'Fraction of Trajectory')
                axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                fig.suptitle('Fraction of Trajectory Length before ' + \
                             'Prediction Divergence', fontsize=28,
                             fontname='serif')
            else:
                axs[0].set_ylabel(f'Time of Position Divergence ' + \
                                f'({POSITION_TOLERANCE:.2f}m) [s]')
                axs[1].set_ylabel(f'Time of Rotation Divergence ' + \
                                f'({ROTATION_TOLERANCE:.2f}rad) [s]')
            axs[0].set_xlabel('Fraction of Trials')
            axs[1].set_xlabel('Fraction of Trials')
            axs[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            axs[0].set_xlim([0, 1])
            axs[1].set_xlim([0, 1])
            axs[0].set_title(
                f'Position Divergence ({POSITION_TOLERANCE*100:.0f}cm)')
            deg_tolerance = ROTATION_TOLERANCE*180/np.pi
            axs[1].set_title(f'Orientation Divergence ({deg_tolerance:.0f}deg)')

            models = ['vysics', 'bsdf', 'pll', 'gt']
            labels = ['Vysics', 'BundleSDF', 'PLL', 'GT Geometry']
            colors = [VYSICS_MESH_HEX, BSDF_MESH_HEX, PLL_MESH_HEX, GT_MESH_HEX]
            linestyles = ['solid', 'solid', 'dashed', 'dashed']

            for model, color, linestyle, label in zip(
                models, colors, linestyles, labels):
                for metric_i, key in enumerate(
                    ['time_before_bad_pos', 'time_before_bad_rot']):
                    data = [dpq.errors[compare_against][model][key] \
                            for dpq in self.dpqs]
                    if relative:
                        comp_data = [dpq.times[-1] for dpq in self.dpqs]
                        data = np.array(data) / np.array(comp_data)
                    count, bins_count = np.histogram(data, bins=11)
                    pdf = count / sum(count)
                    cdf = np.cumsum(pdf)
                    bins_count[0] = 0
                    cdf = np.concatenate([[0], cdf])
                    axs[metric_i].plot(
                        cdf, bins_count, color=color, linestyle=linestyle,
                        linewidth=5, label=label)

            axs[0].grid()
            axs[1].grid()
            axs[0].set_ylim(bottom=0)
            axs[0].legend(prop=dict(weight='bold', family='serif'))

            # Beautify the plot.
            for ax in axs:
                ax.title.set_fontname('serif')
                ax.xaxis.label.set_fontname('serif')
                ax.yaxis.label.set_fontname('serif')

                for tick in ax.get_xticklabels():
                    tick.set_fontname('serif')
                for tick in ax.get_yticklabels():
                    tick.set_fontname('serif')

                ax.tick_params(axis='both', which='major', labelsize=16)

            fig.set_size_inches(13, 10)
            plt.subplots_adjust(bottom=0.15)

            filename = f'{compare_against}_combined_time_divergence_cdf'
            filename += '_relative' if relative else ''
            plt.savefig(op.join(self.save_dir, f'{filename}.png'))
            if not self.interactive:
                plt.close()

    def _plot_time_to_failure(self):
        n_bars = len(self.dpqs)
        for compare_against in ['tracked', 'gt_sim']:
            dpq_list_pos = self.pos_ordered_dpqs[compare_against]
            dpq_list_rot = self.rot_ordered_dpqs[compare_against]

            fig, axs = plt.subplots(
                2, 4, figsize=(12, 9), sharey='row')
            fig.suptitle(f'Conglomerated Time to Dynamics Prediction ' + \
                         f'Divergence, compared against {compare_against}')
            axs[0, 0].set_ylabel(
                f'Time to {POSITION_TOLERANCE:.2f}m Position Error [s]')
            axs[1, 0].set_ylabel(
                f'Time to {ROTATION_TOLERANCE:.2f}rad Rotation Error [s]')
            axs[0, 0].set_title('Vysics')
            axs[0, 1].set_title('BundleSDF')
            axs[0, 2].set_title('PLL')
            axs[0, 3].set_title('Ground Truth Geometry')

            models = ['vysics', 'bsdf', 'pll', 'gt']
            colors = [VYSICS_MESH_HEX, BSDF_MESH_HEX, PLL_MESH_HEX, GT_MESH_HEX]

            bar_width = 0.2
            offsets = np.linspace(-bar_width * (n_bars-1)/2,
                                  bar_width * (n_bars-1)/2,
                                  n_bars)

            for model_i, (model, color) in enumerate(zip(models, colors)):
                for dpq_i, dpq in enumerate(dpq_list_pos):
                    t = dpq.errors[compare_against][model][
                        'time_before_bad_pos']
                    axs[0, model_i].bar(
                        offsets[dpq_i], t, width=bar_width, color=color,
                        label=model)

                for dpq_i, dpq in enumerate(dpq_list_rot):
                    t = dpq.errors[compare_against][model][
                        'time_before_bad_rot']
                    axs[1, model_i].bar(
                        offsets[dpq_i], t, width=bar_width, color=color,
                        label=model)

            vision_assets = [dpq.vision_asset for dpq in self.dpqs]
            axs[1,0].set_xticks(offsets, vision_assets, rotation=45, ha='right')
            axs[1,1].set_xticks([], [])
            axs[1,2].set_xticks([], [])
            axs[1,3].set_xticks([], [])
            axs[0,0].set_xticks([], [])
            axs[0,1].set_xticks([], [])
            axs[0,2].set_xticks([], [])
            axs[0,3].set_xticks([], [])
            plt.tight_layout()
            plt.savefig(op.join(
                self.save_dir, f'{compare_against}_combined_time_' + \
                    f'divergence.png'
            ))
            if not self.interactive:
                plt.close()

    def export_statistics(self):
        """Export a statistics yaml file per model with structure:
            tagless_objects:
                robotocc_{object}:
                    trained_on_toss_{num}:
                        dynamics_prediction_metrics:
                            full_geometry:
                                pos_rollout_error_mean: float
                                rot_rollout_error_mean: float
                                time_before_bad_pos: float
                                time_before_bad_rot: float
                                contact_activation_iou: float
        """
        for model in MODELS_TO_TEST:
            inner_stats = {}
            for dpq in self.dpqs:
                vision_asset = dpq.vision_asset
                object = '_'.join(vision_asset.split('_')[:-1])
                if object not in inner_stats.keys():
                    inner_stats[object] = {}
                trained_on_key = \
                    f'trained_on_toss_{vision_asset.split("_")[-1]}'
                if trained_on_key not in inner_stats[object].keys():
                    inner_stats[object][trained_on_key] = {
                        'dynamics_prediction_metrics': {'full_geometry': {}}}
                one_exp_stats = inner_stats[object][trained_on_key][
                    'dynamics_prediction_metrics']['full_geometry']

                # Get the errors.
                pos_errs = dpq.errors['tracked'][model]['position_error']
                rot_errs = dpq.errors['tracked'][model]['rotation_error']
                t_pos_divergence = dpq.errors['tracked'][model][
                    'time_before_bad_pos']
                t_rot_divergence = dpq.errors['tracked'][model][
                    'time_before_bad_rot']
                contact_iou = dpq.contact_force_dict[model][
                    'contact_activation_iou']

                one_exp_stats['pos_rollout_error_mean'] = float(np.mean(
                    np.array(pos_errs)))
                one_exp_stats['rot_rollout_error_mean'] = float(np.mean(rot_errs))
                one_exp_stats['time_before_bad_pos'] = float(t_pos_divergence)
                one_exp_stats['time_before_bad_rot'] = float(t_rot_divergence)
                one_exp_stats['contact_activation_iou'] = float(contact_iou)

            # Wrap inside 'tagless_objects' key.
            stats = {'tagless_objects': inner_stats}

            # Write to a yaml with the model name.
            dir = file_utils.robot_dynamics_dir()
            filename = f'{model}.yaml'
            file_utils.save_results_to_yaml(stats, dir, filename=filename)
            print(f'Overwrote {dir}/{filename} from files.')


@click.group()
def cli():
    pass

@cli.command('gen')
@click.argument('vision-asset', type=str, required=True)
@click.argument('models-to-test', type=str, nargs=-1, required=True)
@click.option('--debug', is_flag=True,
              help='add extra debugging printouts')
@click.option('--meshcat', is_flag=True,
              help='show the simulated visualizations in meshcat')
@click.option('--plot', is_flag=True,
              help='plot results from existing trajectories')
@click.option('--overwrite', is_flag=True,
              help='overwrite any existing files')
def gen_command(vision_asset: str, models_to_test: Tuple[str], debug: bool,
                meshcat: bool, plot: bool, overwrite: bool):
    assert vision_asset in PLL_BSDF_NERF_IDS_FROM_VISION_ASSET.keys()
    for model_to_test in models_to_test:
        assert model_to_test in MODELS_TO_TEST

    pll_id, bsdf_id, nerf_bsdf_id = ids_from_vision_asset(vision_asset)
    robot_dynamics_predictor = RobotDynamicsPredictor(
        vision_asset, pll_id, bsdf_id, nerf_bsdf_id, BSDF_ITERATION,
        open_meshcat=meshcat, debug=debug)

    for model_to_test in models_to_test:
        print(f'Running simulation for {vision_asset} with {model_to_test}...')
        robot_dynamics_predictor.run_simulation(
            model_to_test, overwrite=overwrite)


@cli.command('plot')
@click.option('--statistics', is_flag=True,
              help='export statistics as a yaml file')
@click.option('--interactive', is_flag=True,
              help='show the plots interactively')
def plot_command(statistics: bool, interactive: bool):
    # Iterate over all the folders in the robot dynamics directory.
    robot_dir = file_utils.robot_dynamics_dir()
    dir_list = os.listdir(robot_dir)
    vision_assets = []
    for dir in dir_list:
        # Check that the item is a directory.
        if op.isdir(op.join(robot_dir, dir)):
            # Check that the item contains the predicted files.
            if np.all(np.array([
                op.exists(op.join(robot_dir, dir, f)) for f in FILES_TO_GENERATE
            ])):
                vision_assets.append(dir)
                print(f'=== Prepared to analyze {dir} ===')
            else:
                print(f'{dir} did not have all files.')
        else:
            print(f'{dir} is not a directory.')

    dpqs = []
    for vision_asset in vision_assets:
        dpq = DynamicsPredictionQuantifier(vision_asset, interactive)
        if not statistics:
            dpq.plot_errors(truncate=False)
            dpq.plot_errors(truncate=True)
            dpq.plot_time_to_failure()
            dpq.plot_contact_activations()
        dpqs.append(dpq)

    cdm = ConglomeratedDynamicsMetrics(dpqs, interactive)
    if statistics:
        cdm.export_statistics()
    else:
        cdm.plot()
    if interactive:
        breakpoint()


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter


