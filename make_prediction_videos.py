"""Script to generate overlay images from stored predictions."""

import click
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import open3d as o3d
import os
import os.path as op
import pdb
from tempfile import TemporaryDirectory
from tqdm import tqdm

import evaluate, eval_utils, file_utils, math_utils, rosbag_processor
from eval_utils import PredictionOverlayGenerator

NUMBERS = ['1', '1-2', '1-3', '1-4', '1-5']
OBJECTS = ['bakingbox', 'cardboard', 'crushedcan', 'gallon', 'greencan',
           'oatly', 'pinkcan', 'stapler', 'styrofoam', 'cube', 'bottle', 'half',
           'milk', 'egg', 'napkin']

PLL_IDS = ['00', '04', '07']  # Don't know how to do overlays for PLL TagSLAM 09
PLL_IDS = [f'pll_id_{pll_id}' for pll_id in PLL_IDS]
CYCLE_ITERATIONS_BY_PLL_ID = {'pll_id_00': 1, 'pll_id_04': 1,
                              'pll_id_07': 1, 'pll_id_09': 0}

BSDF_IDS = ['02', '03']
BSDF_IDS = [f'bundlesdf_id_{pll_id}' for pll_id in BSDF_IDS]
CYCLE_ITERATIONS_BY_BSDF_ID = {'bundlesdf_id_02': 2, 'bundlesdf_id_03': 2}

# Size (3_xyz, 2, 3)
LINEWIDTH = 1
AXIS_SCALING = 0.1
TRACKING_TO_PRED_AXIS_SIZE_RATIO = 0.5
CAMERA_AXES_IN_CAMERA_FRAME = AXIS_SCALING * np.array([[[0, 0, 0], [1, 0, 0]],
                                                       [[0, 0, 0], [0, 1, 0]],
                                                       [[0, 0, 0], [0, 0, 1]]])
FRAMES_TO_SKIP = 1
OPACITY_OVER_TIME = [255]*10 + \
    [int((i+1) * 1.0 / 40 * 255) for i in reversed(range(40))] + [0]*200
COM_TRAJ_COLORS_OVER_TIME = [f'#FF80D5{opacity:02x}' for opacity in OPACITY_OVER_TIME]

UPPER_DISTANCE = 0.013  # Reference: Vysics/BSDF CD averages are 0.0126/0.0136
LOWER_DISTANCE = 0.004

COLOR_MAP = 'rainbow'  # Other good ones: YlGnBu_r, cool, bwr, Spectral_r
N_TOTAL_SPIN_UNITS = 2080  # 2040 maybe too low, 2160 too high, 1920 too low


def get_camera_intrinsics(vision_asset: str):
    object = vision_asset.split('_')[0]
    start_toss = int(vision_asset.split('_')[1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    
    rosbag_number = file_utils.load_rosbag_number_from_yaml(
        object, start_toss, second_toss_number=end_toss)
    raw_bag_file = file_utils.get_depth_bag_filename(rosbag_number)

    # Gets the P matrix.
    intrinsics = rosbag_processor.get_all_camera_intrinsics(raw_bag_file)

    return intrinsics['realsense'].reshape(3, 4)

def convert_world_tfs_to_camera(vision_asset: str, world_tfs: np.ndarray):
    assert world_tfs.shape[1] == 4 and world_tfs.shape[2] == 4

    # Get extrinsics.
    object = vision_asset.split('_')[0]
    cam_trans, cam_axis_angle = file_utils.load_camera_extrinsics(object)

    cam_tfs = np.zeros_like(world_tfs)
    for i in range(world_tfs.shape[0]):
        cam_tfs[i] = math_utils.world_to_camera(
            world_tfs[i], cam_trans, cam_axis_angle)
        
    return cam_tfs

def get_xyz_axis_locations_from_pose(pose: np.ndarray):
    assert pose.shape == (4, 4)

    # The transformation definition requires pos_quat.
    pos_quat = math_utils.trans_mat_to_pos_quat(pose).squeeze()

    return math_utils.transform_point_coordinates_given_pose(
        CAMERA_AXES_IN_CAMERA_FRAME.reshape(6,3), pos_quat).reshape(3,2,3)

def get_xyz_axis_locations_over_time_from_poses(poses: np.ndarray):
    assert poses.shape[1] == 4 and poses.shape[2] == 4

    xyz_axes = np.zeros((3, poses.shape[0], 2, 3))
    for i in range(poses.shape[0]):
        xyz_axes[:, i] = get_xyz_axis_locations_from_pose(poses[i])

    return xyz_axes

"""(n_points, 3) -> (n_points, 2)"""
def convert_3d_points_to_camera_pixels(
        points_cam: np.ndarray, intrinsics_P_mat: np.ndarray):
    assert points_cam.shape[1] == 3 and points_cam.ndim == 2

    # Convert to homogeneous coordinates.
    points_xyz1 = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))

    # Project to camera pixels.
    uvw = intrinsics_P_mat @ points_xyz1.T
    x_pixel = uvw[0] / uvw[2]
    y_pixel = uvw[1] / uvw[2]

    # Return as (n_points, 2) array.
    return np.stack((x_pixel, y_pixel)).T

"""(3_xyz, n_timesteps, 2, 3) -> (3_xyz, n_timesteps, 2, 2)"""
def convert_triad_points_to_camera_pixels(
        triad_points_cam: np.ndarray, intrinsics_P_mat: np.ndarray):
    """Converts a (3_xyz, n_timesteps, 2, 3) array of 3D points in meters to a
    (3_xyz, n_timesteps, 2, 2) array of pixel coordinates in the camera image.
    """
    assert triad_points_cam.shape[0] == 3
    assert triad_points_cam.shape[2] == 2
    assert triad_points_cam.shape[3] == 3
    assert triad_points_cam.ndim == 4

    n_timesteps = triad_points_cam.shape[1]

    xy_pixels = np.zeros((3, n_timesteps, 2, 2))

    # Iterate over the x, y, and z axes.
    for xyz_i in range(3):
        all_axis_points = triad_points_cam[xyz_i].reshape(-1, 3)
        xy_pixels[xyz_i] = convert_3d_points_to_camera_pixels(
            all_axis_points, intrinsics_P_mat).reshape(n_timesteps, 2, 2)

    return xy_pixels

"""(3_xyz, n_timesteps, 2, 2)"""
def plot_pose_axes_over_time(pose_axes: np.ndarray, linewidth: int = 1):
    n_triads = int(pose_axes.shape[1] / FRAMES_TO_SKIP)
    # opacity_hex = [int((i+1) * 1.0 / n_triads * 255) for i in range(n_triads)]
    # opacity_hex = [opacity for opacity in reversed(opacity_hex)]

    opacity_hex = OPACITY_OVER_TIME[:n_triads]

    for i in range(0, pose_axes.shape[1], FRAMES_TO_SKIP):
        plt.plot(pose_axes[0, i, :, 0], pose_axes[0, i, :, 1],
            color=f'#ff0000{opacity_hex[i]:02x}', linewidth=linewidth)
        plt.plot(pose_axes[1, i, :, 0], pose_axes[1, i, :, 1],
            color=f'#00ff00{opacity_hex[i]:02x}', linewidth=linewidth)
        plt.plot(pose_axes[2, i, :, 0], pose_axes[2, i, :, 1],
            color=f'#0000ff{opacity_hex[i]:02x}', linewidth=linewidth)

def plot_pose_pos_over_time(pose_axes: np.ndarray):
    # Get all of the poses over time.
    x_pixels = pose_axes[0, :, 0, 0]
    y_pixels = pose_axes[0, :, 0, 1]

    # Create line segments
    points = np.array([x_pixels, y_pixels]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection
    lc = LineCollection(segments, colors=COM_TRAJ_COLORS_OVER_TIME, linewidth=4)

    # Create a plot
    # fig, ax = plt.subplots()
    plt.gca().add_collection(lc)

    # plt.plot(x_pixels, y_pixels, color='orange', linestyle='--', linewidth=1)

def load_viewable_mesh(mesh_path: str):
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Check if the mesh has vertex normals; if not, compute them.
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Check if the mesh has face normals; if not, compute them.
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    return mesh

def rotate_and_capture(mesh, num_images: int, output_file: str):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    view_control = vis.get_view_control()

    with TemporaryDirectory(prefix="mesh-images-") as tmpdir:
        print(f'Storing temporary files at {tmpdir}')
        # Loop to rotate the mesh and capture images.
        for i in tqdm(range(num_images)):
            view_control.rotate(N_TOTAL_SPIN_UNITS/num_images, 0.0)
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            plt.imsave(f"{tmpdir}/image_{i+1:07d}.png", np.asarray(image))

        vis.destroy_window()

        # Make video with ffmpeg from stored images.
        # -y means overwrite output files without asking.
        # -r {FPS} sets the frame rate.
        # -i {INPUT} specifies the input file pattern.
        # -vcodec libx264 specifies the codec as libx264.
        # -preset slow TODO not sure what this does
        # -crf 18 TODO check: specifies the quality, 0 lossless, 51 worst.
        os.system(f'ffmpeg -y -r 30 -i {tmpdir}/image_%07d.png -vcodec ' + \
                  f'libx264 -preset slow -crf 18 {output_file}')

def make_colorized_mesh_spin_video(
        eval_dir: str, mesh_video_path: str, do_error_color: bool,
        do_learned_mesh: bool = True):
    if do_learned_mesh:
        # Get the mesh from the eval directory.
        learned_mesh_path = op.join(eval_dir, 'bsdf_mesh.obj')
        if not op.exists(learned_mesh_path):
            learned_mesh_path = op.join(eval_dir, 'pll_mesh.obj')
        assert op.exists(learned_mesh_path), f'Could not find {learned_mesh_path}.'
        learned_mesh = load_viewable_mesh(learned_mesh_path)

        # Get the GT mesh from the eval directory.
        true_mesh_path = op.join(eval_dir, 'true_geom_aligned.obj')
        assert op.exists(true_mesh_path), f'Could not find {true_mesh_path}.'
        true_mesh = load_viewable_mesh(true_mesh_path)

        if do_error_color:
            # Compute the chamfer distance between the learned mesh's vertices and
            # sampled points on the true mesh.
            true_mesh_samples = true_mesh.sample_points_poisson_disk(2000)
            _, learned_dists = eval_utils.point_wise_chamfer_distance(
                np.asarray(learned_mesh.vertices),
                np.asarray(true_mesh_samples.points)
            )

            cmap = plt.get_cmap(COLOR_MAP)
            dists_to_scale = np.clip(
                (learned_dists - LOWER_DISTANCE)/(UPPER_DISTANCE - LOWER_DISTANCE),
                0, 1
            )
            colors = cmap(dists_to_scale)[:, :3]
            learned_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        else:
            if not learned_mesh.has_vertex_colors():
                learned_mesh.compute_vertex_normals()

            # Check if the mesh has face normals; if not, compute them.
            if not learned_mesh.has_triangle_normals():
                learned_mesh.compute_triangle_normals()

            # Rename the video to indicate it's not colored.
            mesh_video_path = mesh_video_path.replace('.mp4', '_uncolored.mp4')

        mesh_to_vis = learned_mesh

    else:
        # Get the GT mesh from the eval directory.
        true_mesh_path = op.join(eval_dir, 'true_geom_aligned.obj')
        assert op.exists(true_mesh_path), f'Could not find {true_mesh_path}.'
        mesh_to_vis = load_viewable_mesh(true_mesh_path)

        # Make sure the video path uses the right name.
        object_name = op.basename(mesh_video_path).split('_')[0]
        mesh_video_path = op.join(
            op.dirname(mesh_video_path), f'{object_name}_true.mp4')

    rotate_and_capture(mesh_to_vis, 36, mesh_video_path)


#######################################################################
@click.group()
def cli():
    pass


@cli.command('mesh')
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with pose outputs to use.")
@click.option('--nerf-bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with NeRF outputs to use.")
@click.option('--pll-id',
              type=str,
              default=None,
              help="what PLL run ID to look up -- only include if want to " + \
                "evaluate PLL-only baseline.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (can't choose 0 since that " + \
                "means use TagSLAM poses).")
@click.option('--color/--no-color',
              type=bool,
              default=True,
              help="whether to color the mesh according to vertex-wise CD " + \
                "error.")
@click.option('--learned-mesh/--true-mesh',
              type=bool,
              default=True,
              help="whether to visualize the learned or true mesh.")

def main_command(vision_asset: str, bundlesdf_id: str, nerf_bundlesdf_id: str,
                 pll_id: str, cycle_iteration: int, color: bool,
                 learned_mesh: bool):
    if cycle_iteration == 0:
        assert pll_id is not None, f'Need {pll_id=} if cycle_iteration is 0.'
        assert bundlesdf_id is None, f'Cannot have {bundlesdf_id=} if ' + \
            f'cycle_iteration is 0.'
        assert nerf_bundlesdf_id is None, f'Cannot have {nerf_bundlesdf_id=}' +\
            f' if cycle_iteration is 0.'
    assert cycle_iteration >= 0, f'Invalid {cycle_iteration=}.'
    assert '_' in vision_asset, f'Invalid {vision_asset=}.'

    if pll_id is None:
        assert bundlesdf_id is not None, f'Need {bundlesdf_id=} if not ' + \
            f'{pll_id=}.'

        # Decode the BundleSDF run ID.
        tracking_bundlesdf_id = bundlesdf_id
        if tracking_bundlesdf_id[:13] != 'bundlesdf_id_':
            tracking_bundlesdf_id = f'bundlesdf_id_{tracking_bundlesdf_id}'
        if nerf_bundlesdf_id is None:
            nerf_bundlesdf_id = tracking_bundlesdf_id
        elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
            nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'

    else:
        assert bundlesdf_id is None and nerf_bundlesdf_id is None, f'Can ' + \
            f'only have {pll_id=} if not {bundlesdf_id=} or ' + \
            f'{nerf_bundlesdf_id=}.'

        # Decode the PLL run ID.
        if pll_id[:7] != 'pll_id_':
            pll_id = f'pll_id_{pll_id}'

    # Get the evaluation directory.
    eval_dir = file_utils.evaluation_subdir(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        tracking_bundlesdf_id=bundlesdf_id, nerf_bundlesdf_id=nerf_bundlesdf_id,
        pll_id=pll_id, create=False)
    assert op.exists(eval_dir), f'Could not find {eval_dir}.'

    # Create a mesh video directory and path.
    mesh_video_path = file_utils.inspection_mesh_video_filepath(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        tracking_bundlesdf_id=bundlesdf_id, nerf_bundlesdf_id=nerf_bundlesdf_id,
        pll_id=pll_id, create=True
    )

    make_colorized_mesh_spin_video(
        eval_dir, mesh_video_path, do_error_color=color,
        do_learned_mesh=learned_mesh)


@cli.command('video')
def process_video_comand():
    for bsdf_id in BSDF_IDS:
        cycle_iteration = CYCLE_ITERATIONS_BY_BSDF_ID[bsdf_id]
        for num in NUMBERS:
            for obj in OBJECTS:
                vision_asset = f'{obj}_{num}'

                # First check if prediciton videos already exist.
                eval_dir = file_utils.evaluation_subdir(
                    dataset=vision_asset, cycle_iteration=cycle_iteration,
                    tracking_bundlesdf_id=bsdf_id, nerf_bundlesdf_id=bsdf_id,
                    pll_id=None, create=False
                )
                if not op.exists(eval_dir):
                    print(f'Did not find {eval_dir}, skipping.')
                    continue

                prediction_video_path = \
                    file_utils.evaluation_toss_prediction_video_filepath(
                        dataset=vision_asset, cycle_iteration=cycle_iteration,
                    tracking_bundlesdf_id=bsdf_id, nerf_bundlesdf_id=bsdf_id,
                        pll_id=None,
                    )
                if op.exists(prediction_video_path):
                    print(f'  Found video {op.basename(eval_dir)}; skip.')
                    continue

                # Generate the video.
                print(f'\nVIDEO FOR {op.basename(eval_dir)}\n')
                history = evaluate.traverse_run_history_from_bsdf(
                    vision_asset, bsdf_id, cycle_iteration)
                pgen = PredictionOverlayGenerator(
                    vision_asset=vision_asset,
                    history=history,
                    nerf_bundlesdf_id=bsdf_id,
                    bsdf_only=False,  # automatically set to True if tagless
                    remote=True
                )
                pgen.make_overlay_video()

                del pgen, history, prediction_video_path, eval_dir, vision_asset

    for pll_id in PLL_IDS:
        cycle_iteration = CYCLE_ITERATIONS_BY_PLL_ID[pll_id]
        for num in NUMBERS:
            for obj in OBJECTS:
                vision_asset = f'{obj}_{num}'

            #for pll_id in PLL_IDS:
                #cycle_iteration = CYCLE_ITERATIONS_BY_PLL_ID[pll_id]

                # First check if prediciton videos already exist.
                eval_dir = file_utils.evaluation_subdir(
                    dataset=vision_asset, cycle_iteration=cycle_iteration,
                    tracking_bundlesdf_id=None, nerf_bundlesdf_id=None,
                    pll_id=pll_id, create=False
                )
                if not op.exists(eval_dir):
                    print(f'Did not find {eval_dir}, skipping.')
                    continue

                prediction_video_path = \
                    file_utils.evaluation_toss_prediction_video_filepath(
                        dataset=vision_asset, cycle_iteration=cycle_iteration,
                        tracking_bundlesdf_id=None, nerf_bundlesdf_id=None,
                        pll_id=pll_id,
                    )
                if op.exists(prediction_video_path):
                    print(f'  Found prediction video {op.basename(eval_dir)}' +\
                          f'; skip.')
                    continue

                # Generate the video.
                print(f'\nVIDEO FOR {op.basename(eval_dir)}\n')
                # history = evaluate.traverse_run_history_from_bsdf(
                #     vision_asset, BSDF_PLL_BUNDLESDF_ID, 2)
                history = evaluate.traverse_run_history_from_pll(
                    vision_asset, pll_id, cycle_iteration)
                pgen = PredictionOverlayGenerator(
                    vision_asset=vision_asset,
                    history=history,
                    nerf_bundlesdf_id=None,
                    bsdf_only=False,  # automatically set to True if tagless
                    remote=True
                )
                pgen.make_overlay_video()

                del pgen, history, prediction_video_path, eval_dir, vision_asset


@cli.command('triad')
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with pose outputs to use.")
@click.option('--nerf-bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with NeRF outputs to use.")
@click.option('--pll-id',
              type=str,
              default=None,
              help="what PLL run ID to look up -- only include if want to " + \
                "evaluate PLL-only baseline.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (can't choose 0 since that " + \
                "means use TagSLAM poses).")

def main_command(vision_asset: str, bundlesdf_id: str, nerf_bundlesdf_id: str,
                 pll_id: str, cycle_iteration: int):
    if cycle_iteration == 0:
        assert pll_id is not None, f'Need {pll_id=} if cycle_iteration is 0.'
        assert bundlesdf_id is None, f'Cannot have {bundlesdf_id=} if ' + \
            f'cycle_iteration is 0.'
        assert nerf_bundlesdf_id is None, f'Cannot have {nerf_bundlesdf_id=}' +\
            f' if cycle_iteration is 0.'
    assert cycle_iteration >= 0, f'Invalid {cycle_iteration=}.'
    assert '_' in vision_asset, f'Invalid {vision_asset=}.'

    if pll_id is None:
        assert bundlesdf_id is not None, f'Need {bundlesdf_id=} if not ' + \
            f'{pll_id=}.'

        # Decode the BundleSDF run ID.
        tracking_bundlesdf_id = bundlesdf_id
        if tracking_bundlesdf_id[:13] != 'bundlesdf_id_':
            tracking_bundlesdf_id = f'bundlesdf_id_{tracking_bundlesdf_id}'
        if nerf_bundlesdf_id is None:
            nerf_bundlesdf_id = tracking_bundlesdf_id
        elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
            nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'

        # Obtain the run history.
        history = evaluate.traverse_run_history_from_bsdf(
            vision_asset, tracking_bundlesdf_id, cycle_iteration)

    else:
        assert bundlesdf_id is None and nerf_bundlesdf_id is None, f'Can ' + \
            f'only have {pll_id=} if not {bundlesdf_id=} or ' + \
            f'{nerf_bundlesdf_id=}.'

        # Decode the PLL run ID.
        if pll_id[:7] != 'pll_id_':
            pll_id = f'pll_id_{pll_id}'

        # Obtain the run history.
        history = evaluate.traverse_run_history_from_pll(
            vision_asset, pll_id, cycle_iteration)

    print(f'\nFound run history for {vision_asset}:')
    for key, val in history.items():
        print(f'\t{key} : {val}')

    start_toss = int(vision_asset.split('_')[1].split('-')[0])

    # Do the overlay generation.
    pgen = PredictionOverlayGenerator(
        vision_asset=vision_asset, history=history,
        nerf_bundlesdf_id=nerf_bundlesdf_id, bsdf_only=False, remote=True)

    # Iterate over all tosses.
    pgen._get_absolute_frames()
    for toss_i_0, toss_frame_1 in enumerate(pgen.start_frames):
        # Get the first image of the toss.
        toss_i = toss_i_0 + start_toss  # convert to 1-indexed
        toss_frame_0 = toss_frame_1 - 1  # convert to 0-indexed
        first_image = pgen.rgb_images[toss_frame_0]

        # The predicted and BundleSDF trajectories are stored under
        # pgen.predicted_trajs and pgen.bundlesdf_trajs, both dictionaries with
        # toss number keys.  These are 4x4 transformation matrices in world
        # frame.
        predicted_traj = pgen.predicted_trajs[toss_i]
        bundlesdf_traj = pgen.bundlesdf_trajs[toss_i]

        # Convert to camera frame.
        predicted_traj_cam = convert_world_tfs_to_camera(
            vision_asset, predicted_traj)
        bundlesdf_traj_cam = convert_world_tfs_to_camera(
            vision_asset, bundlesdf_traj)

        # Convert to 3D points in camera frame of the xyz axes of the triad.
        # These wil be of size (3_xyz, n_timesteps, 2, 3).
        predicted_axes_cam = get_xyz_axis_locations_over_time_from_poses(
            predicted_traj_cam)
        bundlesdf_axes_cam = get_xyz_axis_locations_over_time_from_poses(
            bundlesdf_traj_cam) * TRACKING_TO_PRED_AXIS_SIZE_RATIO

        # Get in camera pixels.
        intrinsics_P_mat = get_camera_intrinsics(vision_asset)
        predicted_pixels = convert_triad_points_to_camera_pixels(
            predicted_axes_cam, intrinsics_P_mat)
        bundlesdf_pixels = convert_triad_points_to_camera_pixels(
            bundlesdf_axes_cam, intrinsics_P_mat)

        # Plot the image.
        plt.figure()
        plt.imshow(first_image)
        plot_pose_pos_over_time(bundlesdf_pixels)
        plot_pose_axes_over_time(predicted_pixels, linewidth=LINEWIDTH)
        plt.savefig(file_utils.inspection_triad_image_filepath(
            vision_asset, bundlesdf_id, nerf_bundlesdf_id, cycle_iteration,
            f'pred_gt_{toss_i}'))
        print(f'Wrote image for toss {toss_i}.')
        plt.close()




if __name__ == '__main__':
    cli()
