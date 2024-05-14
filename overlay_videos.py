### Remove the dependency of Jupyter notebook. 

import click
import numpy as np
import os
import os.path as op
import pdb
from PIL import Image
from tempfile import TemporaryDirectory
import rosbag
import yaml
from tqdm import tqdm

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import math_utils
import file_utils



# video resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480



def get_bundletrack_poses_in_cam(ob_in_cam_dir: str):
    frame_num = len([name for name in os.listdir(ob_in_cam_dir)])
    print("%i frames in total!"%frame_num)
    poses = np.zeros((frame_num, 4, 4))
    for frame_id in range(1, frame_num+1):
        pose = np.loadtxt(op.join(ob_in_cam_dir, "%04i.txt" % frame_id))
        poses[frame_id-1] = pose

    return poses

def get_tagslam_poses_in_world(vision_asset: str):
    tagslam_dir = file_utils.tagslam_pose_dir(
        vision_asset, check_exists=True)
    # TagSLAM data stored as [t, x, y, z, qx, qy, qz, qw].
    tagslam_data = np.loadtxt(op.join(tagslam_dir, 'tagslam.txt'))

    # Return TagSLAM poses as 4x4 homogeneous transforms.
    tagslam_poses_in_world = tagslam_data[:, 1:]
    poses = np.zeros((tagslam_poses_in_world.shape[0], 4, 4))
    for i in range(tagslam_poses_in_world.shape[0]):
        pose = tagslam_poses_in_world[i]
        poses[i] = math_utils.pos_quat_to_trans_mat(pose)

    return poses



#######################################################################
@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with pose outputs to use.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (can't choose 0 since that " + \
                "means use TagSLAM poses).")

def main_command(vision_asset: str, bundlesdf_id: str, cycle_iteration: int):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert cycle_iteration > 0, f'Invalid cycle iteration: {cycle_iteration}.'
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    object = vision_asset.split('_')[0]

    start_toss = int(vision_asset.split('_')[1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
            f'-{end_toss} inferred from {vision_asset=}.'

    # Decode the BundleSDF run ID.
    if bundlesdf_id[:13] != 'bundlesdf_id_':
        bundlesdf_id = f'bundlesdf_id_{bundlesdf_id}'
    print(f'Processing toss {vision_asset} from BundleSDF run ID ' + \
            f'{bundlesdf_id}.\n')

    # Get the camera intrinsics and extrinsics.
    fx, fy, cx, cy = file_utils.load_camera_intrinsics()
    cam_trans, cam_axis_vec = file_utils.load_camera_extrinsics(object)


    bundlesdf_tracking_dir = file_utils.bundlesdf_pose_dir(
        vision_asset, cycle_iteration, bundlesdf_id)

    # Get poses in [qw, qx, qy, qz, x, y, z] format in size (N, 7).
    bundletrack_poses = get_bundletrack_poses_in_cam(bundlesdf_tracking_dir)
    tagslam_poses = get_tagslam_poses_in_world(vision_asset)
    print("bundletrack: ", bundletrack_poses.shape)

    # extract images
    rgb_images = []
    rgb_dir = file_utils.bundlesdf_video_rgb_dir(vision_asset)
    for filename in sorted(os.listdir(rgb_dir)):
        rgb_images.append(np.array(Image.open(op.join(rgb_dir, filename))))

    # Compute T_WC, transform from world to camera.
    # For some reason, need to use the inverse transform.  Maybe something is
    # misnamed.
    T_WC = math_utils.extrinsics_T_CW(cam_trans, cam_axis_vec)

    fov_y = 2*np.arctan(IMAGE_HEIGHT/(2*fy))*180/np.pi

    # Can specify zmq_url="tcp://127.0.0.1:6000" argument after the first run
    # but the frame will mismatch.
    vis = meshcat.Visualizer()

    ##################
    # Create the cubes. Set the opacity of the "real" to > 0 if you want to see it,
    # for invisible
    vis["tagslam_cube"].set_object(
        g.Box([0.1048, 0.1048, 0.1048]),
        g.MeshLambertMaterial(
            color=0x00ff00, reflectivity=0.0, transparent=0, opacity=.4)
    )
    mesh_file = '/home/bibit/vision/bundlenets/dair_pll/assets/vision_cube/cube_1/geom_for_pll/bundlesdf_iteration_1/bundlesdf_id_00/mesh.obj'
    vis["bundlesdf_mesh"].set_object(
        g.ObjMeshGeometry.from_file(mesh_file),
        g.MeshLambertMaterial(
            color=0xff0000, reflectivity=0.0, transparent=0, opacity=.4)
    )

    ########################
    base_url = "http://127.0.0.1"
    meshcat_url = f'{base_url}:{vis.url().split(":")[-1]}'

    ### Need x server to run. Either run locally or run remotely with x forward
    # configured.
    from selenium import webdriver

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    # Set the desired window size.
    driver.set_window_size(IMAGE_WIDTH, IMAGE_HEIGHT)
    driver.get(meshcat_url)

    ########################
    # Frames
    # (W) World
    # (M) Meshcat
    # (C) Camera
    # (A) TagSLAM origin
    # (B) BundleSDF origin

    vis["cam"].set_transform(T_WC)
    vis["cam_view"].set_transform(T_WC @ tf.translation_matrix([0,0,.05]))

    # T_MC, look along meshcat's z-axis but rotote by 180 degrees.
    T_MC = tf.translation_matrix([0, 0, -1]) @ \
        tf.rotation_matrix(np.pi, (0,0,1))
    T_MW = T_MC @ tf.inverse_matrix(T_WC)

    # The PerspectiveCamera API does not allow for vertical adjustment of the
    # center of the optical axis.  The horizontal offset is accomplished via
    # the filmOffset according to filmGauge's scale.
    cam = g.PerspectiveCamera(
        fov=fov_y, zoom=1, aspect=IMAGE_WIDTH/IMAGE_HEIGHT,
        filmGauge=IMAGE_WIDTH, filmOffset=IMAGE_WIDTH/2 - cx)
    vis["/Cameras/default/rotated"].set_object(cam)
    vis["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0])
    vis["/Cameras/default"].set_transform(T_MC)

    ###########################
    # view in meshcat save to images
    # Turn off background, axes, and grid.
    vis['/Background'].set_property("visible", False)
    vis['/Grid'].set_property("visible", False)
    vis['/Axes'].set_property("visible", False)

    with TemporaryDirectory(prefix="ros-process-") as tmpdir:
        print(tmpdir)

        # Use tqdm to show a progress bar.
        for i in tqdm(range(bundletrack_poses.shape[0])):
            im = Image.fromarray(rgb_images[i]).convert('RGB')

            T_WA = tagslam_poses[i]
            T_CB = bundletrack_poses[i]

            vis["tagslam_cube"].set_transform(T_MW @ T_WA)
            vis["bundlesdf_mesh"].set_transform(T_MC @ T_CB)

            mesh_im = vis.get_image()

            # If vertical offset were significant, can manually adjust for it
            # with the below lines.
            # adjust_pixels = 2
            # original_data = np.array(mesh_im)
            # large = np.concatenate(
            #    (np.zeros((adjust_pixels, 640, 4)).astype(np.uint8),
            #     original_data), axis=0)
            # small = large[:-adjust_pixels, :, :]
            # assert original_data.shape == small.shape
            # mesh_im = Image.fromarray(small).convert('RGBA')

            # mesh_im.show()
            im.paste(mesh_im, (0,0), mask = mesh_im)
            im.show()
            pdb.set_trace()
            # break
            im.save(tmpdir + '/' + f'{i:07d}' + '.png', format="png")

        # Make video with ffmpeg from stored images.
        # -y means overwrite output files without asking.
        # -r {FPS} sets the frame rate.
        # -i {INPUT} specifies the input file pattern.
        # -vcodec libx264 specifies the codec as libx264.
        # -preset slow TODO not sure what this does
        # -crf 18 TODO check: specifies the quality, 0 is lossless, 51 is worst.
        output_file = f'./videos/{vision_asset}.mp4'
        os.system(f'ffmpeg -y -r 30 -i {tmpdir}/%07d.png -vcodec libx264 ' + \
                  f'-preset slow -crf 18 {output_file}')

    vis.delete()
    # If not exited properly, orphan chrome processes will remain active.
    driver.quit()


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
