"""Generate an overlay video to qualitatively observe the quality of the pose
estimation (can compare BundleSDF and TagSLAM) and of the shape reconstruction.
"""

import click
import numpy as np
import os
import os.path as op
import pdb
from PIL import Image, ImageDraw, ImageFont
import cv2
from tempfile import TemporaryDirectory
import rosbag
from selenium import webdriver
import yaml
from tqdm import tqdm

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import math_utils
import file_utils



# Overlay video settings.
TAGSLAM_COLOR = 0xff0000
BUNDLESDF_COLOR = 0x00ff00
PREDICTION_COLOR = 0x800080

########################
# Frames
# (W) World
# (M) Meshcat
# (C) Camera
# (A) TagSLAM origin
# (B) BundleSDF origin
# (P) Prediction BundleSDF origin


class OverlayVideoGenerator:
    """Generate an overlay video to compare TagSLAM and BundleSDF poses to the
    observed RGB images."""
    def __init__(self, vision_asset: str, cycle_iteration: int,
                 tracking_bundlesdf_id: str = None,
                 nerf_bundlesdf_id: str = None, 
                 pll_id: str = None,
                 bsdf_only: bool = False, remote: bool = False, 
                 gt_mesh: bool = False, bsdf_offset_frames: int = 1) -> None:
        # First decode the system and start/end tosses from the provided asset
        # directory.
        assert cycle_iteration >= 0, f'Invalid {cycle_iteration=}.'
        assert '_' in vision_asset, f'Invalid {vision_asset=}.'
        object = '_'.join(vision_asset.split('_')[:-1])

        start_toss = int(vision_asset.split('_')[-1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
                f'-{end_toss} inferred from {vision_asset=}.'

        assert not (gt_mesh and pll_id is not None), \
            f'Cannot use mesh from {pll_id=} and gt_mesh at the same time.'
        
        # If pll_id is provided, use pll geometry. Otherwise, use bsdf geometry.
        if pll_id is not None:
            if pll_id[:7] != 'pll_id_':
                pll_id = f'pll_id_{pll_id}'
            self.pll_id = pll_id
        else:
            self.pll_id = None
        
        # Decode the BundleSDF run ID.
        if tracking_bundlesdf_id is not None:
            if tracking_bundlesdf_id[:13] != 'bundlesdf_id_':
                tracking_bundlesdf_id = f'bundlesdf_id_{tracking_bundlesdf_id}'
            if nerf_bundlesdf_id is None:
                nerf_bundlesdf_id = tracking_bundlesdf_id
            elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
                nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'
            self.tracking_bundlesdf_id = tracking_bundlesdf_id
            self.nerf_bundlesdf_id = nerf_bundlesdf_id
        else:
            self.tracking_bundlesdf_id = None
            self.nerf_bundlesdf_id = None

        # Automatically detect if BundleSDF-only is necessary based on if the
        # object is a tagless one.
        object = '_'.join(vision_asset.split('_')[:-1])
        if object in file_utils.TAGLESS_OBJECTS:
            bsdf_only = True
            print(f'Automatically setting {bsdf_only=} for tagless {object=}.')
        
        if cycle_iteration == 0:
            assert object not in file_utils.TAGLESS_OBJECTS, f'Cannot use ' + \
                f'TagSLAM poses for tagless {object=}.'
            # TODO: Debug this mode. 
            print('The current implementation of tagslam pose visualization may be incorrect. ')

        self.vision_asset = vision_asset
        self.cycle_iteration = cycle_iteration
        self.remote = remote
        self.start_toss = start_toss
        self.end_toss = end_toss
        self.object = object
        self.bsdf_only = bsdf_only

        # Introduce a start frame offset.  For this class, this will always be
        # zero.  However for prediction videos which can feature images before
        # the first prediction/tracking, this offset can determine when the
        # tracking starts into the video.
        self.start_frame_offset = bsdf_offset_frames - 1
        # bsdf_offset_frames indicates the first frame idx that a BundleSDF run starts at. 
        # bsdf_offset_frames = 1 is the default because its index is 1-based. 

        # Get the camera intrinsics and extrinsics.
        self.fx, self.fy, self.cx, self.cy = file_utils.load_camera_intrinsics(
            self.object)
        self.cam_trans, self.cam_axis_vec = \
            file_utils.load_camera_extrinsics(object)
        
        self._get_rgb_images()

        # Load the tracking
        self._get_bundletrack_poses_in_cam()
        self._get_tagslam_poses_in_world()
        if self.tracking_bundlesdf_id is not None:
            assert self.rgb_images.shape[0] == \
                self.bundlesdf_poses_in_cam.shape[0] + self.start_frame_offset, \
                    f'Inconsistent data sizes' + \
                    f' {self.rgb_images.shape[0]=}, ' + \
                    f'{self.bundlesdf_poses_in_cam.shape[0]=},' + \
                    f'{self.tagslam_poses_in_world.shape[0]=},' + \
                    f'{self.start_frame_offset=}.'
            
        if not self.bsdf_only:
            assert self.rgb_images.shape[0] == \
                self.tagslam_poses_in_world.shape[0] + self.start_frame_offset, \
                    f'Inconsistent data ' + \
                    f'sizes {self.rgb_images.shape[0]=}, ' + \
                    f'{self.tagslam_poses_in_world.shape[0]=},' + \
                    f'{self.start_frame_offset=}.'

        # Load the mesh file.
        if gt_mesh:
            self.mesh_file = file_utils.aligned_true_geometry_filepath(
                dataset=vision_asset, tracking_bundlesdf_id=tracking_bundlesdf_id,
                nerf_bundlesdf_id=nerf_bundlesdf_id, cycle_iteration=cycle_iteration
            )
        elif self.pll_id is not None:
            urdf_results_dir = file_utils.get_pll_urdf_output_dir(
                system=vision_asset, cycle_iteration=cycle_iteration,
                run_name=self.pll_id
            )
            self.mesh_file = op.join(urdf_results_dir, 'body_vis.obj')
        else:
            # Get the path to the mesh file generated by BundleSDF.
            nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
                dataset=vision_asset, cycle_iteration=cycle_iteration,
                tracking_bundlesdf_id=tracking_bundlesdf_id,
                nerf_bundlesdf_id=nerf_bundlesdf_id
            )
            self.mesh_file = op.join(nerf_results_dir, 'textured_mesh.obj')

        # Plan to put the output video in a single directory for all overlay
        # videos.
        self.output_file = file_utils.inspection_overlay_video_filepath(
            dataset=vision_asset, tracking_bundlesdf_id=tracking_bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id, pll_id=self.pll_id,
            cycle_iteration=cycle_iteration, gt_mesh=gt_mesh
        )
        if not op.exists(op.dirname(self.output_file)):
            os.makedirs(op.dirname(self.output_file))
        # Put the output video in the same directory as the pll inputs though 
        # pll does not need the videos. It is for easier visualization. 
        self.output_file_to_pll_input_dir = file_utils.contactnets_input_bsdf_overlay_video_path(
            vision_asset, cycle_iteration, tracking_bundlesdf_id,
            nerf_bundlesdf_id, gt_mesh=gt_mesh)

    def _get_bundletrack_poses_in_cam(self) -> None:
        if self.tracking_bundlesdf_id is None:
            return
        ob_in_cam_dir = file_utils.bundlesdf_pose_dir(
            self.vision_asset, self.cycle_iteration, self.tracking_bundlesdf_id)

        frame_num = len([name for name in os.listdir(ob_in_cam_dir)])
        poses = np.zeros((frame_num, 4, 4))
        for frame_id in range(1, frame_num+1):
            pose = np.loadtxt(op.join(ob_in_cam_dir, "%04i.txt" % frame_id))
            poses[frame_id-1] = pose

        self.bundlesdf_poses_in_cam = poses

    def _get_tagslam_poses_in_world(self) -> None:
        if self.bsdf_only:
            return

        # Use the synchronized pose estimates so the poses are aligned with the
        # images.
        tagslam_dir = file_utils.synchronized_tagslam_pose_dir(
            self.vision_asset, check_exists=True)

        # TagSLAM data stored as [t, x, y, z, qx, qy, qz, qw].
        tagslam_data = np.loadtxt(op.join(tagslam_dir, 'synced_tagslam.txt'))

        # deal with bsdf_offset_frames
        tagslam_data = tagslam_data[self.start_frame_offset:]

        # Return TagSLAM poses as 4x4 homogeneous transforms.
        tagslam_poses_in_world = tagslam_data[:, 1:]
        poses = np.zeros((tagslam_poses_in_world.shape[0], 4, 4))
        for i in range(tagslam_poses_in_world.shape[0]):
            pose = tagslam_poses_in_world[i]
            poses[i] = math_utils.pos_quat_to_trans_mat(pose)

        self.tagslam_poses_in_world = poses

    def _get_rgb_images(self) -> None:
        rgb_images = []
        rgb_dir = file_utils.bundlesdf_video_rgb_dir(self.vision_asset)
        for filename in sorted(os.listdir(rgb_dir)):
            rgb_images.append(np.array(Image.open(op.join(rgb_dir, filename))))
        
        self.rgb_images = np.array(rgb_images)

        self.image_width = self.rgb_images.shape[2]
        self.image_height = self.rgb_images.shape[1]

    def _add_meshcat_objects(self, vis: meshcat.Visualizer) -> None:
        # # Only render the cube in the TagSLAM trajectory if cube asset.
        # if 'cube' in self.vision_asset:
        #     vis["tagslam_cube"].set_object(
        #         g.Box([0.1048, 0.1048, 0.1048]),
        #         g.MeshLambertMaterial(
        #             color=TAGSLAM_COLOR, reflectivity=0.0, transparent=0,
        #             opacity=.4)
        #     )
        # if not self.bsdf_only:
        #     vis["tagslam_triad"].set_object(g.triad(scale=0.05))
        vis["bundlesdf_triad"].set_object(g.triad(scale=0.1))
        vis["bundlesdf_mesh"].set_object(
            g.ObjMeshGeometry.from_file(self.mesh_file),
            g.MeshLambertMaterial(
                color=BUNDLESDF_COLOR, reflectivity=0.0, transparent=0,
                opacity=.4)
        )

    def _set_up_meshcat(self) -> None:
        # Can specify zmq_url="tcp://127.0.0.1:6000" argument after the first
        # run but the frame will mismatch.
        print('\nNo need to open this link: ', end='')
        vis = meshcat.Visualizer()

        ##################
        self._add_meshcat_objects(vis)
        ########################
        base_url = "http://127.0.0.1"
        meshcat_url = f'{base_url}:{vis.url().split(":")[-1]}'

        ### Need x server to run. Either run locally or run remotely with x
        # forward configured.
        self.change_display = False
        if self.remote and os.environ.get('DISPLAY') != ':99':
            # Run Xvfb to create a virtual display.
            print("Running Xvfb (virtual display) for rendering.")
            import subprocess
            self.xvfb_process = subprocess.Popen(['Xvfb', ':99', '-screen', '0',
                                                  '640x480x24'])
            print(f"Changing display environment from " + \
                  f"{os.environ['DISPLAY']} variable to :99")
            self.old_display = os.environ['DISPLAY']
            os.environ['DISPLAY'] = ':99'
            self.change_display = True

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        if self.remote:
            options.add_argument('--no-sandbox')
        self.driver = webdriver.Chrome(options=options)

        # Set the desired window size.
        self.driver.set_window_size(self.image_width, self.image_height)
        self.driver.get(meshcat_url)

        # Compute T_WC, transform from world to camera.
        # TODO For some reason, need to use the inverse transform.  Maybe
        # something is misnamed.
        self.T_WC = math_utils.extrinsics_T_CW(
            self.cam_trans, self.cam_axis_vec)

        # T_MC, look along meshcat's z-axis but rotote by 180 degrees.
        self.T_MC = tf.translation_matrix([0, 0, -1]) @ \
            tf.rotation_matrix(np.pi, (0,0,1))
        self.T_MW = self.T_MC @ tf.inverse_matrix(self.T_WC)

        vis["cam"].set_transform(self.T_WC)
        vis["cam_view"].set_transform(self.T_WC)

        # The PerspectiveCamera API does not allow for vertical adjustment of
        # the center of the optical axis.  The horizontal offset is accomplished
        # via the filmOffset according to filmGauge's scale.
        fov_y = 2*np.arctan(self.image_height/(2*self.fy))*180/np.pi
        cam = g.PerspectiveCamera(
            fov=fov_y, zoom=1, aspect=self.image_width/self.image_height,
            filmGauge=self.image_width, filmOffset=self.image_width/2 - self.cx)
        vis["/Cameras/default/rotated"].set_object(cam)
        vis["/Cameras/default/rotated/<object>"].set_property(
            "position", [0,0,0])
        vis["/Cameras/default"].set_transform(self.T_MC)

        ###########################
        # view in meshcat save to images
        # Turn off background, axes, and grid.
        vis['/Background'].set_property("visible", False)
        vis['/Grid'].set_property("visible", False)
        vis['/Axes'].set_property("visible", False)

        self.vis = vis

    def _clean_up_meshcat(self) -> None:
        self.vis.delete()
        # If not exited properly, orphan chrome processes will remain active.
        self.driver.quit()
        print(f'Done with keyframe overlay images.')
        if self.remote and self.change_display:
            # If not exited properly, orphan Xvfb processes will remain active.
            self.xvfb_process.terminate()
            self.xvfb_process.wait()
            os.environ['DISPLAY'] = self.old_display
            print("Terminated Xvfb process")
            print(f"Recovered {os.environ['DISPLAY']=}.")

    def _get_absolute_frames(self) -> None:
        """Determine the start and end frames of each PLL toss relative to the
        beginning of the full overlay video.  This is necessary to label the
        video with the toss number."""
        relative_start_frames = np.array([file_utils.load_field_from_yaml(
            self.object, toss_i, 'start_frame') for toss_i in range(
                self.start_toss, self.end_toss+1)])
        relative_end_frames = np.array([file_utils.load_field_from_yaml(
            self.object, toss_i, 'end_frame') for toss_i in range(
                self.start_toss, self.end_toss+1)])
        start_ros_times = np.array([file_utils.load_toss_time_from_yaml(
            self.object, toss_i, 'start_time', as_ros_time=True) for toss_i \
                in range(self.start_toss, self.end_toss+1)])
        cnets_data_gen_dir = file_utils.cnets_data_gen_dataset_dir(
            self.vision_asset, check_exists=True)
        bundlesdf_times = np.loadtxt(
            op.join(cnets_data_gen_dir, 'bundlesdf_timestamps.txt'))

        self.start_frames = math_utils.convert_relative_frames_to_absolute(
            relative_start_frames, bundlesdf_times, start_ros_times)
        self.end_frames = math_utils.convert_relative_frames_to_absolute(
            relative_end_frames, bundlesdf_times, start_ros_times)

    def _within_which_toss(self, image_frame_i: int) -> int:
        """Determine if the current frame index is within a toss or not.  If so,
        return the toss number.  Otherwise, return None."""
        if not hasattr(self, 'start_frames'):
            self._get_absolute_frames()

        # A frame is within a PLL toss if the last toss whose start frame it
        # satisfies is the first toss whose end frame it satisfies.
        good_starts = np.where(image_frame_i >= self.start_frames)[0]
        good_ends = np.where(image_frame_i < self.end_frames)[0]
        if len(good_starts) > 0 and len(good_ends) > 0 and \
            good_starts[-1] == good_ends[0]:
            toss_i = self.start_toss + good_starts[-1]
            return toss_i
        return None

    def _add_watermark(self, im: Image, image_frame_i: int) -> Image:
        """Add a label to the image to specify whether the portion of the video
        is part of the PLL toss or not."""
        # First determine if the frame index is within a PLL toss.
        toss_i = self._within_which_toss(image_frame_i)

        if toss_i is not None:
            # Add a PLL toss label to the image.
            draw = ImageDraw.Draw(im)
            draw.polygon([(25, 435), (100, 435), (100, 475), (25, 475)],
                         fill='black')
            font_path = op.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
            font = ImageFont.truetype(font_path, 20)
            draw.text((30, 440), f'Toss {toss_i}', fill='white',
                      font=font)

        return im

    def _set_meshcat_object_poses(self, frame_i: int, T_WA: np.ndarray,
                          T_CB: np.ndarray) -> None:
        # if 'cube' in self.vision_asset:
        #     self.vis["tagslam_cube"].set_transform(self.T_MW @ T_WA)
        # if T_WA is not None:
        #     self.vis["tagslam_triad"].set_transform(self.T_MW @ T_WA)
        if self.cycle_iteration == 0:
            # Use TagSLAM poses.
            self.vis["bundlesdf_mesh"].set_transform(self.T_MW @ T_WA)
            self.vis["bundlesdf_triad"].set_transform(self.T_MW @ T_WA)
        else:
            if T_CB is not None:
                self.vis["bundlesdf_triad"].set_transform(self.T_MC @ T_CB)
                self.vis["bundlesdf_mesh"].set_transform(self.T_MC @ T_CB)

            else:
                out_of_view_tf = self.T_MC @ tf.translation_matrix([0, 0, -1])

                self.vis["bundlesdf_triad"].set_transform(out_of_view_tf)
                self.vis["bundlesdf_mesh"].set_transform(out_of_view_tf)

    def _render_one_image(self, frame_i: int, T_WA: np.ndarray,
                          T_CB: np.ndarray) -> Image:
        """Given the frame index, TagSLAM pose in world, and BundleSDF pose in
        camera, render and return the image with the overlay."""
        im = Image.fromarray(self.rgb_images[frame_i]).convert('RGB')

        self._set_meshcat_object_poses(frame_i, T_WA, T_CB)
        mesh_im = self.vis.get_image()

        # Sadly meshcat's PerspectiveCamera doesn't compensate for cy, only cx.
        # So the below manually compensates for cy.
        shift_pixels_up = round(2*(self.image_height/2 - self.cy))
        upper_buffer = np.zeros(
            (max(0, -shift_pixels_up), self.image_width, 4)).astype(np.uint8)
        lower_buffer = np.zeros(
            (max(0, shift_pixels_up), self.image_width, 4)).astype(np.uint8)

        original_data = np.array(mesh_im)
        large = np.concatenate(
            (upper_buffer, original_data, lower_buffer), axis=0)
        start = max(0,shift_pixels_up)
        end = original_data.shape[0] + start
        small = large[start:end, :, :]
        assert original_data.shape == small.shape
        mesh_im = Image.fromarray(small).convert('RGBA')

        im.paste(mesh_im, (0,0), mask = mesh_im)

        # Add annotation to show what portions of the video are part of a toss
        # trajectory we give to PLL.  Add 1 since the watermarks are determined
        # based on 1-indexing frame numbers.
        self._add_watermark(im, image_frame_i=frame_i+1)

        return im

    def make_overlay_video(self, to_pll_input_dir=False) -> None:
        print(f'Starting overlay video generation (could take minutes).')
        self._set_up_meshcat()

        with TemporaryDirectory(prefix="ros-process-") as tmpdir:
            print(f'Storing temporary files at {tmpdir}')

            # Use tqdm to show a progress bar.
            for i in tqdm(range(self.rgb_images.shape[0])):
                # Get the transformations of BundleSDF and/or TagSLAM tracking
                # for annotations.
                try:
                    index = max(i - self.start_frame_offset, 0)
                    T_WA = None if not hasattr(self, 'tagslam_poses_in_world') \
                        else self.tagslam_poses_in_world[index]
                    T_CB = None if not hasattr(self, 'bundlesdf_poses_in_cam') \
                        else self.bundlesdf_poses_in_cam[index]

                # If the video is longer than the BundleSDF tracking, still
                # render the video but without annotated poses.
                except IndexError:
                    T_WA = None
                    T_CB = None

                im = self._render_one_image(i, T_WA=T_WA, T_CB=T_CB)
                im.save(op.join(tmpdir, f'{i+1:07d}.png'), format="png")

            # Make video with ffmpeg from stored images.
            # -y means overwrite output files without asking.
            # -r {FPS} sets the frame rate.
            # -i {INPUT} specifies the input file pattern.
            # -vcodec libx264 specifies the codec as libx264.
            # -preset slow TODO not sure what this does
            # -crf 18 TODO check: specifies the quality, 0 lossless, 51 worst.
            # -pix_fmt yuv420p to ensure compatibility with most players.
            # (e.g., Windows Media Player)
            output_file = self.output_file if not to_pll_input_dir else \
                self.output_file_to_pll_input_dir
            os.system(f'ffmpeg -y -r 30 -i {tmpdir}/%07d.png -vcodec ' + \
                      f'libx264 -pix_fmt yuv420p -preset slow -crf 18 {output_file}')

        self._clean_up_meshcat()

    def _get_bundletrack_keyframes(self) -> None:
        # Get the keyframe indices from the BundleSDF tracking results' last
        # frame directory, in keyframes.yml.
        self.keyframe_indices = \
            file_utils.load_keyframe_indices_from_nerf_results_yml(
                self.vision_asset, self.cycle_iteration, self.tracking_bundlesdf_id)

        # Get the adjusted keyframe poses from the BundleSDF NeRF results'
        # poses_after_nerf.txt.
        self.keyframe_tfs = \
            file_utils.load_optimized_keyframe_poses_from_nerf_results(
                dataset=self.vision_asset, cycle_iteration=self.cycle_iteration,
                tracking_bundlesdf_id=self.tracking_bundlesdf_id,
                nerf_bundlesdf_id=self.nerf_bundlesdf_id
            )

    def make_optimized_keyframe_overlay_images(self) -> None:
        """For all keyframes, render and save an overlay image with the geometry
        rendered at the optimized pose."""
        print(f'Starting keyframe overlay image generation.')
        self._get_bundletrack_keyframes()
        self._set_up_meshcat()

        output_dir = file_utils.inspection_keyframe_overlay_image_dir(
            dataset=self.vision_asset,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id,
            cycle_iteration=self.cycle_iteration
        )

        if os.listdir(output_dir):
            os.system(f'rm {output_dir}/*')

        for bsdf_i, bsdf_pose in zip(self.keyframe_indices, self.keyframe_tfs):
            # This i is 1-indexed since it's the BundleSDF frame number.  All of
            # the data stored in this class is 0-indexed.
            data_i = bsdf_i - 1

            T_WA = None if self.bsdf_only else \
                self.tagslam_poses_in_world[data_i]
            T_CB = bsdf_pose

            im = self._render_one_image(data_i, T_WA=T_WA, T_CB=T_CB)
            im.save(op.join(output_dir, f'keyframe_{bsdf_i:04d}.png'),
                    format="png")
            print(f'\tGenerated keyframe {bsdf_i}')

        self._clean_up_meshcat()


#######################################################################
@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (can't choose 0 since that " + \
                "means use TagSLAM poses).")
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
              help="what PLL run ID associated with pose outputs to use.")
@click.option('--bsdf-only',
              is_flag=True,
              help="whether to generate just BundleSDF-related data.")
@click.option('--all/--images-only',
              type=bool,
              default=True,
              help="whether to generate everything or just keyframe images.")
@click.option('--remote/--local',
              default=False,
              help="whether to run on a remote server.")
@click.option('--gt-mesh',
              is_flag=True,
              help="whether to use ground truth mesh for video.")
@click.option('--bsdf-offset-frames',
              type=int,
              default=1,
              help="the first frame index that BundleSDF run starts at.")

def main_command(vision_asset: str, 
                 cycle_iteration: int, 
                 bundlesdf_id: str, nerf_bundlesdf_id: str,
                 pll_id: str,bsdf_only: bool, all: bool,
                 remote: bool, gt_mesh: bool,
                 bsdf_offset_frames: int) -> None:
    overlay_video_generator = OverlayVideoGenerator(
        vision_asset, cycle_iteration, 
        tracking_bundlesdf_id=bundlesdf_id, 
        nerf_bundlesdf_id=nerf_bundlesdf_id, 
        pll_id=pll_id,
        bsdf_only=bsdf_only, remote=remote, gt_mesh=gt_mesh,
        bsdf_offset_frames=bsdf_offset_frames
    )

    if all:
        overlay_video_generator.make_overlay_video()
    else:
        print('Skipping generating overlay video.')

    if bundlesdf_id is not None:
        overlay_video_generator.make_optimized_keyframe_overlay_images()


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
