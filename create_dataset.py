"""Create the BundleSDF dataset in bundlenets/data for toss examples.  This
creates one subfolder inside bundlenets/data/ for a given vision asset, which
can be e.g. cube_2 (cube experiment, toss 2) or cube_2-3 (cube experiment,
tosses 2-3).  The subfolder contains the following data:
    - annotated_poses/
        - XXXX.txt, 0 through N.
        - The 0000.txt is the initial pose of the object's TagSLAM origin,
          reported in camera frame, represented as a 4x4 transformation matrix.
        - All other XXXX.txt files are the identity.  TODO not sure why these
          are necessary.
    - Annotations/
        - a sparse set of XXXX.png files showing masks -- unsure if this needs
          to be created or if it's created by the BundleSDF.
        - TODO:  Not sure where these come from.
    - depth/
        - XXXX.png, 1 through N
    - masks/
        - XXXX.png, 1 through N
        - TODO:  Figure out where these come from.
    - rgb/
        - XXXX.png, 1 through N
    - cam_K.txt
        - The camera intrinsics.  These should be the same for all experiments,
          so they can be copied over from cnets-data-generation/cam_K.txt,
          accessed by file_utils.get_camera_intrinsics_filepath().

In addition to the bundlenets/data directory, this script also creates a folder
at bundlenets/cnets-data-generation/dataset/{vision_asset} that contains the
following:
    - bundlesdf_timestamps.txt
        - The timestamps associated with the images in the BundlesDF dataset.
    - tagslam_poses/
        - XXXX.txt, 1 through N+n
        - tagslam.txt
        - All of the poses are recorded as 7 elements, in order [x, y, z, qx,
          qy, qz, qw].
        - The tagslam.txt file contains all the other XXXX.txt file contents
          plus timestamps at the front:  [t, x, y, z, qx, qy, qz, qw].
    - synchronized_tagslam_poses/
        - XXXX.txt, 1 through N
        - synced_tagslam.txt
        - The synchronized poses are the TagSLAM poses interpolated to the
          timestamps of the BundleSDF images in bundlesdf_timestamps.txt.
    - TODO: Figure out if the other subfolders are required.
"""

import click
import numpy as np
import os
import os.path as op
import pdb

import file_utils
import rosbag_processor


class DatasetCreator:
    """Class to assist with dataset creation for a given vision asset."""
    def __init__(self, vision_asset: str, tagslam_only: bool = False, bsdf_only: bool = False):
        self.vision_asset = vision_asset
        self.tagslam_only = tagslam_only
        self.bsdf_only = bsdf_only

        # Parse the system and start/end tosses from the provided asset name.
        assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
        # want to split {xxx_xxx_..._xxx}_{x-x} into object={xxx_xxx_..._xxx}
        # and {x-x}
        # need to only split on the last underscore
        object = vision_asset.split('_')[:-1]
        self.object = '_'.join(object)

        start_toss = int(vision_asset.split('_')[-1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
            f'-{end_toss} inferred from {vision_asset=}.'
        self.start_toss = start_toss
        self.end_toss = end_toss

        # Get the remaining information to do the processing:  directories,
        # times, camera extrinsics.
        self._set_up_directories()
        self.start_time = file_utils.load_toss_time_from_yaml(
            self.object, start_toss, 'start_time', as_ros_time=True)
        self.end_time = file_utils.load_toss_time_from_yaml(
            self.object, end_toss, 'end_time', as_ros_time=True)
        self.cam_trans, self.cam_axis_vec = \
            file_utils.load_camera_extrinsics(self.object)

    def _set_up_directories(self):
        if not self.tagslam_only:
            # Get the RGB and depth directories.
            self.rgb_dir = file_utils.bundlesdf_video_rgb_dir(
                self.vision_asset, check_exists=False)
            self.depth_dir = file_utils.bundlesdf_video_depth_dir(
                self.vision_asset, check_exists=False)
            self.data_dir = file_utils.bundlesdf_video_dir(
                self.vision_asset, check_exists=False)
            file_utils.assure_created(self.data_dir)

        if not self.bsdf_only:
            self.annotated_poses_dir = file_utils.bundlesdf_annotated_poses_dir(
                self.vision_asset, create=True)

            self.tagslam_dir = file_utils.tagslam_pose_dir(
                self.vision_asset, check_exists=False)
            file_utils.assure_created(self.tagslam_dir)

            self.synced_tagslam_dir = file_utils.synchronized_tagslam_pose_dir(
                self.vision_asset, check_exists=False)
            file_utils.assure_created(self.synced_tagslam_dir)


    def create(self):
        # Get the ROS bag, ensuring the start toss and end tosses are in the
        # same bag.
        self.rosbag_number = file_utils.load_rosbag_number_from_yaml(
            self.object, self.start_toss, second_toss_number=self.end_toss)

        print(f'Processing {self.vision_asset} in ROS bag ' + \
              f'{self.rosbag_number}.\n')

        if not self.tagslam_only:
            self._create_images()
        if not self.bsdf_only:
            self._create_tagslam_poses()
            self._create_synchronized_tagslam_poses()
            self._create_annotated_poses()

        if not self.tagslam_only:
            # Copy the camera intrinsics.
            os.system(f'cp {file_utils.get_camera_intrinsics_filepath()} ' + \
                    f'{op.join(self.data_dir, "cam_K.txt")}')

            self._compute_table_offset()

            print(f'Finished creating dataset for {self.vision_asset}.')

        else:
            print(f'Finished creating TagSLAM data for {self.vision_asset}.')

    def _create_images(self):
        depth_bag_file = file_utils.get_depth_bag_filename(self.rosbag_number)

        # Get the depth offset.
        self.depth_offset_mm = -12
        print(f'NOTE: Using hardcoded {self.depth_offset_mm=} mm.\n')

        # Extract the synchronized RGB and depth images, writing them to
        # bundlenets/data/{vision_asset}/.
        rosbag_processor.extract_synchronized_rgb_and_depth_images(
            start_time=self.start_time, end_time=self.end_time,
            bag_file=depth_bag_file, rgb_output_dir=self.rgb_dir,
            depth_output_dir=self.depth_dir,
            dataset_dir=op.dirname(self.tagslam_dir),
            depth_offset_mm=self.depth_offset_mm
        )

    def _create_tagslam_poses(self):
        odom_bag_file = file_utils.get_odom_bag_filename(self.rosbag_number)
        odom_ros_topic = f"/tagslam/odom/body_{self.object}"

        # Extract the TagSLAM poses, writing them to bundlenets/
        # cnets-data-generation/dataset/{vision_asset}/tagslam_poses/.
        rosbag_processor.extract_tagslam_poses(
            start_time=self.start_time, end_time=self.end_time,
            bag_file=odom_bag_file, odom_topic=odom_ros_topic,
            tagslam_pose_output_dir=self.tagslam_dir
        )

    def _create_synchronized_tagslam_poses(self):
        """Synchronizes the TagSLAM poses at the timestamps of the RGBD images.
        This is done by linearly interpolating the TagSLAM poses to the
        timestamps of the RGBD images, and writing the synchronized poses to
        bundlenets/cnets-data-generation/dataset/{vision_asset}/
        synchronized_tagslam_poses/.  This directory includes XXXX.txt for every
        pose (7, 1), as well as a synced_tagslam.txt file (N, 8) that includes
        timestamps, which should exactly match the timestamps in bundlenets/
        cnets-data-generation/dataset/{vision_asset}/bundlesdf_timestamps.txt.
        """
        # First load the BundleSDF timestamps.
        bsdf_times = np.loadtxt(
            op.join(op.dirname(self.tagslam_dir), 'bundlesdf_timestamps.txt'))

        # Load the TagSLAM data.
        tagslam_data = np.loadtxt(op.join(self.tagslam_dir, 'tagslam.txt'))
        tagslam_times = tagslam_data[:, 0]
        tagslam_poses = tagslam_data[:, 1:]

        # Give a rough idea of how many poses TagSLAM skipped.
        n_tagslam_in_window = np.sum(
            (tagslam_times >= self.start_time.to_sec()) &
            (tagslam_times <= self.end_time.to_sec())
        )
        print(f'\nSynchronizing TagSLAM poses:  going from ' + \
              f'{n_tagslam_in_window} TagSLAM poses to {len(bsdf_times)} ' + \
              f'to synchronize with BundleSDF timestamps.\n')

        # Estimate each pose at the BundleSDF timestamps.
        synced_poses = np.zeros((len(bsdf_times), 7))

        for i, bsdf_time in enumerate(bsdf_times):
            # Find the two TagSLAM times that sandwich the current BSDF time.
            try:
                after_idx = np.where(tagslam_times > bsdf_time)[0][0]
            except IndexError:
                after_idx = len(tagslam_times) - 1
                print(f'bsdf frame {i} is after the last TagSLAM pose.')
            try:
                before_idx = np.where(tagslam_times < bsdf_time)[0][-1]
            except IndexError:
                before_idx = 0
                print(f'bsdf frame {i} is before the first TagSLAM pose.')

            # Lin2early interpolate the TagSLAM pose.
            t1 = tagslam_times[before_idx]
            t2 = tagslam_times[after_idx]
            p1 = tagslam_poses[before_idx]
            p2 = tagslam_poses[after_idx]
            if t1 == t2:
                assert (before_idx == 0 and after_idx == 0) or \
                    (before_idx == len(tagslam_times)-1 and after_idx == len(tagslam_times)-1), \
                    f'Expected {t1=} == {t2=} only at the beginning or end.'
                interpolated_pose = p1
            else:
                assert after_idx-before_idx == 1, f'Expected 1 between ' + \
                    f'{before_idx=} and {after_idx=}.'
                fraction = (bsdf_time - t1) / (t2 - t1)
                interpolated_pose = p1 + fraction*(p2 - p1)

            # Correct the quaternion in case it is no longer unit normal.
            interpolated_pose[3:7] /= np.linalg.norm(interpolated_pose[3:7])

            # Write the synchronized pose to file; use 1-indexing.
            np.savetxt(op.join(self.synced_tagslam_dir, f'{i+1:04d}.txt'),
                       interpolated_pose)

            synced_poses[i] = interpolated_pose

        # Write the synchronized poses to a synced_tagslam.txt file.
        synced_tagslam_data = np.hstack(
            (bsdf_times.reshape(-1, 1), synced_poses))
        filepath = op.join(self.synced_tagslam_dir, 'synced_tagslam.txt')
        np.savetxt(filepath, synced_tagslam_data)

        print(f'Wrote synchronized TagSLAM data to {filepath}.')

    def _create_annotated_poses(self):
        # Create the annotated poses.  These are all the identity 4x4
        # transformation matrix, except for 0000.txt which has the first TagSLAM
        # pose in camera frame.
        rosbag_processor.save_initial_tagslam_pose_in_camera_frame(
            synced_tagslam_world_poses_dir=self.synced_tagslam_dir,
            annotated_poses_dir=self.annotated_poses_dir,
            cam_trans=self.cam_trans, cam_rot_axis_angle=self.cam_axis_vec
        )
        frame_num = len([name for name in os.listdir(self.rgb_dir)])
        for frame_id in range(1, frame_num+1):
            file_utils.create_annotated_poses(
                output_dir=self.annotated_poses_dir, frame_id=frame_id)

    def _compute_table_offset(self):
        # Visualize the depth offset with the ability to make adjustments for
        # future calls to create_dataset.
        if 'cube' in self.vision_asset:
            print(f'Skip visualizing the results of {self.depth_offset_mm=} for ' + \
                f'{self.vision_asset}.')
            # import inspect_camera_alignments
            # inspect_camera_alignments.interactive_offset_adjustment(
            #     self.vision_asset, 1)
        else:
            print(f'No ground truth geometry for {self.vision_asset} so ' + \
                  f'cannot visualize the results of the depth offset.')

        # Lastly, compute the table offset for the experiment.
        os.system('python ' + \
                op.join(file_utils.DATA_GEN_DIR, 'compute_table_offsets.py') + \
                f' single --vision-asset={self.vision_asset} --overwrite ' + \
                f'--redirect-output  --no-visualize')



@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2-3; encodes " + \
                   "system and tosses.")
@click.option('--tagslam-only',
              is_flag=True,
              help="whether to generate just TagSLAM-related data.")
@click.option('--bsdf-only',
              is_flag=True,
              help="whether to generate just Bundlesdf-related data.")
@click.option('--clear-data/--keep-data',
              default=False,
              help="whether to clear data folder before regenerating.")

def main_command(vision_asset: str, tagslam_only: bool, bsdf_only: bool, clear_data: bool):
    # Get the data and pose directories, checking if they already exist.
    data_dir = file_utils.bundlesdf_video_dir(vision_asset, check_exists=False)
    tagslam_dir = file_utils.tagslam_pose_dir(vision_asset, check_exists=False)
    if not tagslam_only and op.exists(data_dir):
        if clear_data:
            print(f'Overwriting existing video data at {data_dir}.')
            os.system(f'rm -r {data_dir}')
        else:
            print(f'Exiting:  Video data already exists at {data_dir} -- use ' + \
                '--clear-data next time.')
            exit()

    if not bsdf_only and op.exists(tagslam_dir):
        if clear_data:
            print(f'Overwriting existing TagSLAM data at {tagslam_dir}.')
            os.system(f'rm -r {tagslam_dir}')
        else:
            print(f'Exiting:  TagSLAM data already exists at {tagslam_dir} ' + \
                f' -- use --clear-data next time.')
            exit()

    # Create the dataset.
    dataset_creator = DatasetCreator(vision_asset, tagslam_only, bsdf_only)
    dataset_creator.create()


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
