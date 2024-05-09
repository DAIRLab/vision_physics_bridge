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
    - tagslam_poses/
        - XXXX.txt, 1 through N
        - tagslam.txt
        - All of the poses are recorded as 7 elements, in order [x, y, z, qx,
          qy, qz, qw].
        - The tagslam.txt file contains all the other XXXX.txt file contents
          plus timestamps at the front:  [t, x, y, z, qx, qy, qz, qw].
    - bundlesdf_timestamps.txt
        - The timestamps associated with the images in the BundlesDF dataset.
    - TODO: Figure out if the other subfolders are required.
"""

import click
import os
import os.path as op
import pdb

import file_utils
import rosbag_processor



@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2-3; encodes " + \
                   "system and tosses.")
@click.option('--clear-data/--keep-data',
              default=False,
              help="whether to clear data folder before regenerating.")

def main_command(vision_asset: str, clear_data: bool):
    # First parse the system and start/end tosses from the provided asset name.
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    # want to split {xxx_xxx_..._xxx}_{x-x} into object={xxx_xxx_..._xxx} and {x-x}
    # need to only split on the last underscore
    object = vision_asset.split('_')[:-1]
    object = '_'.join(object)

    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
        f'-{end_toss} inferred from {vision_asset=}.'

    # Get the ROS bag, ensuring the start toss and end tosses are in the same
    # bag.
    rosbag_number = file_utils.load_rosbag_number_from_yaml(
        object, start_toss, second_toss_number=end_toss)
    depth_bag_file = file_utils.get_depth_bag_filename(rosbag_number)
    odom_bag_file = file_utils.get_odom_bag_filename(rosbag_number)
    odom_ros_topic = f"/tagslam/odom/body_{object}"

    # Get the data and pose directories, checking if they already exist.
    data_dir = file_utils.bundlesdf_video_dir(vision_asset, check_exists=False)
    tagslam_dir = file_utils.tagslam_pose_dir(vision_asset, check_exists=False)
    if op.exists(data_dir) or op.exists(tagslam_dir):
        if clear_data:
            print(f'Overwriting existing data at {data_dir} and/or ' + \
                  f'{tagslam_dir}.')
            if op.exists(data_dir):  os.system(f'rm -r {data_dir}')
            if op.exists(tagslam_dir):  os.system(f'rm -r {tagslam_dir}')
        else:
            print(f'Exiting:  Data already exists at {data_dir} and/or ' + \
                  f'{tagslam_dir} -- use --clear-data next time.')
            exit()
    file_utils.assure_created(data_dir)
    file_utils.assure_created(tagslam_dir)

    print(f'Processing {vision_asset} in ROS bag {rosbag_number}.\n')

    # Get the remaining information to do the processing:  directories, times,
    # camera extrinsics.
    rgb_dir = file_utils.bundlesdf_video_rgb_dir(vision_asset)
    depth_dir = file_utils.bundlesdf_video_depth_dir(vision_asset)
    annotated_poses_dir = file_utils.bundlesdf_annotated_poses_dir(
        vision_asset, create=True)
    start_time = file_utils.load_toss_time_from_yaml(
        object, start_toss, 'start_time', as_ros_time=True)
    end_time = file_utils.load_toss_time_from_yaml(
        object, end_toss, 'end_time', as_ros_time=True)
    cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(object)

    # Get the depth offset.
    depth_offset_mm = -12
    print(f'NOTE: Using hardcoded depth offset of {depth_offset_mm} mm.\n')

    # Extract the synchronized images and TagSLAM poses, writing them to
    # bundlenets/data/{vision_asset}/ and bundlenets/cnets-data-generation/
    # dataset/{vision_asset}/tagslam_poses/.
    rosbag_processor.extract_synchronized_images_and_tagslam_poses(
        start_time=start_time, end_time=end_time, depth_bag_file=depth_bag_file,
        odom_bag_file=odom_bag_file, odom_topic=odom_ros_topic,
        tagslam_pose_output_dir=tagslam_dir, rgb_output_dir=rgb_dir,
        depth_output_dir=depth_dir, depth_offset_mm=depth_offset_mm
    )

    # Copy the camera intrinsics.
    os.system(f'cp {file_utils.get_camera_intrinsics_filepath()} ' + \
              f'{op.join(data_dir, "cam_K.txt")}')

    # Create the annotated poses.  These are all the identity 4x4 transformation
    # matrix, except for 0000.txt which has the first TagSLAM pose in camera
    # frame.
    rosbag_processor.save_initial_tagslam_pose_in_camera_frame(
        tagslam_world_poses_dir=tagslam_dir,
        annotated_poses_dir=annotated_poses_dir, cam_trans=cam_trans,
        cam_rot_axis_angle=cam_rot_axis_angle
    )
    frame_num = len([name for name in os.listdir(rgb_dir)])
    for frame_id in range(1, frame_num+1):
        file_utils.create_annotated_poses(
            output_dir=annotated_poses_dir, frame_id=frame_id)

    print(f'Finished creating dataset for {vision_asset}.')

    # Visualize the depth offset with the ability to make adjustments for future
    # calls to create_dataset.
    if 'cube' in vision_asset:
        print(f'Visualizing the results of depth offset = {depth_offset_mm}' + \
              f' for {vision_asset}.')
        import inspect_camera_alignments
        inspect_camera_alignments.interactive_offset_adjustment(vision_asset, 1)
    else:
        print(f'No ground truth geometry for {vision_asset} so cannot ' + \
              'visualize the results of the depth offset.')

    # Lastly, compute the table offset for the experiment.
    os.system('python ' + \
              op.join(file_utils.DATA_GEN_DIR, 'compute_table_offsets.py') + \
              f' single --vision-asset={vision_asset} --overwrite ' + \
              f'--redirect-output')


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
