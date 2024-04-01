""""""
import click
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pdb
import sys
import yaml

from scipy.optimize import minimize_scalar
from typing import Tuple

import file_utils, math_utils, rosbag_processor


X_LIMS = [-0.3, 0.9]
Y_LIMS = [-0.6, 0.6]
Z_LIMS = [-0.3, 0.9]

TIGHT_X_LIMS = [0.1, 0.9]
TIGHT_Y_LIMS = [-0.5, 0.5]
TIGHT_Z_LIMS = [-0.1, 0.1]

# Give tighter bounds for the possible table height.  If allowed to be
# TIGHT_Z_LIMS, then often the optimization problem that searches for the z
# height that maximizes the number of points within epsilon of it gets stuck in
# a local minimum that is much too high.
TABLE_Z_LIMS = [-0.08, 0.02]

EPS_LEFT = 1e-3
EPS_RIGHT = 1e-2
TOLERANCE = 1e-5
PATIENCE = 200

DESIRED_RETENTION = 0.5


def get_table_height_from_log(log_file: str) -> float:
    """From a recorded log file, extract the optimized table height."""
    # Handle case where the log doesn't exist.
    if not op.exists(log_file):
        print(f'Did not find {log_file} -- Skipping.')
        return None

    with open(log_file, 'r') as f:
        lines = f.readlines()
    last_lines = lines[-4:]

    # Handle the case where the log exists but it didn't compute table height.
    # --> there won't be lines with "Saved plot to ...".
    if 'Saved plot to' not in last_lines[-1]:
        print(f'Found {log_file} but did not report table height -- Skipping.')
        return None

    # Handle the case where the optimization hit iteration limit.
    # --> there will be a "No progress with ..." line; use line above.
    if 'No progress with' in last_lines[-3]:
        line_i = -4

        # This line may not have the selected table height.  Check based on if
        # this line's retention percentage is above the desired retention rate.
        retention_percent = float(lines[line_i].split('(')[1].split('%)')[0])
        while retention_percent < DESIRED_RETENTION*100:
            # TODO pick a different line
            line_i -= 1
            assert len(lines) + line_i >= 0, f'Could not find retention ' + \
                f'rate in {log_file} above threshold {DESIRED_RETENTION} (' + \
                f'{line_i=}, {len(lines)=}).'
            retention_percent = float(
                lines[line_i].split('(')[1].split('%)')[0])

        line = lines[line_i]

    # Handle the case where the optimization satisfied tolerance.
    # --> there will be two "Saved plot to ..." lines; use line above.
    else:
        line = last_lines[-3]

    assert 'Found optimal z height' in line, f'Thought line that would ' + \
        f'report optimal z height would be {line} but did not find it.'

    height_str = line.split('z height: ')[1].split(' m with ')[0]
    return float(height_str)


class ROSBagDepthPlaneViewer:
    """Compute the table height for a single toss."""
    def __init__(self, vision_asset: str) -> None:
        self.vision_asset = vision_asset
        object = vision_asset.split('_')[:-1]
        object = object[0] if len(object) == 1 else f'{object[0]}_{object[1]}'

        start_toss = int(vision_asset.split('_')[-1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
            f'-{end_toss} inferred from {vision_asset=}.'
        
        # Locate all the related files and directories for the given vision
        # asset.
        rosbag_num = file_utils.load_rosbag_number_from_yaml(
            object, start_toss, second_toss_number=end_toss)
        try:
            self.depth_bag_file = file_utils.get_depth_bag_filename(rosbag_num)
        except AssertionError as e:
            print(e)
            self.success = False
            return

        print(f'Processing toss {vision_asset} in raw_{rosbag_num}.bag.\n')

        # Get the camera extrinsics, which are stored for every object.
        self.cam_p, self.cam_R_axis_angle = \
            file_utils.load_camera_extrinsics(object)
        self.cam_p = self.cam_p.squeeze()
        self.cam_R_axis_angle = self.cam_R_axis_angle.squeeze()
    
        # Start/end times are for the start and end of a BundleSDF trajectory,
        # which starts with the object unmoving on the table, includes the toss
        # wind-up and execution, and ends with the object unmoving on the table
        # again.
        self.start_ros_times = np.array([file_utils.load_toss_time_from_yaml(
            object, toss_i, 'start_time', as_ros_time=True) for toss_i in range(
                start_toss, end_toss+1)])
        self.end_ros_times = np.array([file_utils.load_toss_time_from_yaml(
            object, toss_i, 'end_time', as_ros_time=True) for toss_i in range(
                start_toss, end_toss+1)])
        
        self.times, self.raw_images = rosbag_processor.extract_depth_images(
            self.start_ros_times[0], self.end_ros_times[0], self.depth_bag_file)

        self.success = True

    def convert_depth_image_to_point_cloud(self) -> None:
        """Given the depth images stored at self.raw_images, convert them to
        point clouds in meters in the world frame."""
        fx, fy, cx, cy = file_utils.load_camera_intrinsics()

        for depth_image in self.raw_images[:1]:
            # Get height and width of the depth image.
            height, width = depth_image.shape

            # Generate pixel grid.
            x = np.arange(0, width)
            y = np.arange(0, height)
            xv, yv = np.meshgrid(x, y)

            # Calculate corresponding 3D coordinates.
            X = (xv - cx) * depth_image / fx
            Y = (yv - cy) * depth_image / fy
            Z = depth_image

            # Stack the coordinates and reshape.
            point_cloud = np.stack((X, Y, Z), axis=-1)
            point_cloud = point_cloud.reshape((-1, 3))

            # Filter out the non-returns and convert millimeters to meters.
            point_cloud = point_cloud[np.any(point_cloud != 0, axis=1)] / 1000.0

            # Convert camera frame to world frame.
            theta = np.linalg.norm(self.cam_R_axis_angle)
            R_WC = math_utils.axis_angle_to_rotation_matrix(
                self.cam_R_axis_angle, theta)
            self.point_cloud = (point_cloud @ R_WC.T) + self.cam_p

    def compute_epsilon_and_table_offset(
            self, show: bool = True, save: bool = False) -> None:
        """From the stored point cloud, zoom in on the table area and find the
        tightest epsilon value that retains at least a pre-specified fraction of
        the points as residing within epsilon of an optimized table height."""
        # Crop the point cloud to the relevant robot workspace.
        pc_workspace = self.point_cloud[
            (self.point_cloud[:, 0] > X_LIMS[0]) & \
            (self.point_cloud[:, 0] < X_LIMS[1]) & \
            (self.point_cloud[:, 1] > Y_LIMS[0]) & \
            (self.point_cloud[:, 1] < Y_LIMS[1]) & \
            (self.point_cloud[:, 2] > Z_LIMS[0]) & \
            (self.point_cloud[:, 2] < Z_LIMS[1])
        ]

        # Get a tighter crop that focuses on the table.
        pc_surface = pc_workspace[
            (pc_workspace[:, 0] > TIGHT_X_LIMS[0]) & \
            (pc_workspace[:, 0] < TIGHT_X_LIMS[1]) & \
            (pc_workspace[:, 1] > TIGHT_Y_LIMS[0]) & \
            (pc_workspace[:, 1] < TIGHT_Y_LIMS[1]) & \
            (pc_workspace[:, 2] > TIGHT_Z_LIMS[0]) & \
            (pc_workspace[:, 2] < TIGHT_Z_LIMS[1])
        ]
    
        # Compute the tightest epsilon that finds a table height that retains at
        # least the desired retention rate of the points in the table crop.
        self._find_tightest_eps_with_fraction_points_at_table_height(
            pc_surface, show=show, save=save)

        # Store the cropped versions of the point cloud.
        self.pc_workspace = pc_workspace
        self.pc_surface = pc_surface

    def _find_tightest_eps_with_fraction_points_at_table_height(
            self, cropped_point_cloud: np.ndarray, show: bool = True,
            save: bool = False
    ) -> Tuple[float, float]:
        """Find the smallest epsilon value such that the table-cropped point
        cloud still contains at least the desired fraction of the points at the
        (epsilon-dependent) table height.  To do this, implement a bisection
        algorithm on the function of epsilon that returns the number of points
        at the table height, finding the epsilon value at which the number is
        a desired fraction of the total.  Stop when reach a tolerance.  Store
        the epsilon and corresponding table height as class attributes."""
        n_total = cropped_point_cloud.shape[0]
        
        # Start with really wide bounds.
        eps_left = EPS_LEFT
        eps_right = EPS_RIGHT

        z_left, n_left = self._find_optimal_z_height_for_epsilon(
            cropped_point_cloud, eps_left, verbose=False)
        z_right, n_right = self._find_optimal_z_height_for_epsilon(
            cropped_point_cloud, eps_right, verbose=False)

        # Don't need to check the left side because the bounded optimization
        # problem will just choose eps_left (the smallest epsilon) if the
        # fraction of included points is already greater than the desired
        # retention there.
        # assert n_left/n_total < DESIRED_RETENTION, f'Found > ' + \
        #     f'{DESIRED_RETENTION} of the points at {eps_left=} (' + \
        #     f'{n_left/n_total}).'

        # Do check the right side, because we do want at least the retention
        # rate within the bounds.
        assert n_right/n_total > DESIRED_RETENTION, f'Found < ' + \
            f'{DESIRED_RETENTION} of the points at {eps_right=} (' + \
            f'{n_right/n_total}).'
        
        epsilons = [eps_left, eps_right]
        fractions = [n_left/n_total, n_right/n_total]
        heights = [z_left, z_right]
        
        # Create a plot that will automatically update with every iteration.
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        ax1 = fig.add_subplot(211)
        ax1.set_xlabel('Epsilon (m)')
        ax1.set_ylabel('Fraction of points at table height')
        ax1.set_xlim([eps_left, eps_right])
        ax1.set_xscale('log')
        ax1.set_ylim([0.0, 1.0])
        ax1.plot([eps_left, eps_right], [DESIRED_RETENTION, DESIRED_RETENTION],
                'r--', label='Desired retention rate')
        fraction_line = ax1.plot(epsilons, fractions, 'co', label='Samples')[0]
        ax1.legend()
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.set_xlabel('Epsilon (m)')
        ax2.set_ylabel('Z Height (m)')
        ax2.set_xlim([eps_left, eps_right])
        ax2.set_xscale('log')
        ax2.set_ylim(TABLE_Z_LIMS)
        height_line = ax2.plot(epsilons, heights, 'co', label='Samples')[0]
        ax2.legend()
        
        # Start logging progress.
        loop = 0
        no_progress = 0
        print(f'[{loop:02d}] ', end='')

        # Calculate the fraction of points at table height at the midpoint of
        # epsilon.
        eps_mid = math_utils.log_mean(eps_left, eps_right)
        z_mid, n_mid = self._find_optimal_z_height_for_epsilon(
            cropped_point_cloud, eps_mid)
        fraction = n_mid / n_total

        while np.abs(fraction - DESIRED_RETENTION) > TOLERANCE:
            loop += 1
            print(f'[{loop:02d}] ', end='')

            n_last = n_mid

            # Cut the bounds in half, choosing the half that straddles the
            # epsilon sweet spot.
            if fraction > 0.5:
                eps_right = eps_mid
            else:
                eps_left = eps_mid
            eps_mid = math_utils.log_mean(eps_left, eps_right)

            # Re-evaluate the fraction of points at the new midpoint.
            z_mid, n_mid = self._find_optimal_z_height_for_epsilon(
                cropped_point_cloud, eps_mid)
            fraction = n_mid / n_total

            # Update progress plot.
            epsilons.append(eps_mid)
            fractions.append(fraction)
            heights.append(z_mid)
            fraction_line.set_data(epsilons, fractions)
            height_line.set_data(epsilons, heights)

            # Check for progress, exiting if stuck.
            if n_mid == n_last:
                no_progress += 1
                if no_progress > PATIENCE:
                    # Choose the smallest epsilon that got more than the desired
                    # fraction of points at the table height.
                    eps_mid = np.min(np.array(epsilons)[
                        np.array(fractions) > DESIRED_RETENTION
                    ])
                    z_mid, n_mid = self._find_optimal_z_height_for_epsilon(
                        cropped_point_cloud, eps_mid, verbose=False)

                    print(f'No progress with {PATIENCE=} iterations.  Exiting.')
                    break
            else:
                no_progress = 0
        
        self.epsilon = eps_mid
        self.z_table = z_mid

        ax1.plot([self.epsilon], [n_mid/n_total], 'm*', label='Selected',
                 markersize=10)
        ax2.plot([self.epsilon], [self.z_table], 'm*', label='Selected',
                 markersize=10)
        ax1.legend()
        ax2.legend()

        if save:
            file = file_utils.point_cloud_processing_plot_filepath(
                self.vision_asset, eps=True)
            plt.savefig(file)
            print(f'Saved plot to {file}.')
        if show:
            plt.show()
        plt.close()

    def plot_point_cloud(self, show: bool = True, save: bool = False) -> None:
        """Plot the provided depth image and the processing."""
        # Color the point cloud differently for points determined to be near the
        # optimal table height.
        pc_table = self.pc_workspace[
            (self.pc_workspace[:, 2] > self.z_table - self.epsilon) & \
            (self.pc_workspace[:, 2] < self.z_table + self.epsilon)
        ]
        pc_other = self.pc_workspace[
            (self.pc_workspace[:, 2] < self.z_table - self.epsilon) | \
            (self.pc_workspace[:, 2] > self.z_table + self.epsilon)
        ]

        pc_table_t = self.pc_surface[
            (self.pc_surface[:, 2] > self.z_table - self.epsilon) & \
            (self.pc_surface[:, 2] < self.z_table + self.epsilon)
        ]
        pc_other_t = self.pc_surface[
            (self.pc_surface[:, 2] < self.z_table - self.epsilon) | \
            (self.pc_surface[:, 2] > self.z_table + self.epsilon)
        ]

        # Compute retention rate.
        retention_rate = pc_table_t.shape[0]/self.pc_surface.shape[0]

        # Make the plots:  one with the workspace view, the other with the
        # tighter crop to show the retention rate.
        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(pc_other[:, 0], pc_other[:, 1], pc_other[:, 2],
                   s=0.1, label='Outside')
        ax.scatter(pc_table[:, 0], pc_table[:, 1], pc_table[:, 2],
                   s=0.1, label='Within')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_xlim(X_LIMS)
        ax.set_ylim(Y_LIMS)
        ax.set_zlim(Z_LIMS)
        ax.legend()
        ax.set_title(f'z_table = {self.z_table:.6f}m for max points within ' + \
                     f'+/- {self.epsilon:.6f}m ({retention_rate*100.0:.2f}%' + \
                     f' retention).')
        ax = fig.add_subplot(212, projection='3d')
        ax.scatter(pc_other_t[:, 0], pc_other_t[:, 1], pc_other_t[:, 2],
                   s=0.1, label='Outside')
        ax.scatter(pc_table_t[:, 0], pc_table_t[:, 1], pc_table_t[:, 2],
                   s=0.1, label='Within')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_xlim(TIGHT_X_LIMS)
        ax.set_ylim(TIGHT_Y_LIMS)
        ax.set_zlim(TIGHT_Z_LIMS)
        ax.legend()
        ax.set_title(f'z_table = {self.z_table:.6f}m for max points within ' + \
                     f'+/- {self.epsilon:.6f}m ({retention_rate*100.0:.2f}%' + \
                     f' retention).')
        
        fig.suptitle(f'Point cloud for {self.vision_asset}.')
        fig.set_figwidth(10)
        fig.set_figheight(16)

        if save:
            file = file_utils.point_cloud_processing_plot_filepath(
                self.vision_asset)
            plt.savefig(file)
            print(f'Saved plot to {file}.')
        if show:
            plt.show()
        plt.close()
        
    def _find_optimal_z_height_for_epsilon(
            self, cropped_point_cloud: np.ndarray, epsilon: float,
            verbose: bool = True
        ) -> Tuple[float, int]:
        """Find the z height of the table that maximizes the number of points in
        the cropped point cloud that are within EPS of the z height.  Hopefully
        this corresponds to the table's height, but this can be inspected."""
        def z_height_objective_function(z_height, point_cloud, epsilon):
            # Count points within epsilon range around z_height.
            point_count = np.sum(np.abs(point_cloud[:, 2]-z_height) <= epsilon)

            # Return negative count to maximize.
            return -point_count

        # Find z height that maximizes the objective function.
        result = minimize_scalar(
            z_height_objective_function,
            args=(cropped_point_cloud, epsilon),
            bounds=TABLE_Z_LIMS,
            method='bounded',
            # options={'disp': 3}
        )

        optimal_z_height = result.x
        number_of_points = -result.fun
        retention_rate = number_of_points/cropped_point_cloud.shape[0]

        if verbose:
            print(f'Found optimal z height: {optimal_z_height:.5f} m with ' + \
                f'{number_of_points} points ({retention_rate*100.0:.5f}%)' + \
                f' within {epsilon:.8f} m of it.')

        return optimal_z_height, number_of_points
        


#######################################################################
@click.group()
def cli():
    pass


# Use 'single' command to process a single vision asset.
@cli.command('single')
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--visualize/--no-visualize',
              type=bool,
              default=True,
              help="whether to visualize the point cloud processing.")
def process_single_command(vision_asset: str, visualize: bool):
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    
    depth_plane_viewer = ROSBagDepthPlaneViewer(vision_asset)
    if depth_plane_viewer.success:
        depth_plane_viewer.convert_depth_image_to_point_cloud()
        depth_plane_viewer.compute_epsilon_and_table_offset(
            save=True, show=visualize)
        depth_plane_viewer.plot_point_cloud(save=True, show=visualize)


# Use 'all' command to process all tosses found in config.yaml.
@cli.command('all')
@click.option('--visualize/--no-visualize',
              type=bool,
              default=True,
              help="whether to visualize the point cloud processing.")
@click.option('--overwrite/--keep-data',
              type=bool,
              default=False,
              help="whether to overwrite or keep previously generated results")
def process_all_command(visualize: bool, overwrite: bool):
    # Get all the vision assets from the config.yaml file.
    vision_assets = []
    objects = file_utils.load_toss_objects_from_yaml()
    for obj in objects:
        tosses = file_utils.load_toss_numbers_from_object_in_yaml(obj)
        for toss in tosses:
            vision_assets.append(f'{obj}_{toss}')

    # Process each vision asset, logging the results to log files.
    for vision_asset in vision_assets:
        log_file = file_utils.point_cloud_processing_log_filepath(vision_asset)
        if op.exists(log_file) and not overwrite:
            plot_file = file_utils.point_cloud_processing_plot_filepath(
                vision_asset)
            if op.exists(plot_file):
                print(f'Skipping {vision_asset} since found prior results.')
                continue
            else:
                print(f'Looking for bag for: ', end='')

        print(f'Processing {vision_asset} --> {log_file}')
        with open(log_file, 'w') as f:
            sys.stdout = f
            depth_plane_viewer = ROSBagDepthPlaneViewer(vision_asset)
            if depth_plane_viewer.success:
                depth_plane_viewer.convert_depth_image_to_point_cloud()
                depth_plane_viewer.compute_epsilon_and_table_offset(
                    save=True, show=visualize)
                depth_plane_viewer.plot_point_cloud(save=True, show=visualize)
        sys.stdout = sys.__stdout__


# Use 'combine' command to read all the previously-generated results and write
# them to a yaml file.
@cli.command('combine')
@click.option('--overwrite/--keep-data')
def combine_command(overwrite: bool):
    # Check if the yaml file has already been written before.
    calibration_yaml_file = file_utils.table_calibration_yaml_filepath()
    if op.exists(calibration_yaml_file) and not overwrite:
        print(f'Already found {calibration_yaml_file} -- use --overwrite ' + \
              f'next time if you want to overwrite.')
        return

    # Build the dictionary.
    object_toss_height_dict = {}
    objects = file_utils.load_toss_objects_from_yaml()
    for obj in objects:
        object_toss_height_dict[obj] = {}
        tosses = file_utils.load_toss_numbers_from_object_in_yaml(obj)
        for toss in tosses:
            vision_asset = f'{obj}_{toss}'
            log_file = file_utils.point_cloud_processing_log_filepath(
                vision_asset)
            object_toss_height_dict[obj][toss] = get_table_height_from_log(
                log_file)

            print(f'{vision_asset}: {object_toss_height_dict[obj][toss]}')

    with open(calibration_yaml_file, 'w') as f:
        yaml.dump(object_toss_height_dict, f)

    print(f'Conglomerated results into yaml {calibration_yaml_file}.')



if __name__ == '__main__':
    cli()
