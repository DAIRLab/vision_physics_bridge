"""Get TagSLAM processed trajectories."""

from typing import Tuple

import click
import numpy as np
import os.path as op
from scipy import signal
from scipy.spatial.transform import Rotation
import torch

import file_utils, math_utils
from conversion_bsdf_to_pll import FILTER_ORIENTATIONS, FILTER_POSITIONS, \
    FILTER_LINEAR_VELOCITIES, FILTER_ANGULAR_VELOCITIES, FILTER_TYPE, \
    SAVGOL_FILTER_WINDOW_LENGTH, SAVGOL_FILTER_POLYORDER, \
    MEDIAN_FILTER_KERNEL_SIZE


class TagSLAMTrajectoryConverter:
    """TODO Something."""

    def __init__(self, vision_asset: str, z_table: float) -> None:
        self.z_table = z_table
        self.tagslam_dir = file_utils.synchronized_tagslam_pose_dir(
            vision_asset, check_exists=True)
        self._load_tagslam_poses()

        # Parse the vision asset.
        start_toss = int(vision_asset.split('_')[-1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
                f'-{end_toss} inferred from {vision_asset=}.'

        self.dataset = vision_asset
        self.start_toss = start_toss
        self.end_toss = end_toss

        self._get_absolute_toss_frames()

    def _get_absolute_toss_frames(self) -> None:
        object = '_'.join(self.dataset.split('_')[:-1])

        relative_start_frames = np.array([file_utils.load_field_from_yaml(
            object, toss_i, 'start_frame') for toss_i in range(
                self.start_toss, self.end_toss+1)])
        relative_end_frames = np.array([file_utils.load_field_from_yaml(
            object, toss_i, 'end_frame') for toss_i in range(
                self.start_toss, self.end_toss+1)])
        
        start_ros_times = np.array([file_utils.load_toss_time_from_yaml(
            object, toss_i, 'start_time', as_ros_time=True) for toss_i in range(
                self.start_toss, self.end_toss+1)])
        
        self.start_frames = math_utils.convert_relative_frames_to_absolute(
            relative_start_frames, self.tagslam_full_times, start_ros_times)
        self.end_frames = math_utils.convert_relative_frames_to_absolute(
            relative_end_frames, self.tagslam_full_times, start_ros_times)

    def _load_tagslam_poses(self) -> None:
        """Load all the poses reported by TagSLAM.  These are in world
        coordinates of the TagSLAM body origin with the following ordering:
            [x, y, z, qx, qy, qz, qw]
        """
        tagslam_data = np.loadtxt(
            op.join(self.tagslam_dir, 'synced_tagslam.txt'))

        self.tagslam_full_times = tagslam_data[:, 0]
        self.tagslam_t_full_poses = tagslam_data[:, 1:]

    def _process_poses(
            self, poses, times, filter_rot=FILTER_ORIENTATIONS,
            filter_pos=FILTER_POSITIONS,
            filter_lin_vel=FILTER_LINEAR_VELOCITIES,
            filter_ang_vel=FILTER_ANGULAR_VELOCITIES,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate contactnets-format trajectory components.  This requires
        estimating velocities from differences in pose and converting everything
        to PLL format.  The PLL format is in:
	        [ quaternion  position  angular_velocity  linear_velocity ]
        where:
            - quaternion: 		[qw, qx, qy, qz]
            - position:  		[x, y, z] in meters
            - angular_velocity:	[wx, wy, wz] in rad/second in body frame
            - linear_velocity: 	[vx, vy, vz] in meters/second

        Args:
            poses (N, 7):  Poses in order of [x, y, z, qx, qy, qz, qw].
            times (N,)

        Outputs:
            qs_wxyz (N, 4)
            ps (N, 3)
            ws (N, 3)
            vs (N, 3)
        """
        # Do some input checking.
        assert times.ndim == 1, f'{times.shape=} not of expected size (N,).'
        assert poses.shape == (times.shape[0], 7), f'{poses.shape=} not of ' + \
            f'expected size ({times.shape[0]}, 7).'

        # Split the poses into positions and quaternions.
        ps = poses[:, :3]
        qs_xyzw = poses[:, 3:7]
        t = times

        # Do orientation filtering if desired.
        if filter_rot:
            qs_xyzw = self._filter_quaternions(qs_xyzw)
        else:
            qs_xyzw = self._make_quaternions_consistent(qs_xyzw)

        # Do position filtering if desired.
        if filter_pos:
            if FILTER_TYPE == 'savgol':
                ps = signal.savgol_filter(
                    ps, window_length=SAVGOL_FILTER_WINDOW_LENGTH,
                    polyorder=SAVGOL_FILTER_POLYORDER, axis=0)
            elif FILTER_TYPE == 'median':
                for i in range(3):
                    ps[:, i] = signal.medfilt(
                        ps[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
            else:
                raise NotImplementedError

        # Subtract out table height so z=0 corresponds to being on the table.
        ps[:, 2] -= self.z_table

        # Calculate derivatives.
        vs = self._estimate_linear_velocities(
            ts=t, ps=ps, filter=filter_lin_vel)
        ws = self._estimate_angular_velocities(
            ts=t, qs=qs_xyzw, filter=filter_ang_vel)

        # Package into PLL format.
        qs_wxyz = math_utils.xyzw2wxyz(qs_xyzw)
        return qs_wxyz, ps, ws, vs

    def _estimate_linear_velocities(
            self, ts, ps, filter=FILTER_LINEAR_VELOCITIES) -> np.ndarray:
        """From times and positions, estimate the linear velocities at each time
        step.

        Args:
            ts (N,)
            ps (N, 3)
            filter:  whether or not to filter the result.

        Outputs:
            vs (N, 3)
        """
        # Do some input checking.
        assert ts.ndim == 1, f'{ts.shape=} not of expected size (N,).'
        assert ps.shape == (ts.shape[0], 3), f'{ps.shape=} not of expected ' + \
            f'size ({ts.shape[0]}, 3).'

        # Compute time differences.
        t_start = ts[0]
        t = ts - t_start
        tdiff = np.tile((t[1:] - t[:-1]).reshape([-1, 1]), [1, 3])

        # Compute velocities based on differences in position over time step.
        pdiff = ps[1:, :] - ps[:-1, :]
        vs = pdiff / tdiff

        # Repeat first row so that \delta p = v' \delta t.
        vs = np.vstack((vs[[0], :], vs))

        if filter:
            if FILTER_TYPE == 'savgol':
                vs = signal.savgol_filter(
                    vs, window_length=SAVGOL_FILTER_WINDOW_LENGTH,
                    polyorder=SAVGOL_FILTER_POLYORDER, axis=0)
            elif FILTER_TYPE == 'median':
                for i in range(3):
                    vs[:, i] = signal.medfilt(
                        vs[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
            else:
                raise NotImplementedError

        return vs

    def _estimate_angular_velocities(
            self, ts, qs, filter=FILTER_ANGULAR_VELOCITIES) -> np.ndarray:
        """From times and orientations, estimate the angular velocities at each
        time step.  These angular velocities are reported in body frame.

        Args:
            ts (N,)
            qs (N, 4)
            filter:  whether or not to filter the result.

        Outputs:
            ws (N, 3)
        """
        # Do some input checking.
        assert ts.ndim == 1, f'{ts.shape=} not of expected size (N,).'
        assert qs.shape == (ts.shape[0], 4), f'{qs.shape=} not of expected ' + \
            f'size ({ts.shape[0]}, 4).'

        # Compute time differences.
        t_start = ts[0]
        t = ts - t_start
        tdiff = np.tile((t[1:] - t[:-1]).reshape([-1, 1]), [1, 3])

        # Compute velocities based on differences in orientation over time step.
        rot_t = Rotation.from_quat(qs)
        rot_rel = rot_t[:-1].inv() * rot_t[1:]
        rel_vecs = rot_rel.as_rotvec()
        ws = rel_vecs / tdiff     # (N, 3)

        # Repeat first row so that \delta q = w' \delta t.
        ws = np.vstack((ws[[0], :], ws))

        if filter:
            if FILTER_TYPE == 'savgol':
                ws = signal.savgol_filter(
                    ws, window_length=SAVGOL_FILTER_WINDOW_LENGTH,
                    polyorder=SAVGOL_FILTER_POLYORDER, axis=0)
            elif FILTER_TYPE == 'median':
                for i in range(3):
                    ws[:, i] = signal.medfilt(
                        ws[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
            else:
                raise NotImplementedError

        return ws

    def _filter_quaternions(self, quat):
        """Filter and process reported quaternions."""
        # Do some input checking.
        assert quat.shape[1] == 4, f'{quat.shape=} not of expected size (N, 4).'
        assert quat.ndim == 2, f'{quat.shape=} not of expected size (N, 4).'

        quat = self._make_quaternions_consistent(quat)
        rot_t = Rotation.from_quat(quat)

        # Fix and filter quaternions.
        rvecs = math_utils.rotvecfix(rot_t.as_rotvec())
        for i in range(3):
            # Always use medfilt no matter FILTER_TYPE.
            rvecs[:, i] = signal.medfilt(
                rvecs[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
        rot_t = rot_t.from_rotvec(rvecs)

        return rot_t.as_quat()

    def _make_quaternions_consistent(self, quat):
        # Do some input checking.
        assert quat.shape[1] == 4, f'{quat.shape=} not of expected size (N, 4).'
        assert quat.ndim == 2, f'{quat.shape=} not of expected size (N, 4).'

        return math_utils.fix_quaternions(quat)

    def _trim_processed_trajectories(self) -> None:
        self.tagslam_t_toss_processed_states = []
        self.tagslam_toss_times = []

        for i in range(len(self.start_frames)):
            # Need to do one less than provided start and end frames because
            # the annotated start_frame and end_frame in config.yaml were
            # annotated based on the 1-indexed images, but the below needs to
            # use 0-indexing.
            t_start = self.start_frames[i] - 1
            t_end = self.end_frames[i] - 1

            self.tagslam_t_toss_processed_states.append(
                self.tagslam_t_full_processed_states[t_start:t_end])
            self.tagslam_toss_times.append(
                self.tagslam_full_times[t_start:t_end])

    def process_and_save(self) -> None:
        q_ts, p_ts, w_ts, v_ts = self._process_poses(
            self.tagslam_t_full_poses, self.tagslam_full_times)
        self.tagslam_t_full_processed_states = np.concatenate(
            (q_ts, p_ts, w_ts, v_ts), axis=1)

        # Compute trimmed trajectories for toss only.
        self._trim_processed_trajectories()

        # Store all of the results to the TagSLAM T trajectory directory.
        tagslam_t_traj_dir = file_utils.synchronized_tagslam_t_state_dir(
            dataset=self.dataset, create=True)

        # Full trajectory.
        torch.save(
            torch.tensor(self.tagslam_t_full_processed_states),
            op.join(tagslam_t_traj_dir, 'tagslam_t.pt'))
        print(f"\t{op.join(tagslam_t_traj_dir, 'tagslam_t.pt')}")

        # Do toss trajectories.
        toss_filenames = [f'{toss_i}.pt' for toss_i in range(
            self.start_toss, self.end_toss+1)]
        for i in range(len(toss_filenames)):
            torch.save(
                torch.tensor(self.tagslam_t_toss_processed_states[i]),
                op.join(tagslam_t_traj_dir, toss_filenames[i]))
            print(f'\t{op.join(tagslam_t_traj_dir, toss_filenames[i])}')



#######################################################################
@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")

def main_command(vision_asset: str):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    object = '_'.join(vision_asset.split('_')[:-1])

    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
            f'-{end_toss} inferred from {vision_asset=}.'
    
    # Get the table height.  Use the average if using multiple tosses.
    table_heights = np.array([
        file_utils.load_table_z_height(object, toss) for toss in
        range(start_toss, end_toss+1)
    ])
    z_table = np.mean(table_heights)

    # Get the TagSLAM trajectory converter.
    converter = TagSLAMTrajectoryConverter(vision_asset, z_table)
    converter.process_and_save()


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
