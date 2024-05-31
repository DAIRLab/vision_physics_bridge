"""A script to help with adjusting the start frames for dynamics predictions.

Still to be adjusted:                           Not to be adjusted:
 X bakingbox                                     - burger
 X cardboard                                     - chocolate
 - croc                                          - cream
 X crushedcan                                    - duck
 X gallon                                        - hotdog
 X greencan                                      - toothpaste
 X icetray                                       - box
 X mug
 X oatly
 X pinkcan
 X stapler
 X styrofoam
 X cube
 X bottle
 - napkin --> has some trajectory issues to look into
 X toblerone
 X half
 X milk
 X prism
 X egg
"""

import click
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pdb
import torch
from torch import Tensor

import evaluate, eval_utils, file_utils

from evaluate import DynamicsPredictor


START_ADJUST_OPTIONS = [0, 1, 2, 3, 4]
COLOR_BY_LABEL = {'target': 'grey',
                  'start_adjust 0': '#ff0000',
                  'start_adjust 1': '#ff00ff',
                  'start_adjust 2': '#008080',
                  'start_adjust 3': '#008000',
                  'start_adjust 4': '#800080',}


class DynamicsStartAdjuster(DynamicsPredictor):
    def __init__(self, vision_asset: str):
        # Obtain the run history -- use BundleSDF 02 from cycle 2.
        bundlesdf_id = 'bundlesdf_id_02'
        nerf_bundlesdf_id = 'bundlesdf_id_02'
        # bundlesdf_id = 'bundlesdf_id_00'
        # nerf_bundlesdf_id = 'bundlesdf_id_00'
        cycle_iteration = 2
        history = evaluate.traverse_run_history_from_bsdf(
            vision_asset, bundlesdf_id, cycle_iteration)

        # Automatically detect if BundleSDF-only is necessary based on if the
        # object is a tagless one.
        bsdf_only = False
        object = vision_asset.split('_')[0]
        if object in file_utils.TAGLESS_OBJECTS:
            bsdf_only = True
            print(f'Automatically setting {bsdf_only=} for tagless {object=}.')
        else:
            print(f'Using TagSLAM and BundleSDF: {bsdf_only=}')

        super().__init__(
            vision_asset=vision_asset, history=history,
            nerf_bundlesdf_id=nerf_bundlesdf_id, bsdf_only=bsdf_only
        )

    def inspect_start_adjust(self):
        """Make visuals to inspect what start adjust makes sense."""
        # Make the simulation PLL system and get the tracked trajectories
        # (BundleSDF and TagSLAM if available).
        self._create_pll_sim_system()
        self._get_tracked_trajectories()

        # Prepare to keep track of start adjusts.
        start_adjusts_by_toss = {}

        # Get the predictions.
        trajs = self.bundlesdf_trajs #if not hasattr(self, 'tagslam_b_trajs') \
            # else self.tagslam_b_trajs

        for toss_key, target_traj in trajs.items():
            # Grab just the first portion of the trajectory.
            target_traj = target_traj[:10]

            # Try different start adjusts.
            rollouts_by_start_adjust = {}
            for start_adjust in START_ADJUST_OPTIONS:
                rollouts_by_start_adjust[start_adjust] = \
                    eval_utils.get_pll_rollout_trajectory(
                        system=self.pll_system, target_traj=Tensor(target_traj),
                        start_adjust=start_adjust
                    )
                
            # Visualize the different results.
            start_adjusts_by_toss[toss_key] = \
                self.visualize_and_set_start_adjust_rollouts(
                    toss_key, target_traj, rollouts_by_start_adjust)

        # self.start_adjusts_by_toss = start_adjusts_by_toss
        # print(f'Start adjusts by toss:')
        # for k, v in start_adjusts_by_toss.items():
        #     print(f'\tToss {k}: {v}')
            
    def visualize_and_set_start_adjust_rollouts(
            self, toss_key: str, target_traj: Tensor,
            rollouts_by_start_adjust: dict):
        """Visualize the rollouts and set the start adjust."""
        # Make a plot.
        def make_columns_in_row_share_y(ax, row, up_to=4):
            for i in range(up_to):
                ax[row, i].sharey(ax[row, 0])

        fig, ax = plt.subplots(4, 4, figsize=(15, 15), sharex='all',
                               sharey='none')

        make_columns_in_row_share_y(ax, 0, up_to=3)
        make_columns_in_row_share_y(ax, 1)
        ax[1, 0].set_ylim([-1, 1])
        make_columns_in_row_share_y(ax, 2, up_to=3)
        make_columns_in_row_share_y(ax, 3, up_to=3)

        def plot_trajectory(ax, traj, label):
            if type(ax) == np.ndarray:
                # Positions.
                ax[0,0].plot(traj[:, 4], color=COLOR_BY_LABEL[label], label=label)
                ax[0,1].plot(traj[:, 5], color=COLOR_BY_LABEL[label], label=label)
                ax[0,2].plot(traj[:, 6], color=COLOR_BY_LABEL[label], label=label)
                # Quaternions.
                ax[1,0].plot(traj[:, 0], color=COLOR_BY_LABEL[label], label=label)
                ax[1,1].plot(traj[:, 1], color=COLOR_BY_LABEL[label], label=label)
                ax[1,2].plot(traj[:, 2], color=COLOR_BY_LABEL[label], label=label)
                ax[1,3].plot(traj[:, 3], color=COLOR_BY_LABEL[label], label=label)
                # Linear velocities.
                ax[2,0].plot(traj[:, 10], color=COLOR_BY_LABEL[label], label=label)
                ax[2,1].plot(traj[:, 11], color=COLOR_BY_LABEL[label], label=label)
                ax[2,2].plot(traj[:, 12], color=COLOR_BY_LABEL[label], label=label)
                # Angular velocities.
                ax[3,0].plot(traj[:, 7], color=COLOR_BY_LABEL[label], label=label)
                ax[3,1].plot(traj[:, 8], color=COLOR_BY_LABEL[label], label=label)
                ax[3,2].plot(traj[:, 9], color=COLOR_BY_LABEL[label], label=label)
            else:
                ax.plot(traj[:, 6], color=COLOR_BY_LABEL[label], label=label)

        # Plot the trajectories.
        plot_trajectory(ax, target_traj, 'target')
        for i in rollouts_by_start_adjust.keys():
            plot_trajectory(ax, rollouts_by_start_adjust[i], f'start_adjust {i}')

        # Set titles.
        ax[0, 0].set_title('X Position')
        ax[0, 1].set_title('Y Position')
        ax[0, 2].set_title('Z Position')
        ax[1, 0].set_title('W Quaternion')
        ax[1, 1].set_title('X Quaternion')
        ax[1, 2].set_title('Y Quaternion')
        ax[1, 3].set_title('Z Quaternion')
        ax[2, 0].set_title('X Velocity')
        ax[2, 1].set_title('Y Velocity')
        ax[2, 2].set_title('Z Velocity')
        ax[3, 0].set_title('X Angular Velocity')
        ax[3, 1].set_title('Y Angular Velocity')
        ax[3, 2].set_title('Z Angular Velocity')
        
        # More formatting.
        ax[0, 0].set_ylabel('Position [m]')
        ax[1, 0].set_ylabel('Orientation')
        ax[2, 0].set_ylabel('Linear Velocity [m/s]')
        ax[3, 0].set_ylabel('Angular Velocity [rad/s]')
        ax[3, 0].set_xlabel('Time from first pose [s]')
        ax[3, 1].set_xlabel('Time from first pose [s]')
        ax[3, 2].set_xlabel('Time from first pose [s]')
        ax[3, 3].set_xlabel('Time from first pose [s]')

        ax[0, 1].tick_params(labelleft=False)
        ax[0, 2].tick_params(labelleft=False)
        ax[1, 1].tick_params(labelleft=False)
        ax[1, 2].tick_params(labelleft=False)
        ax[1, 3].tick_params(labelleft=False)
        ax[2, 1].tick_params(labelleft=False)
        ax[2, 2].tick_params(labelleft=False)
        ax[2, 3].tick_params(labelleft=False)
        ax[3, 1].tick_params(labelleft=False)
        ax[3, 2].tick_params(labelleft=False)
        ax[3, 3].tick_params(labelleft=False)

        handles, labels = ax[1,2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')

        fig.suptitle(f'Start Adjust for {toss_key}')
        plt.savefig(op.join(
            file_utils.start_adjustment_dir(), f'{self.object}_{toss_key}.png'))
        # plt.show()
        plt.close()

        # Do a close-up of the z plot.
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        plot_trajectory(ax, target_traj, 'target')
        for i in rollouts_by_start_adjust.keys():
            plot_trajectory(ax, rollouts_by_start_adjust[i], f'start_adjust {i}')
        ax.set_ylabel('Position [m]')
        ax.set_title('Z Position')
        fig.suptitle(f'Start Adjust for {toss_key}')
        fig.legend(handles, labels, loc='lower center')
        plt.savefig(op.join(
            file_utils.start_adjustment_dir(), f'{self.object}_{toss_key}_zoom.png'))
        # plt.show()
        plt.close()


        # Set the start adjust.
        # start_adjust = input(f'Select start adjust for {toss_key}: ')
        # while not start_adjust.isdigit():
        #     start_adjust = input(f'\tUse number: ')
        start_adjust = 0

        return int(start_adjust)

#######################################################################
@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--do-videos/--skip-videos',
              type=bool,
              default=False,
              help="whether to generate videos.")

def main_command(vision_asset: str, do_videos: bool):
    assert '_' in vision_asset, f'Invalid {vision_asset=}.'

    start_adjuster = DynamicsStartAdjuster(vision_asset)
    start_adjuster.inspect_start_adjust()


if __name__ == "__main__":
    main_command()  # pylint: disable=no-value-for-parameter
