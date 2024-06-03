"""Test script to inspect the trajectories in PLL assets."""

import matplotlib.pyplot as plt
import os.path as op
import pdb
import torch


import eval_utils, file_utils, math_utils


ASSET = 'milk_1'
BSDF_ID = 'bundlesdf_id_00'

# First load the TagSLAM and BundleSDF data.
tagslam_full_traj = torch.load(op.join(
        file_utils.contactnets_input_dir_tagslam(
            dataset=ASSET, full=True, create=False),
        'tagslam.pt'))
bundlesdf_full_traj = torch.load(op.join(
    file_utils.contactnets_input_dir_bundlesdf(
        dataset=ASSET, iteration=1, bundlesdf_id=BSDF_ID, full=True,
        create=False),
    f'{BSDF_ID}.pt'))


# Transform TagSLAM to match BundleSDF origin.
b_trans_mat, t_trans_mat = eval_utils.get_synced_bsdf_tagslam_toss_poses(
    vision_asset=ASSET, bundlesdf_id=BSDF_ID, cycle_iteration=1,
    desired_toss_num=0)
tagslam_b_traj = math_utils.transform_t_origin_to_b_origin_pll_format(
    full_tagslam_trajectory=tagslam_full_traj, synced_bsdf_pose=b_trans_mat,
    synced_tagslam_pose=t_trans_mat)

# Straighten out the quaternion.
tagslam_full_traj = tagslam_full_traj.numpy()
bundlesdf_full_traj = bundlesdf_full_traj.numpy()
# tagslam_b_traj = tagslam_b_traj.numpy()  # Already numpy.

tagslam_full_traj[:, :4] = math_utils.fix_quaternions_wxyz(
    tagslam_full_traj[:, :4])
bundlesdf_full_traj[:, :4] = math_utils.fix_quaternions_wxyz(
    bundlesdf_full_traj[:, :4])
tagslam_b_traj[:, :4] = math_utils.fix_quaternions_wxyz(tagslam_b_traj[:, :4])


# Plot it.
plt.ion()
fig, ax = plt.subplots(2, 4, figsize=(15, 15), sharex='all', sharey='row')
for i in range(4):
    ax[0, i].plot(tagslam_full_traj[:, i], label='TagSLAM')
    ax[0, i].plot(bundlesdf_full_traj[:, i], label='BundleSDF')
    ax[0, i].plot(tagslam_b_traj[:, i], label='TagSLAM, B')

    if i < 3:
        ax[1, i].plot(tagslam_full_traj[:, i+4], label='TagSLAM')
        ax[1, i].plot(bundlesdf_full_traj[:, i+4], label='BundleSDF')
        ax[1, i].plot(tagslam_b_traj[:, i+4], label='TagSLAM, B')

ax[0, 3].legend()
ax[0, 0].set_title('qw')
ax[0, 1].set_title('qx')
ax[0, 2].set_title('qy')
ax[0, 3].set_title('qz')
ax[1, 0].set_title('x')
ax[1, 1].set_title('y')
ax[1, 2].set_title('z')


pdb.set_trace()


