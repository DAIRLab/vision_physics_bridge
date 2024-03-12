import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from math_utils import transform_bundletrack_origin_to_tagslam_origin
import torch
import yaml

"""Not working.
"""

def matrix_to_trajectory(matrices, dt):
    N = len(matrices)
    trajectory = np.zeros((N, 13))

    # 1. Extract rotation and translation components
    rotations = np.array([m[:3, :3] for m in matrices])
    positions = np.array([m[:3, 3] for m in matrices])

    # 2. Convert rotation matrices to quaternions
    quats = [Rotation.from_matrix(rot).as_quat() for rot in rotations]
    # Convert quaternion format from xyzw to wxyz
    quats = np.array([q[[3, 0, 1, 2]] for q in quats])

    # 3. Compute linear velocities
    diff_positions = np.diff(positions, axis=0)
    first_diff = positions[0].reshape(1, 3)
    linear_velocities = np.vstack((first_diff, diff_positions)) / dt

    # 4. Compute angular velocities in body frame
    angular_velocities = []
    for i in range(N-1):
        rel_rot = Rotation.from_matrix(rotations[i+1]) * Rotation.from_matrix(rotations[i]).inv()
        angular_velocity_world = rel_rot.as_rotvec() / dt
        angular_velocity_body = rotations[i].T @ angular_velocity_world
        angular_velocities.append(angular_velocity_body)
    angular_velocities.append(angular_velocities[-1])
    angular_velocities = np.array(angular_velocities)
    
    trajectory[:, :4] = np.array(quats)
    trajectory[:, 4:7] = np.array(positions)
    trajectory[:, 7:10] = np.array(angular_velocities)
    trajectory[:, 10:] = np.array(linear_velocities)
    print(positions.shape, quats.shape, linear_velocities.shape, angular_velocities.shape)
    visualize_trajectory(positions, quats, linear_velocities, angular_velocities)
    torch.save(torch.tensor(trajectory), CONTACTNETS_INPUT_DIR + "{}.pt".format(toss_id))
    print(f'{toss_id}.pt saved to {CONTACTNETS_INPUT_DIR}')
    return trajectory

def visualize_trajectory(p_t, q_t, dp_t, w_t_body):
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))

    # Plot positions
    ax[0, 0].plot(p_t[:, 0])
    ax[0, 0].set_title('X Position')
    ax[0, 1].plot(p_t[:, 1])
    ax[0, 1].set_title('Y Position')
    ax[0, 2].plot(p_t[:, 2])
    ax[0, 2].set_title('Z Position')

    # Plot Quaternion components
    ax[1, 0].plot(q_t[:, 0])
    ax[1, 0].set_title('Quaternion w')
    ax[1, 1].plot(q_t[:, 1])
    ax[1, 1].set_title('Quaternion x')
    ax[1, 2].plot(q_t[:, 2])
    ax[1, 2].set_title('Quaternion y')
    ax[2, 0].plot(q_t[:, 3])
    ax[2, 0].set_title('Quaternion z')

    # Plot angular velocities
    ax[2, 1].plot(w_t_body[:, 0])
    ax[2, 1].set_title('Angular Velocity X')
    ax[2, 2].plot(w_t_body[:, 1])
    ax[2, 2].set_title('Angular Velocity Y')
    ax[3, 0].plot(w_t_body[:, 2])
    ax[3, 0].set_title('Angular Velocity Z')

    # Plot linear velocities
    ax[3, 1].plot(dp_t[:, 0])
    ax[3, 1].set_title('Linear Velocity X')
    ax[3, 2].plot(dp_t[:, 1])
    ax[3, 2].set_title('Linear Velocity Y')

    plt.tight_layout()
    fig.suptitle('ContactNets cube trajectory')
    fig_name = "./test.png"
    plt.savefig(fig_name)
    print(f'Saved to {fig_name}')
    plt.show()

toss_id=1
filename = f'old_toss_{toss_id}'
BUNDLESDF_POSE_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/results/"+filename+"/ob_in_cam/"
ODOM_FILE_PATH = "/home/cnets-vision/mengti_ws/BundleSDF/data/"+filename+"/annotated_poses/"
CAMERA_EXTRINSICS_FILE = '/home/cnets-vision/mengti_ws/robot_filter/assets/realsense_pose_cube_old.yaml'
CONTACTNETS_INPUT_DIR = f"/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_test/"
cam = 'cam0' # realsense camera name
with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
cam_pos_dict = data_loaded[cam]['pose']['position']
cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
cam_rot_dict = data_loaded[cam]['pose']['rotation']
cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])

matrices = []
for frame_id in range(366, 385):
    pose = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % frame_id)
    pose = transform_bundletrack_origin_to_tagslam_origin(pose, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH, cam_trans, cam_axis_vec, to_world=True)
    matrices.append(pose)

dt = 0.1
result = matrix_to_trajectory(matrices, dt)

