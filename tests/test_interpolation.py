import numpy as np
from scipy.interpolate import interp1d
import torch 
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

def visualize_trajectory(fig_name, quats, positions, angular_vels, linear_vels):
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))

    # Plot positions
    ax[0, 0].plot(positions[:, 0])
    ax[0, 0].set_title('X Position')
    ax[0, 1].plot(positions[:, 1])
    ax[0, 1].set_title('Y Position')
    ax[0, 2].plot(positions[:, 2])
    ax[0, 2].set_title('Z Position')

    # Plot Quaternion components
    ax[1, 0].plot(quats[:, 0])
    ax[1, 0].set_title('Quaternion q0')
    ax[1, 1].plot(quats[:, 1])
    ax[1, 1].set_title('Quaternion q1')
    ax[1, 2].plot(quats[:, 2])
    ax[1, 2].set_title('Quaternion q2')
    ax[2, 0].plot(quats[:, 3])
    ax[2, 0].set_title('Quaternion q3')

    # Plot angular velocities
    ax[2, 1].plot(angular_vels[:, 0])
    ax[2, 1].set_title('Angular Velocity X')
    ax[2, 2].plot(angular_vels[:, 1])
    ax[2, 2].set_title('Angular Velocity Y')
    ax[3, 0].plot(angular_vels[:, 2])
    ax[3, 0].set_title('Angular Velocity Z')

    # Plot linear velocities
    ax[3, 1].plot(linear_vels[:, 0])
    ax[3, 1].set_title('Linear Velocity X')
    ax[3, 2].plot(linear_vels[:, 1])
    ax[3, 2].set_title('Linear Velocity Y')

    plt.tight_layout()
    fig.suptitle('ContactNets cube trajectory')
    plt.savefig(fig_name)
    print(f'Saved to {fig_name}')
    plt.show()

def quaternion_to_angular_velocity(q0, q1, dt):
    dq = q1 * q0.inverse
    if dq.w < 0:
        dq = -dq
    angle = 2 * np.arctan2(np.linalg.norm([dq.x, dq.y, dq.z]), dq.w)
    norm = np.linalg.norm([dq.x, dq.y, dq.z])
    if norm < 1e-10:  # ignorable rotation
        axis = np.array([0.0, 0.0, 0.0])
    else:
        axis = np.array([dq.x, dq.y, dq.z]) / norm
    return axis * angle / dt

file_path = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_cube/0.pt'
data = torch.load(file_path)   #p_t, quat_shuffle, dp_t, w_t
filename = f'old_toss_1'
GT_POSE_DIR = "/home/cnets-vision/mengti_ws/robot_filter/dataset/"+filename+"/tagslam_poses/"
csv = np.loadtxt(GT_POSE_DIR+'tagslam.txt') #N,9
start_frame = 366
end_frame = 384
print('data loaded', data.shape)
csv = csv[start_frame:end_frame]
bundletrack_time, gt_time = csv[:, 0], csv[:, 1] #N,
t_start = bundletrack_time[0]
bundletrack_time = bundletrack_time - t_start
print(bundletrack_time)
N = data.shape[0]
positions = data[:, :3].numpy() #N,3
quats = data[:, 3:7].numpy() #N,4
print(positions.shape, quats.shape)
# Preprocess quaternions
quats = quats / np.linalg.norm(quats, axis=1)[:, np.newaxis]
for i in range(1, N):
    if np.dot(quats[0], quats[i]) < 0:
        quats[i] = -quats[i]

# upsample factor
M = 100
new_times = np.linspace(0, 1, M)

########################### interpolate positions ######################
interp_func_x = interp1d(np.linspace(0, 1, N), positions[:, 0], kind='linear')
interp_func_y = interp1d(np.linspace(0, 1, N), positions[:, 1], kind='linear')
interp_func_z = interp1d(np.linspace(0, 1, N), positions[:, 2], kind='linear')

new_positions = np.vstack([
    interp_func_x(new_times),
    interp_func_y(new_times),
    interp_func_z(new_times)
]).T

########################### interpolate quaternions #########################
new_quaternions = np.zeros((M, 4))
for i in range(N-1):
    start_idx = i * (M // (N-1))
    end_idx = start_idx + (M // (N-1))
    t_array = np.linspace(0, 1, M//(N-1))
    q0 = Quaternion(quats[i, :])
    q1 = Quaternion(quats[i+1, :]) if i < N-2 else Quaternion(quats[-1, :])
    interpolated_quats = [Quaternion.slerp(q0, q1, amount=t).elements for t in t_array]
    new_quaternions[start_idx:end_idx, :] = np.array(interpolated_quats)
if end_idx < M:
    new_quaternions[end_idx:] = quats[-1]

########################### interpolate timestamps ###########################
interp_func_timestamps = interp1d(np.linspace(0, 1, N), bundletrack_time, kind='linear')
new_timestamps = interp_func_timestamps(new_times)

########################### compute velocities ###########################
dt_upsampled = np.diff(new_timestamps)
linear_velocities_upsampled = np.diff(new_positions, axis=0) / dt_upsampled[:, np.newaxis]
linear_velocities_upsampled = np.vstack([linear_velocities_upsampled, linear_velocities_upsampled[-1]])

angular_velocities_upsampled = np.array([quaternion_to_angular_velocity(Quaternion(new_quaternions[i]), 
                                                                      Quaternion(new_quaternions[i+1]), 
                                                                      dt_upsampled[i]) 
                                         for i in range(M-1)])
angular_velocities_upsampled = np.vstack([angular_velocities_upsampled, angular_velocities_upsampled[-1]])
visualize_trajectory('test_interp.png', new_quaternions, new_positions, angular_velocities_upsampled, linear_velocities_upsampled)
# visualize_trajectory('original_traj.png', quats, positions, angular_vels, linear_vels)