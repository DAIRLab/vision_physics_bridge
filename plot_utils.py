import numpy as np 
import scipy.spatial as sp
import matplotlib.pyplot as plt

GT_POSE_DIR = "/home/cnets-vision/mengti_ws/robot_filter/tagslam_poses/"
OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/poses/"

def get_cosine_sim(frame_id):
        """Compare the output of BundleTrack with the ground-truth poses of tagslam.
        """
        gt_pose = np.loadtxt(GT_POSE_DIR+'%04i.txt'% frame_id)
        output_pose = np.loadtxt(OUTPUT_POSE_DIR+'%04i.txt'% frame_id)
        return 1 - sp.distance.cdist(gt_pose, output_pose, 'cosine')

def get_angle(P, Q):
    R = np.dot(P, Q.T)
    theta = (np.trace(R) -1)/2
    return np.arccos(theta) * (180/np.pi)

def plot_xyz(start_frame, end_frame):
    """Plot the x, y, z of BundleTrack output versus ground-truth poses of tagslam.
    """
    output_x, output_y, output_z = [], [], [] #translation
    gt_x, gt_y, gt_z = [], [], []
    angles = []

    for frame_id in range(start_frame, end_frame):
        gt_pose = np.loadtxt(GT_POSE_DIR+'%04i.txt'% frame_id)
        output_pose = np.loadtxt(OUTPUT_POSE_DIR+'%04i.txt'% frame_id)
        angle = get_angle(output_pose[:3, :3], gt_pose[:3, :3])
        angles.append(angle)
        output_x.append(output_pose[0, 3])
        output_y.append(output_pose[1, 3])
        output_z.append(output_pose[2, 3])
        
        gt_x.append(gt_pose[0, 3])
        gt_y.append(gt_pose[1, 3])
        gt_z.append(gt_pose[2, 3])
    output_x, output_y, output_z = np.array(output_x), np.array(output_y), np.array(output_z)
    gt_x, gt_y, gt_z = np.array(gt_x), np.array(gt_y), np.array(gt_z)
    
    x = np.arange(0, output_x.shape[0])
    plt.plot(x, output_x, label="Bundletrack")
    plt.plot(x, gt_x, label="ground-truth")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Position x")
    plt.legend()
    plt.show()

    plt.plot(x, output_y, label="BundleTrack")
    plt.plot(x, gt_y, label="ground-truth")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Position y")
    plt.legend()
    plt.show()

    plt.plot(x, output_z, label="BundleTrack")
    plt.plot(x, gt_z, label="ground-truth")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Position z")
    plt.legend()
    plt.show()

    plt.plot(x, angles)
    plt.xlabel("X-axis")
    plt.ylabel("angle diff (degrees)")
    plt.title("Angle difference")
    plt.show()


if __name__ ==  '__main__':
    plot_xyz(1, 933)