import numpy as np
import cv2
import pywavefront
import os
import yaml
from math_utils import setup_extrinsic, transform_bundletrack_output

# DATASET_NAME = "cube_hand_toss_60"
DATASET_NAME = "cube_hand_1"
MESH_FILE = "./assets/contactnets_cube.obj"
BUNDLESDF_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "bundlenets")
BUNDLESDF_DATASET_DIR = os.path.join(BUNDLESDF_DIR, "data", DATASET_NAME)
BUNDLESDF_RESULT_DIR = os.path.join(BUNDLESDF_DIR, "results", DATASET_NAME)
ANNOTATED_POSE_DIR = os.path.join(BUNDLESDF_DATASET_DIR, "annotated_poses/")
ANNOTATED_POSE_FILE = os.path.join(BUNDLESDF_DATASET_DIR, "annotated_poses", "0000.txt")
CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube_hand_60_2.yaml"
RGB_FILE = os.path.join(BUNDLESDF_DATASET_DIR, "rgb", "0001.png")

cam = 'cam0' # realsense camera name
with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded[cam]['pose']['position'])
cam_pos_dict = data_loaded[cam]['pose']['position']
cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
cam_rot_dict = data_loaded[cam]['pose']['rotation']
cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])

intrinsics = [380.2484436035156, 379.8265380859375,314.2138977050781, 240.59800720214844,] # fx, fy, cx, cy
fx = intrinsics[0]
fy = intrinsics[1]
cx = intrinsics[2]
cy = intrinsics[3]
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

def transform_points(points, transformation_matrix):
    homogeneous_points = np.vstack((points, np.ones(points.shape[1])))
    transformed_points = np.dot(transformation_matrix, homogeneous_points)
    return transformed_points[:3, :]

def project_points(points, intrinsic_matrix):
    image_points = np.dot(intrinsic_matrix, points[:3, :])
    image_points /= image_points[2, :]
    return image_points[:2, :]

def get_initial_pose_overlay():
    scene = pywavefront.Wavefront(MESH_FILE, collect_faces=True)
    vertices = np.array(scene.vertices).T
    object_pose = np.loadtxt(ANNOTATED_POSE_FILE)
    extrinsic_matrix = setup_extrinsic(cam_trans, cam_axis_vec)
    total_transform = np.dot(extrinsic_matrix, object_pose)
    transformed_vertices = transform_points(vertices, total_transform)

    image_points = project_points(transformed_vertices, intrinsic_matrix)
    image = cv2.imread(RGB_FILE)

    for face in scene.mesh_list[0].faces:
        triangle = np.array([image_points[:, face[i]] for i in range(3)], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(image, [triangle], (0, 255, 0))

    # # Show the image
    # cv2.imshow("Image with Cube Overlay", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("overlay.png", image)

def get_overlay_video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('overlay_video.mp4', fourcc, 20.0, (640, 480))
    scene = pywavefront.Wavefront(MESH_FILE, collect_faces=True)
    vertices = np.array(scene.vertices).T

    num_frames = len(os.listdir(os.path.join(BUNDLESDF_DATASET_DIR, "rgb")))
    for i in range(1, num_frames):
        pose_file_name = os.path.join(BUNDLESDF_RESULT_DIR, "ob_in_cam", f"{i:04}.txt")
        rgb_file_name = os.path.join(BUNDLESDF_DATASET_DIR, "rgb", f"{i:04}.png")

        object_pose = np.loadtxt(pose_file_name)
        object_pose = transform_bundletrack_output(
            object_pose,
            os.path.join(BUNDLESDF_RESULT_DIR, "ob_in_cam/"),
            ANNOTATED_POSE_DIR,
            cam_trans,
            cam_axis_vec,
            # to_world=True,
        ) # camera frame
        # extrinsic_matrix = setup_extrinsic(cam_trans, cam_axis_vec)
        # total_transform = np.dot(extrinsic_matrix, object_pose)
        total_transform = object_pose
        transformed_vertices = transform_points(vertices, total_transform)
        image_points = project_points(transformed_vertices, intrinsic_matrix)

        image = cv2.imread(rgb_file_name)

        for face in scene.mesh_list[0].faces:
            triangle = np.array([image_points[:, face[j]] for j in range(3)], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [triangle], (0, 255, 0))

        out.write(image)

    out.release()

if __name__ == "__main__":
    get_overlay_video()