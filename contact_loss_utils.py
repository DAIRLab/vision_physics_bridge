import os
import shutil
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, KDTree
import yaml
import argparse

def best_fit_transform(A, B):
    assert len(A) == len(B)
    # Compute mean of both datasets
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # Subtract mean
    AA = A - centroid_A
    BB = B - centroid_B
    # Rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # Translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    # Homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T, R, t

def icp(A, B, max_iterations=20, tolerance=1e-8):
    prev_error = 0
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[0:3, :] = np.copy(A.T)
    dst[0:3, :] = np.copy(B.T)
    # KDTree for fast nearest neighbor search
    tree = KDTree(B)
    distances = []
    for i in range(max_iterations):
        # Find the nearest neighbors between the current source and destination
        distances, indices = tree.query(src[:3, :].T)
        # Compute the transformation between the current source and nearest destination
        T, _, _ = best_fit_transform(src[:3, :].T, B[indices])
        # Update the current source
        src = np.dot(T, src)
        # Check for convergence
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T, _, _ = best_fit_transform(A, src[:3, :].T)
    return T, distances

def setup_extrinsic(translation, axis_vec):
    """
    Convert translation and axis-angle representation to extrinsic matrix.
    
    Parameters:
    - translation: 3x1 numpy array, translation vector.
    - axis_vec: 3x1 numpy array, rotation represented in axis-angle (rodriques) form.

    Returns:
    - 4x4 numpy array, extrinsic matrix.
    """
    rotation_matrix = R.from_rotvec(axis_vec.ravel()).as_matrix()
    rotation_inverse = rotation_matrix.T
    translation_inverse = -rotation_inverse @ translation.ravel()
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_inverse
    extrinsic[:3, 3] = translation_inverse
    return extrinsic

def clean_and_denoise_mesh(mesh):
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_laplacian(10, 0.5)
    return mesh

def transform_points(points, transformation_matrix):
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    transformed_points = points_homogeneous @ transformation_matrix.T
    return transformed_points[:, :3]

def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo

def transform_pts_to_normalized_space(contact_pts, ob_init_cam, translation, sc_factor):
    ob_init_cam[:,3] = np.array([0,0,0,1])
    contact_cam = transform_points(contact_pts, ob_init_cam)
    offset = np.array([[1, 0, 0, -4.835662426039277761e-10],
                        [0, 1, 0, -1.494223023090768265e-08],
                        [0, 0, 1, -5.918124390547063740e-08],
                        [0, 0, 0, 1]])

    contact_cam = transform_points(contact_cam, offset)
    contact_cam += translation.reshape(1,3)
    contact_cam *= sc_factor
    return contact_cam

def transform_mesh_to_normalized_space(gt_mesh, ob_init_cam, T, translation, sc_factor):
    vertices = gt_mesh.vertices
    normals = gt_mesh.vertex_normals
    faces = gt_mesh.faces
    offset = np.array([[1, 0, 0, -4.835662426039277761e-10],
                        [0, 1, 0, -1.494223023090768265e-08],
                        [0, 0, 1, -5.918124390547063740e-08],
                        [0, 0, 0, 1]])
    ob_init_cam[:, 3] = np.array([0, 0, 0, 1])
    transformed_vertices = transform_pts_to_normalized_space(vertices, ob_init_cam, translation, sc_factor)
    
    transformed_vertices = transform_points(transformed_vertices, T)
    
    # Transform normals (apply only rotation)
    rotation_matrices = [ob_init_cam[:3, :3], offset[:3, :3], T[:3, :3]]
    transformed_normals = normals
    for rot in rotation_matrices:
        transformed_normals = np.dot(transformed_normals, rot.T)
    transformed_normals = transformed_normals / np.linalg.norm(transformed_normals, axis=1, keepdims=True)
    mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces, vertex_normals=transformed_normals)
    # mesh.export('./transformed_mesh.obj', file_type='obj')
    return mesh

def generate_contact_loss_data(path, output_path, output_w_path, ob_init_cam, translation, sc_factor, save=False, num_samples=1000):
    '''
    Visaulize two points clouds in C_prime frame where C_prime is BundleSDF's object body frame.
      pC_prime = TC_C_prime @ TW_C @ TB_W @ p_B.

    @path:  body frame contact points, aka p_B
    @output_path: bundlesdf output in normalized space, aka mesh_cleaned.obj
    @output_w_path: bundlesdf output transformed to real world unit, aka textured_mesh.obj
    @ob_init_cam: object's initial pose in camera frame, aka TB_C
    @cam_extrinsic: World to camera transformation, aka TW_C
    
    '''
    # contact_pts = np.load(path)
    gt_mesh = trimesh.load(path, force='mesh')
    contact_pts = gt_mesh.sample(num_samples)
    # contact_pcd = o3d.io.read_point_cloud(path)
    # contact_pts = np.asarray(contact_pcd.points)
    # contact_pts = contact_pts[np.random.choice(contact_pts.shape[0], num_samples, replace=False)]

    normalized_mesh = trimesh.load(output_path, force='mesh')
    normalized_pts = normalized_mesh.sample(num_samples)
    
    pred_mesh = trimesh.load(output_w_path, force='mesh') #in cam
    
    output_pts = pred_mesh.sample(num_samples)
    contact_cam = transform_pts_to_normalized_space(contact_pts, ob_init_cam, translation, sc_factor)
    

    contact_pcd = o3d.geometry.PointCloud()
    contact_pcd.points = o3d.utility.Vector3dVector(contact_cam)
    
    T, _ = icp(contact_cam, normalized_pts)
    contact_pcd.transform(T)
    contact_cam = np.asarray(contact_pcd.points)
    # SDF values of contact points are 0
    contact_sdf_values = np.zeros(contact_cam.shape[0])

    transformed_mesh = transform_mesh_to_normalized_space(gt_mesh, ob_init_cam, T, translation, sc_factor)
    near_surface_pts, near_surface_sdf = get_near_surface_pts_and_sdf(transformed_mesh)
    
    all_pts = np.concatenate((contact_cam, near_surface_pts), axis=0)
    all_sdf = np.concatenate((contact_sdf_values, near_surface_sdf), axis=0)
    # print(all_sdf[2000:2100])
    all_sdf *= sc_factor
    print(all_sdf[2000:2100])
    print(all_pts.shape, all_sdf.shape)
    if save:
        np.save('./contact_and_near_surface_pts_with_various_dist.npy', all_pts)
        np.save('./contact_and_near_surface_sdf_with_various_dist.npy', all_sdf)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    contact_pts_cpu = contact_cam
    output_pts = normalized_pts
    near_surface = near_surface_pts
    ax.scatter(near_surface[:, 0], near_surface[:, 1], near_surface[:, 2], color='green', label='near surface pts')
    ax.scatter(contact_pts_cpu[:, 0], contact_pts_cpu[:, 1], contact_pts_cpu[:, 2], color='blue', label='contact pts')
    ax.scatter(output_pts[:,0], output_pts[:,1], output_pts[:,2], color='red', label='output pts')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
    # plt.savefig('./vis.png')

def sample_points_on_surface(mesh, number_of_points):
    """
    Sample points uniformly from the surface of the mesh.
    
    :param mesh: trimesh object
    :param number_of_points: number of points to sample
    :return: array of sampled points and their normals
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, number_of_points)
    normals = mesh.face_normals[face_indices]
    return points, normals

def get_near_surface_pts_and_sdf(obj_file, distance=0.05, num_pts=2, surface_points=500):
    '''
    Generate sets of near-surface points and corresponding signed distances to augment contact point data in order to regularize BundleSDF's NeRF
    '''
    mesh = trimesh.load(obj_file)
    surface_pts, surface_normals = sample_points_on_surface(mesh, number_of_points=surface_points)
    sdf_points = []
    sdf_values = []
    # print(mesh.vertices.shape)
    for point, normal in zip(surface_pts, surface_normals):
        for _ in range(num_pts):
            d = np.random.uniform(0, distance)  # Random distance within the range
            outward_point = point + d * normal
            sdf_points.append(outward_point)
            sdf_values.append(d)  # Positive SDF

        # Points along the negative normal
        for _ in range(num_pts):
            d = np.random.uniform(0, distance)  # Random distance within the range
            inward_point = point - d * normal
            sdf_points.append(inward_point)
            sdf_values.append(-d)  # Negative SDF
    sdf_points, sdf_values = np.array(sdf_points), np.array(sdf_values)
    print(sdf_points.shape, sdf_values.shape)
    # visualize_pts(sdf_points)
    return sdf_points, sdf_values

def get_transformed_obj_for_nerf_init(gt_path, output_path, ob_init_cam, num_samples=1000):
    """
    Given a ground-truth .obj, transform to BundleSDF's normalized space for Octree initialization.
    @gt_path: path of ground-truth .obj
    @output_path: path of mesh_cleaned.obj
    @ob_init_cam: object's initial pose in camera frame, aka TB_C
    @num_samples: number of sampled points
    """
    gt_mesh = trimesh.load(gt_path, force='mesh')
    contact_pts = gt_mesh.sample(num_samples)
    contact_cam = transform_pts_to_normalized_space(contact_pts, ob_init_cam, translation, sc_factor)

    normalized_mesh = trimesh.load(output_path, force='mesh')
    normalized_pts = normalized_mesh.sample(num_samples)
    T, _ = icp(contact_cam, normalized_pts)
    transformed_mesh = transform_mesh_to_normalized_space(gt_mesh, ob_init_cam, T, translation, sc_factor)
    transformed_pts = transformed_mesh.sample(num_samples)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    contact_pts_cpu = transformed_pts
    output_pts = normalized_pts
    ax.scatter(contact_pts_cpu[:, 0], contact_pts_cpu[:, 1], contact_pts_cpu[:, 2], color='blue', label='transformed pts')
    ax.scatter(output_pts[:,0], output_pts[:,1], output_pts[:,2], color='red', label='output pts')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()

def visualize_pts(pts):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(contact_pts_cpu[:, 0], contact_pts_cpu[:, 1], contact_pts_cpu[:, 2], color='blue', label='contact_pts')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='red', label='output pts')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=bool,
        required=False,
    )
    args = parser.parse_args()
    save = args.save
    visible_pcd = './visible_pcd.ply'
    gt_mesh = './assets/gt_cube_simple.obj'
    output_path = './mesh_cleaned_cube2.obj'
    output_w_path = './textured_mesh_cube1.obj'
    # contact_path = './transformed_pts.npy'
    contact_path = './final_pts_tf.npy'
    DATASET = 'cube'
    toss_id=2
    ODOM_FILE_PATH = f"/home/cnets-vision/mengti_ws/BundleSDF/data/{DATASET}_{toss_id}/annotated_poses/"
    init_pose = np.loadtxt(
        ODOM_FILE_PATH + "%04i.txt" % 0
    )  # initial cube pose in camera frame, bundletrack's internal coordinate system
    OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/{DATASET}_{toss_id}/ob_in_cam/"
    CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube.yaml"

    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    print(data_loaded[cam]['pose']['position'])
    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    extrinsic = setup_extrinsic(cam_trans, cam_axis_vec)
    ob_init_pose = np.loadtxt(
        ODOM_FILE_PATH + "%04i.txt" % 0
    )  # initial cube pose represented in camera frame, matching tagslam
    pred_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % 1)

    translation = np.array([0.027168031322692257, -0.006110663910054243, 0.020553499466653358])
    sc_factor = 6.294841024247843
    # generate_contact_loss_data(gt_mesh, output_path, output_w_path, ob_init_pose, translation, sc_factor, save=save)
    get_transformed_obj_for_nerf_init(gt_mesh, output_path, ob_init_pose, num_samples=2000)