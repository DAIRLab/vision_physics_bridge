import numpy as np
import argparse
import yaml
import os
import trimesh
from scipy.spatial import cKDTree
import open3d as o3d
from math_utils import pos_quat_to_trans_mat, transform_bundletrack_output

def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
    return homo

def add_err(pred,gt,model_pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    """
    pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
    gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
    e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    return e

def adi_err(pred,gt,model_pts):
    """
    @pred: 4x4 mat
    @gt:
    @model: (N,3)
    """
    pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
    gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e

def compute_auc(rec, max_val=0.1):
    '''https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/2df96b720e8e499b9f0d5fcebfbae2bcfa51ab19/eval_ycb.py#L45
    '''
    if len(rec)==0:
        return 0
    rec = np.sort(np.array(rec))
    n = len(rec)
    prec = np.arange(1,n+1) / float(n)
    rec = rec.reshape(-1)
    prec = prec.reshape(-1)
    index = np.where(rec<max_val)[0]
    rec = rec[index]
    prec = prec[index]

    mrec=[0, *list(rec), max_val]
    mpre=[0, *list(prec), prec[-1]]

    for i in range(1,len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    mpre = np.array(mpre)
    mrec = np.array(mrec)
    i = np.where(mrec[1:]!=mrec[0:len(mrec)-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) / max_val
    return ap

def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud

def chamfer_distance_between_clouds_mutual(pts1,pts2):
  kdtree1 = cKDTree(pts1)
  dists1, indices1 = kdtree1.query(pts2)
  kdtree2 = cKDTree(pts2)
  dists2, indices2 = kdtree2.query(pts1)
  return 0.5*(dists1.mean()+dists2.mean())   #!NOTE should not be mean of all, see https://pdal.io/en/stable/apps/chamfer.html

def sample_points_from_mesh(mesh, num_points):
    """Sample points from a mesh."""
    # Get the area of each face
    face_areas = mesh.area_faces
    
    # Sample faces based on their area weights
    sampled_faces = np.random.choice(len(mesh.faces), num_points, p=face_areas/face_areas.sum())
    
    # Sample a random point from each sampled face
    sampled_points = [trimesh.sample.sample_surface(mesh, 1, face) for face in sampled_faces]
    return np.vstack(sampled_points)

def benchmark_one_video():
    gt_data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
    pred_poses, gt_poses = [], []
    for frame_id in range(1, frame_num):
        output_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % frame_id)
        output_pose = transform_bundletrack_output(
            output_pose,
            OUTPUT_POSE_DIR,
            ODOM_FILE_PATH,
            cam_trans,
            cam_axis_vec,
            to_world=True
        )
        pred_poses.append(output_pose)
        tagslam_pose = gt_data[frame_id, 2:]
        tagslam_mat = pos_quat_to_trans_mat(tagslam_pose)
        gt_poses.append(tagslam_mat)
    gt_mesh = trimesh.load(GT_MESH_FILE)
    pred_mesh = trimesh.load(PRED_MESH_FILE)
    gt_poses = np.array(gt_poses)
    pred_poses = np.array(pred_poses)
    adi_errs, add_errs = [], []
    for i in range(len(pred_poses)):
        adi = adi_err(pred_poses[i],gt_poses[i],gt_mesh.vertices.copy())
        add = add_err(pred_poses[i],gt_poses[i],gt_mesh.vertices.copy())
        adi_errs.append(adi)
        add_errs.append(add)
    adi_errs = np.array(adi_errs)
    add_errs = np.array(add_errs)
    ADDS_AUC = compute_auc(adi_errs)*100
    ADD_AUC = compute_auc(add_errs)*100
    print(ADD_AUC, ADDS_AUC)
    
    ### mesh 
    cd = np.inf
    if gt_mesh and pred_mesh is not None:
        pred_pts,_ = trimesh.sample.sample_surface(pred_mesh, 99999, face_weight=None, sample_color=False)
        pcd_pred = toOpen3dCloud(pred_pts)
        pcd_pred = pcd_pred.voxel_down_sample(0.005)
        # pcd_gt = toOpen3dCloud(gt_pts)
        pcd_gt = o3d.io.read_point_cloud(GT_PCD_DIR)
        gt_pts = np.asarray(pcd_gt.points)
        thres = 0.02
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd_pred, pcd_gt, thres, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
        pred_pts_icp = (reg_p2p.transformation@to_homo(pred_pts).T).T[:,:3]
        chamfer_dists = chamfer_distance_between_clouds_mutual(pred_pts_icp, gt_pts)
        cd = chamfer_dists.mean()*100
        print("chamfer_dist(cm)",cd)

        # pcd = toOpen3dCloud(gt_pts)
        # o3d.io.write_point_cloud(f'{PCD_DIR}gt_pts.ply',pcd)
        # pcd = toOpen3dCloud(pred_pts)
        # o3d.io.write_point_cloud(f'{PCD_DIR}old_toss_{toss_id}_pred_pts.ply',pcd)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    toss_id = args.toss_id
    OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/old_toss_{toss_id}/ob_in_cam_exp_4/"
    ODOM_FILE_PATH = f"/home/cnets-vision/mengti_ws/BundleSDF/data/old_toss_{toss_id}/annotated_poses/"
    GT_POSE_DIR = f"./dataset/old_toss_{toss_id}/tagslam_poses/"
    PCD_DIR = f"./assets/"
    GT_PCD_DIR = f"./assets/gt_cube.ply"
    CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube_old.yaml"
    PRED_MESH_FILE = "./assets/mesh_cn_run_and_refined.obj"
    GT_MESH_FILE = "./assets/contactnets_cube.obj"
    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    print(data_loaded[cam]['pose']['position'])
    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    frame_num = len([name for name in os.listdir(OUTPUT_POSE_DIR)])
    benchmark_one_video()