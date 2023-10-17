import numpy as np 
import trimesh
import open3d as o3d
import argparse

def sample_points_from_mesh(mesh, num_points):
    """Sample points from a mesh."""
    mesh = trimesh.load_mesh(GT_MESH_FILE)
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    filename = './output.ply'
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    print(f'pcd {filename} exported')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    toss_id = args.toss_id
    DATASET = f'napkin_{toss_id}'
    GT_MESH_FILE = "/home/cnets-vision/mengti_ws/robot_filter/assets/gt_napkin_simple.obj"
    PRED_MESH_FILE = f'/home/cnets-vision/mengti_ws/BundleSDF/results/{DATASET}/textured_mesh.obj'
    pred_mesh = trimesh.load(PRED_MESH_FILE)
    pred_pts,_ = trimesh.sample.sample_surface(pred_mesh, 99999, face_weight=None, sample_color=False)
    print(f'pred_pt is {len(pred_pts)}')
    gt_mesh = trimesh.load(GT_MESH_FILE)
    sample_points_from_mesh(gt_mesh, len(pred_pts))