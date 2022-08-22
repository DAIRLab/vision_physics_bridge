import open3d as o3d
import numpy as np

from utils import axis_angle_to_rotation_matrix, get_table_world_coordinates
"""
Filter out everything in the depth image except for the object of interest.
TODO: Map table height/width in reality to point cloud height
TODO: Take camera orientation into consideration when orienting the obb
"""
depth_image_title = './data/frame000001.png'
color_image_title = './rgb_data/frame000001.png'
depth = o3d.io.read_image(depth_image_title)
color = o3d.io.read_image(color_image_title)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
# Setup camera intrinsic
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=380.2484436035156, fy=379.8265380859375, cx=314.2138977050781, cy=240.59800720214844)
# Setup camera extrinsic
translation = np.array([[1.14164360], [0.15815239], [0.66422200]])
axis_vec = [-1.57165949, -1.63112887, 1.07928078]
angle = np.linalg.norm(axis_vec)
axis = axis_vec / angle
rotation = axis_angle_to_rotation_matrix(axis, angle) # directions of the world-axes in camera coordinates
rotation_prime = rotation.T #USE THIS
translation_prime = -rotation_prime @ translation #USE THIS
extrinsic = np.vstack((np.hstack((rotation_prime, translation_prime)), np.array([0,0,0,1])))
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=pinhole_camera_intrinsic, extrinsic=extrinsic)
K = [[380.2484436035156, 0, 314.2138977050781],
    [0, 379.8265380859375, 240.59800720214844], 
    [0, 0, 1]]
# Setup the orientatin of point cloud
pcd_obb = pcd.get_oriented_bounding_box()
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0,0,0])
pcd_array = np.asarray(pcd.points)
mesh_frame.translate(pcd_obb.center)

# print("pcd:",pcd.get_oriented_bounding_box().R)
# print("mesh:",mesh_frame.get_oriented_bounding_box().R)
# o3d.visualization.draw_geometries([pcd, mesh_frame])

left_bottom, right_bottom, left_top, right_top = get_table_world_coordinates()

left_bottom_pcl = np.array((left_bottom[0], left_bottom[1], left_bottom[2]))
right_bottom_pcl = np.array((right_bottom[0], right_bottom[1], right_bottom[2]))
left_top_pcl = np.array((left_top[0], left_top[1], left_top[2]))
right_top_pcl = np.array((right_top[0], right_top[1], right_top[2]))
height_buffer = 0.1
left_bottom_pcl_ = np.array((left_bottom[0], left_bottom[1], left_bottom[2]-height_buffer))
right_bottom_pcl_ = np.array((right_bottom[0], right_bottom[1], right_bottom[2]-height_buffer))
left_top_pcl_ = np.array((left_top[0], left_top[1], left_top[2]-height_buffer))
right_top_pcl_ = np.array((right_top[0], right_top[1], right_top[2]-height_buffer))

corners = np.array([left_bottom_pcl, right_bottom_pcl, left_top_pcl, right_top_pcl, left_bottom_pcl_, right_bottom_pcl_, left_top_pcl_, right_top_pcl_])
print('x:', left_bottom_pcl[0], right_bottom_pcl[0], left_top_pcl[0], right_top_pcl[0], left_bottom_pcl_[0], right_bottom_pcl_[0], left_top_pcl_[0], right_top_pcl_[0])
print('y:', left_bottom_pcl[1], right_bottom_pcl[1], left_top_pcl[1], right_top_pcl[1], left_bottom_pcl_[1], right_bottom_pcl_[1], left_top_pcl_[1], right_top_pcl_[1])
print('z:', left_bottom_pcl[2], right_bottom_pcl[2], left_top_pcl[2], right_top_pcl[2], left_bottom_pcl_[2], right_bottom_pcl_[2], left_top_pcl_[2], right_top_pcl_[2])

# Rotate the bounding polygon to match the orientation of the point cloud. bbox -> obb
bounding_polygon = corners.astype("float64")
coord_obb = mesh_frame.get_axis_aligned_bounding_box().get_oriented_bounding_box()
# Rotate oriented_bounding_box to mesh_frame
bounding_points = o3d.utility.Vector3dVector(bounding_polygon)
oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bounding_points)
relative_rotation = coord_obb.R @ np.linalg.inv(oriented_bounding_box.R)
oriented_bounding_box.rotate(R=relative_rotation, center=oriented_bounding_box.center)

o3d.visualization.draw_geometries([pcd, oriented_bounding_box, mesh_frame])

obb = pcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)
length = 182.8/100 #x in ros, y in open3d
width = 91.3/100 #y in ros, x in open3d
height = 73.2/100 #z

cropped_pcd = pcd.crop(oriented_bounding_box)
# print(cropped_pcd.points)
o3d.visualization.draw_geometries([cropped_pcd, oriented_bounding_box, mesh_frame])
# pick_points(pcd)