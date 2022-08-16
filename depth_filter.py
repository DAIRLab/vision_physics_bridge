from turtle import right
import open3d as o3d
import numpy as np

from utils import axis_angle_to_rotation_matrix, get_table_world_coordinates, world_to_point_cloud

"""
Filter out everything in the depth image except for the object of interest.
TODO: Map table height/width in reality to point cloud height
"""
depth_image_title = './aligned_data/frame000001.png'
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
rotation = axis_angle_to_rotation_matrix(axis, angle)
extrinsic = np.vstack((np.hstack((rotation, translation)), np.array([0,0,0,1])))

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=pinhole_camera_intrinsic,extrinsic=extrinsic)

# flip the orientation, so it looks upright, not upside-down
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
pcd_array = np.asarray(pcd.points)
# print(pcd_array)
# o3d.visualization.draw_geometries([mesh_frame, pcd])    # visualize the point cloud
left_bottom, right_bottom, left_top, right_top = get_table_world_coordinates()
K = [[380.2484436035156, 0, 314.2138977050781], 
        [0, 379.8265380859375, 240.59800720214844], 
        [0, 0, 1]]
left_bottom_pcl = world_to_point_cloud(left_bottom[0], left_bottom[1], left_bottom[2], K, rotation, depth_scale=1000)
right_bottom_pcl = world_to_point_cloud(right_bottom[0], right_bottom[1], right_bottom[2], K, rotation, depth_scale=1000)
left_top_pcl = world_to_point_cloud(left_top[0], left_top[1], left_top[2], K, rotation, depth_scale=1000)
right_top_pcl = world_to_point_cloud(right_top[0], right_top[1], right_top[2], K, rotation, depth_scale=1000)

corners = np.array([left_bottom_pcl, right_bottom_pcl, left_top_pcl, right_top_pcl])
print(left_bottom_pcl, right_bottom_pcl, left_top_pcl, right_top_pcl)
# Convert the corners array to have type float64
bounding_polygon = corners.astype("float64")

# Create a SelectionPolygonVolume
vol = o3d.visualization.SelectionPolygonVolume()

# You need to specify what axis to orient the polygon to.
# I choose the "Y" axis. I made the max value the maximum Y of
# the polygon vertices and the min value the minimum Y of the
# polygon vertices.
vol.orthogonal_axis = "Y"
vol.axis_max = np.max(bounding_polygon[:, 1])
vol.axis_min = np.min(bounding_polygon[:, 1])

# Set all the Y values to 0 (they aren't needed since we specified what they
# should be using just vol.axis_max and vol.axis_min).
bounding_polygon[:, 1] = 0

# Convert the np.array to a Vector3dVector
vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

# Crop the point cloud using the Vector3dVector
cropped_pcd = vol.crop_point_cloud(pcd)

# # Mask out above a height
# # threshold data
# points = np.asarray(pcd.points)
# pcd_sel = pcd.select_by_index(np.where(points[:, 2] >=0)[0])

# # visualize different point clouds
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([pcd_sel])

# bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-2, 0, -2), max_bound=(2, 1, 4))
# cropped_pcd = pcd.crop(bbox)
# print(cropped_pcd.points)
# o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0.0, 0.0, 0.0]))
# print(cropped_pcd.has_normals)
# viewer = o3d.visualization.Visualizer()
# viewer.create_window()

# viewer.add_geometry(cropped_pcd)
# opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.5, 0.5, 0.5])
# viewer.run()
# viewer.destroy_window()
o3d.visualization.draw_geometries([cropped_pcd])