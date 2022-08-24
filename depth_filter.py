import open3d as o3d
import numpy as np
from utils import CUBE_LENGTH, axis_angle_to_rotation_matrix, get_table_world_coordinates
from visualization import VisOpen3D
"""
Filter out everything in the depth image except for the object of interest.
TODO: Generate depth images for all frames in the rosbag.
"""
class DepthFilter:
    def __init__(self, frame_id):
        self.color_image_title = './rgb_data/frame00000{}.png'.format(frame_id)
        self.depth_image_title = './filtered_data/depth_without_robot_frame00000{}.png'.format(frame_id)
        self.cube_screen_image_dir = "./cube_data/screen_image_frame00000{}.png".format(frame_id)
        self.cube_depth_image_dir = "./cube_data/depth_image_frame00000{}.png".format(frame_id)
        self.extrinsic = self.setup_extrinsic()
        self.K = np.array([[380.2484436035156, 0, 314.2138977050781],
                    [0, 379.8265380859375, 240.59800720214844], 
                    [0, 0, 1]])
        self.pcd = self.generate_pcd()
        self.cropped_pcd = self.crop()

    def setup_extrinsic(self):
        # Setup camera extrinsic
        translation = np.array([[1.14164360], [0.15815239], [0.66422200]])
        axis_vec = [-1.57165949, -1.63112887, 1.07928078]
        angle = np.linalg.norm(axis_vec)
        axis = axis_vec / angle
        rotation = axis_angle_to_rotation_matrix(axis, angle) # directions of the world-axes in camera coordinates
        rotation_prime = rotation.T #USE THIS
        translation_prime = -rotation_prime @ translation #USE THIS
        extrinsic = np.vstack((np.hstack((rotation_prime, translation_prime)), np.array([0,0,0,1])))
        return extrinsic

    def generate_pcd(self):
        depth = o3d.io.read_image(self.depth_image_title)
        color = o3d.io.read_image(self.color_image_title)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
        # Setup camera intrinsic
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=380.2484436035156, fy=379.8265380859375, cx=314.2138977050781, cy=240.59800720214844)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=pinhole_camera_intrinsic, extrinsic=self.extrinsic)
        return pcd
    
    def crop(self):
        # Setup the orientatin of point cloud
        pcd_obb = self.pcd.get_oriented_bounding_box()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0,0,0])
        mesh_frame.translate(pcd_obb.center)
        # o3d.visualization.draw_geometries([pcd, mesh_frame])

        left_bottom, right_bottom, left_top, right_top = get_table_world_coordinates()
        left_bottom_pcl = np.array((left_bottom[0], left_bottom[1], left_bottom[2]))
        right_bottom_pcl = np.array((right_bottom[0], right_bottom[1], right_bottom[2]))
        left_top_pcl = np.array((left_top[0], left_top[1], left_top[2]))
        right_top_pcl = np.array((right_top[0], right_top[1], right_top[2]))
        left_bottom_pcl_ = np.array((left_bottom[0], left_bottom[1], left_bottom[2]-CUBE_LENGTH))
        right_bottom_pcl_ = np.array((right_bottom[0], right_bottom[1], right_bottom[2]-CUBE_LENGTH))
        left_top_pcl_ = np.array((left_top[0], left_top[1], left_top[2]-CUBE_LENGTH))
        right_top_pcl_ = np.array((right_top[0], right_top[1], right_top[2]-CUBE_LENGTH))
        corners = np.array([left_bottom_pcl, right_bottom_pcl, left_top_pcl, right_top_pcl, left_bottom_pcl_, right_bottom_pcl_, left_top_pcl_, right_top_pcl_])
        # print('x:', left_bottom_pcl[0], right_bottom_pcl[0], left_top_pcl[0], right_top_pcl[0], left_bottom_pcl_[0], right_bottom_pcl_[0], left_top_pcl_[0], right_top_pcl_[0])
        # print('y:', left_bottom_pcl[1], right_bottom_pcl[1], left_top_pcl[1], right_top_pcl[1], left_bottom_pcl_[1], right_bottom_pcl_[1], left_top_pcl_[1], right_top_pcl_[1])
        # print('z:', left_bottom_pcl[2], right_bottom_pcl[2], left_top_pcl[2], right_top_pcl[2], left_bottom_pcl_[2], right_bottom_pcl_[2], left_top_pcl_[2], right_top_pcl_[2])

        bounding_polygon = corners.astype("float64")
        coord_obb = mesh_frame.get_axis_aligned_bounding_box().get_oriented_bounding_box()
        # Rotate oriented_bounding_box to mesh_frame
        bounding_points = o3d.utility.Vector3dVector(bounding_polygon)
        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bounding_points)
        relative_rotation = coord_obb.R @ np.linalg.inv(oriented_bounding_box.R)
        oriented_bounding_box.rotate(R=relative_rotation, center=oriented_bounding_box.center)
        # Before cropping
        # o3d.visualization.draw_geometries([self.pcd, oriented_bounding_box, mesh_frame])
        cropped_pcd = self.pcd.crop(oriented_bounding_box)
        # After cropping
        # o3d.visualization.draw_geometries([cropped_pcd, oriented_bounding_box, mesh_frame])
        # pick_points(pcd)
        return cropped_pcd
    
    def visualize_depth_image(self):
        w = 640
        h = 480
        window_visible = True
        vis = VisOpen3D(width=w, height=h, visible=window_visible)
        vis.add_geometry(self.cropped_pcd)
        intrinsic_matrix = self.K
        extrinsic_matrix = self.extrinsic
        vis.load_view_point(intrinsic_matrix, extrinsic_matrix)
        depth = vis.capture_depth_float_buffer(show=False)
        image = vis.capture_screen_float_buffer(show=False)
        vis.capture_screen_image(self.cube_screen_image_dir)
        vis.capture_depth_image(self.cube_depth_image_dir)
        # vis.draw_camera(intrinsic_matrix, extrinsic_matrix, scale=0.5, color=[0.8, 0.2, 0.8])
        vis.run()
        # vis.destroy_window() #This causes segmentation fault.
        # del vis

def main(frame_id):
    # depth_image_title = './aligned_data/frame000001.png'
    # color_image_title = './rgb_data/frame00000{}.png'.format(frame_id)
    # depth_image_title = './filtered_data/depth_without_robot_frame00000{}.png'.format(frame_id)
    # cube_screen_image_dir = "./cube_data/screen_image_frame00000{}.png".format(frame_id)
    # cube_depth_image_dir = "./cube_data/depth_image_frame00000{}.png".format(frame_id)
    # color_image_title = './rgb_without_robot.png'
    system = DepthFilter(frame_id)
    # Visualize the depth image
    system.visualize_depth_image()

if __name__ == "__main__":
    start_frame_id = 1
    end_frame_id = 10
    for frame_id in range(start_frame_id, end_frame_id):
        main(frame_id)