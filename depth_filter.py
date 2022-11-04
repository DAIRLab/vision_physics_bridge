import numpy as np
import open3d as o3d
from math_utils import axis_angle_to_rotation_matrix
from visualization import VisOpen3D
from tqdm import tqdm

"""
Filter out everything in the depth image except for the object of interest.
"""
TABLE_LENGTH = 182.8/100 #in meters
TABLE_WIDTH = 91.3/100
TABLE_HEIGHT = 73.2/100
ROBOT_TO_LENGTH = 46.9/100 # robot base to long side of table
ROBOT_TO_WIDTH = 33.7/100 # robot base to short side of table
ROBOT_TO_HEIGHT = 1/100 # height from robot base to table surface
CAMERA_X = 1.14164360
CAMERA_Y = 0.15815239
CAMERA_Z = 0.66422200
# CUBE_LENGTH = 10.2/100
# For new box 10/31/2022
BOX_LENGTH = 9.6/100
ROBOT_BASE_LENGTH = 12/100 # this is inaccurate
HEIGHT_BUFFER = 0.3 # We want to bound the cube when the robot grabs it in the air. This is a buffer for the end effector configuration space.
WIDTH_BUFFER = 0.3 # Need to make the left and right bounds larger than the table since the robot may move the cube wildly in space.
PLANK_WIDTH = 36.3/100

class DepthFilter:
    def __init__(self, frame_id):
        self.color_image_title = './rgb_data/%04i.png' % frame_id
        self.depth_image_title = './depth_data/%04i.png' % frame_id
        # self.color_image_title = './filtered_data/rgb_without_robot_frame%04i.png'%frame_id
        # self.depth_image_title = './filtered_data/depth_without_robot_frame%04i.png'%frame_id
        self.cube_screen_image_dir = "./cube_data/screen_image_frame%04i.png"%frame_id
        self.cube_depth_image_dir = "./cube_data/depth_image_frame%04i.png"%frame_id
        self.extrinsic = self.setup_extrinsic()
        self.K = np.array([[380.2484436035156, 0, 314.2138977050781],
                    [0, 379.8265380859375, 240.59800720214844], 
                    [0, 0, 1]])
        self.pcd = self.generate_pcd()
        self.cropped_pcd = self.crop()

    def setup_extrinsic(self):
        # Setup camera extrinsic
        # translation = np.array([[1.14164360], [0.15815239], [0.66422200]])
        # axis_vec = [-1.57165949, -1.63112887, 1.07928078]
        # For new data 10/31/2022
        translation = np.array([[1.11076422], [-0.07966290], [0.67947702]])
        axis_vec = [-1.61997882, -1.56988553, 0.86362178]
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

        left_bottom, right_bottom, left_top, right_top = get_bounding_box_world_coordinates()
        left_bottom_pcl = np.array((left_bottom[0], left_bottom[1], left_bottom[2]))
        right_bottom_pcl = np.array((right_bottom[0], right_bottom[1], right_bottom[2]))
        left_top_pcl = np.array((left_top[0], left_top[1], left_top[2]))
        right_top_pcl = np.array((right_top[0], right_top[1], right_top[2]))
        left_bottom_pcl_ = np.array((left_bottom[0], left_bottom[1], left_bottom[2]-BOX_LENGTH-HEIGHT_BUFFER))
        right_bottom_pcl_ = np.array((right_bottom[0], right_bottom[1], right_bottom[2]-BOX_LENGTH-HEIGHT_BUFFER))
        left_top_pcl_ = np.array((left_top[0], left_top[1], left_top[2]-BOX_LENGTH-HEIGHT_BUFFER))
        right_top_pcl_ = np.array((right_top[0], right_top[1], right_top[2]-BOX_LENGTH-HEIGHT_BUFFER))
        corners = np.array([left_bottom_pcl, right_bottom_pcl, left_top_pcl, right_top_pcl, left_bottom_pcl_, right_bottom_pcl_, left_top_pcl_, right_top_pcl_])
        
        bounding_polygon = corners.astype("float64")
        coord_obb = mesh_frame.get_axis_aligned_bounding_box().get_oriented_bounding_box()
        # Rotate oriented_bounding_box to mesh_frame
        bounding_points = o3d.utility.Vector3dVector(bounding_polygon)
        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bounding_points)
        relative_rotation = coord_obb.R @ np.linalg.inv(oriented_bounding_box.R)
        oriented_bounding_box.rotate(R=relative_rotation, center=oriented_bounding_box.center)
        oriented_bounding_box.color = np.array((1,0,0))
        
        # Forming another bounding box to filter out the plank
        plank_left_bottom, plank_right_bottom, plank_left_top, plank_right_top = get_plank_bounding_box_world_coordinates()
        plank_left_bottom_pcl = np.array((plank_left_bottom[0], plank_left_bottom[1], plank_left_bottom[2]))
        plank_right_bottom_pcl = np.array((plank_right_bottom[0], plank_right_bottom[1], plank_right_bottom[2]))
        plank_left_top_pcl = np.array((plank_left_top[0], plank_left_top[1], plank_left_top[2]))
        plank_right_top_pcl = np.array((plank_right_top[0], plank_right_top[1], plank_right_top[2]))
        plank_left_bottom_pcl_ = np.array((plank_left_bottom[0], plank_left_bottom[1], plank_left_bottom[2]-BOX_LENGTH-HEIGHT_BUFFER))
        plank_right_bottom_pcl_ = np.array((plank_right_bottom[0], plank_right_bottom[1], plank_right_bottom[2]-BOX_LENGTH-HEIGHT_BUFFER))
        plank_left_top_pcl_ = np.array((plank_left_top[0], plank_left_top[1], plank_left_top[2]-BOX_LENGTH-HEIGHT_BUFFER))
        plank_right_top_pcl_ = np.array((plank_right_top[0], plank_right_top[1], plank_right_top[2]-BOX_LENGTH-HEIGHT_BUFFER))
        corners_ = np.array([plank_left_bottom_pcl, plank_right_bottom_pcl, plank_left_top_pcl, plank_right_top_pcl, plank_left_bottom_pcl_, plank_right_bottom_pcl_, plank_left_top_pcl_, plank_right_top_pcl_])
        plank_bounding_polygon = corners_.astype("float64")
        plank_bounding_points = o3d.utility.Vector3dVector(plank_bounding_polygon)
        plank_oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(plank_bounding_points)
        relative_rotation_ = coord_obb.R @ np.linalg.inv(plank_oriented_bounding_box.R)
        plank_oriented_bounding_box.rotate(R=relative_rotation_, center=plank_oriented_bounding_box.center)
        plank_oriented_bounding_box.color = np.array((0,1,0))
        # Before cropping
        # o3d.visualization.draw_geometries([self.pcd, oriented_bounding_box, plank_oriented_bounding_box, mesh_frame])
        
        cropped_pcd = self.pcd.crop(oriented_bounding_box)
        cropped_pcd_ = self.pcd.crop(plank_oriented_bounding_box)
        if np.asarray(cropped_pcd_.points).shape[0] > np.asarray(cropped_pcd.points).shape[0]: 
            print("Cube is over the plank.")
            result = cropped_pcd_
        else:
            print("Cube is on the table.")
            result = cropped_pcd
        # After cropping
        # o3d.visualization.draw_geometries([result, plank_oriented_bounding_box, mesh_frame])
        # pick_points(pcd)
        return result
    
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
        # vis.run() #visualize the screen and depth images
        # vis.destroy_window() #This causes segmentation fault.
        # del vis

def get_bounding_box_world_coordinates():
    """
    Get bounding box's world coordinate with camera extrinsics. X is towards to camera. Y is towards to right of the camera(facing the robot). Z is upward. 
    Note: Assume robot is in the middle of the table.
    return: (a, b, c, d) that corresponds to the left bottom, right bottom, left top, right top corner of the table.
    """
    left_bottom = [CAMERA_X+0.41, -TABLE_WIDTH/2-WIDTH_BUFFER, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER]
    right_bottom = [CAMERA_X+0.41, TABLE_WIDTH/2+WIDTH_BUFFER, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER]
    left_top = [ROBOT_BASE_LENGTH, -TABLE_WIDTH/2-WIDTH_BUFFER, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER]
    right_top = [ROBOT_BASE_LENGTH, TABLE_WIDTH/2+WIDTH_BUFFER, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER]
    # left_bottom = [CAMERA_X+0.41, -TABLE_WIDTH/2-WIDTH_BUFFER, -ROBOT_TO_HEIGHT+CUBE_LENGTH+HEIGHT_BUFFER]
    # right_bottom = [CAMERA_X+0.41, TABLE_WIDTH/2+WIDTH_BUFFER, -ROBOT_TO_HEIGHT+CUBE_LENGTH+HEIGHT_BUFFER]
    # left_top = [ROBOT_BASE_LENGTH-0.3, -TABLE_WIDTH/2-WIDTH_BUFFER, -ROBOT_TO_HEIGHT+CUBE_LENGTH+HEIGHT_BUFFER]
    # right_top = [ROBOT_BASE_LENGTH-0.3, TABLE_WIDTH/2+WIDTH_BUFFER, -ROBOT_TO_HEIGHT+CUBE_LENGTH+HEIGHT_BUFFER]
    return left_bottom, right_bottom, left_top, right_top

def get_plank_bounding_box_world_coordinates():
    """
    Get the world coordinates of the bounding box of the plank.
    """
    buffer = 0.3
    HEIGHT_BUFFER_= 0.4
    left_bottom = [CAMERA_X+0.41, -TABLE_WIDTH/2-buffer, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER_]
    right_bottom = [CAMERA_X+0.41, TABLE_WIDTH/2+buffer, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER_]
    left_top = [ROBOT_BASE_LENGTH-0.3, -TABLE_WIDTH/2-buffer, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER_]
    right_top = [ROBOT_BASE_LENGTH-0.3, TABLE_WIDTH/2+buffer, -ROBOT_TO_HEIGHT+BOX_LENGTH+HEIGHT_BUFFER_]
    return left_bottom, right_bottom, left_top, right_top

def main(frame_id):
    system = DepthFilter(frame_id)
    system.visualize_depth_image()

if __name__ == "__main__":
    start_frame_id = 1
    end_frame_id = 3372
    for frame_id in tqdm(range(start_frame_id, end_frame_id)):
        main(frame_id)