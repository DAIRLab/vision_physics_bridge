import numpy as np

'''
Filter the simulated image from the real depth image
'''
def filter(real_img, sim_img):
    masked_img = np.subtract(real_img, sim_img)
    return masked_img

'''
Load data generated from rosbag
'''
def import_data(img_file, position_file, velocity_file):
    images = np.loadtxt(img_file)
    positions = np.loadtxt(position_file)
    velocities = np.loadtxt(velocity_file)
    return images, positions, velocities
