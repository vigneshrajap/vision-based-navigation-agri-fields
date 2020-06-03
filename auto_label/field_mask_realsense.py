#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from rectilinear_camera_model_tools import RectiLinearCameraModel
import os
from tqdm import tqdm
from field_mask import *
import cv2

if __name__ == "__main__":
    #%% Demo code: Hot to make an image field mask
    
    #Dummy field mask
    polygon_field_mask = make_field_mask(widths = [0.6,0.6,0.6], labels = [1,0,1], extent = 5)
    
    #%%Transformations 
    #Realsense model
    calib_file = os.path.join('../camera_data_collection/realsense_model_cropped.xml')
    cam_model = RectiLinearCameraModel(calib_file)

    #Camera setup
    camera_xyz = [0.749, 0.033, 1.242] #fixme read from urdf
    #camera_xyz = [0.0, 0.0, 0.8] 
    #camera_rpy = [0.000, -0.332, 0.000] #radians, opposite sign of pitch as in urdf?
    camera_rpy = [0.000, -0.332, 0.000]
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)

    
    #Robot position 
    #dummy values, should get values from outside
    lateral_offset = 0.0134
    yaw = 0 #-0.1078
    robot_rpy = [0,0,yaw]
    robot_xyz = [0,lateral_offset,0]
    T_robot_to_world = set_up_robot_to_world_transform(rpy = robot_rpy, xyz = robot_xyz)
    
    T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
    image_mask = make_image_mask_from_polygons(cam_model, polygon_field_mask, T_cam_to_world)
        
    plt.figure(10)
    plt.imshow(image_mask[:,:,0])  
    plt.figure(11)          
    plt.imshow(image_mask[:,:,1])

    #Visualize on top of example image
    camera_im = plt.imread(r'../Frogn_Dataset/images_prepped_train/20191010_L1_N_0185.png')
    plt.figure(12)
    plt.imshow(camera_im)

    mask = np.uint8(np.zeros((image_mask.shape[0], image_mask.shape[1],3)))
    mask[:,:,1]=image_mask[:,:,1]
    overlay_im = camera_im + 0.2*mask
    plt.figure(13)
    plt.imshow(overlay_im)
    plt.show()
