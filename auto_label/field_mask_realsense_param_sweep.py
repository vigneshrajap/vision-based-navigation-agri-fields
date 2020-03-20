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
    polygon_field_mask = make_field_mask(widths = [0.7,0.7,0.7], labels = [1,0,1], extent = 5)
    
    #%%Transformations 
    #Realsense model
    calib_file = os.path.join('../camera_data_collection/realsense_model_cropped.xml')
    cam_model = RectiLinearCameraModel(calib_file)

    #Camera setup #fixme read from urdf
    camera_xyz = np.array([0.749, 0.033, 1.242])
    camera_rpy = np.array([0.000, -0.332, 0.000])

    #cam_height_sweep = camera_xyz[2] + np.array([0, -0.1, -0.2, -0.4])
    #cam_pitch_sweep = camera_rpy[1] + np.array([-0.1, -0.05, 0, 0.05, 0.1])
    cam_height_sweep = [camera_xyz[2]]
    cam_pitch_sweep = [camera_rpy[1]]

    #Robot position and angle
    lateral_offset = 0.0134
    yaw = 0 #-0.1078

    yaw_sweep = [-0.1, -0.05, 0, 0.05, 0.1]

    for p in cam_pitch_sweep:
        for h in cam_height_sweep:
            for y in yaw_sweep:
                camera_xyz[2] = h
                camera_rpy[1] = p
                yaw = y

                T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)
                
                #Robot position #read from ROS topic or txt file
                robot_rpy = [0,0,yaw]
                robot_xyz = [0,lateral_offset,0]
                T_robot_to_world = set_up_robot_to_world_transform(rpy = robot_rpy, xyz = robot_xyz)
                
                T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
                image_mask = make_image_mask_from_polygons(cam_model, polygon_field_mask, T_cam_to_world)

                #Visualize on top of example image
                camera_im = plt.imread(r'../Frogn_Dataset/images_prepped_train/20191010_L1_N_0185.png')
                plt.figure(12)
                plt.imshow(camera_im)

                mask = np.uint8(np.zeros((image_mask.shape[0], image_mask.shape[1],3)))
                mask[:,:,1]=image_mask[:,:,1]
                cropped_im = camera_im
                overlay_im = cropped_im + 0.2*mask
                '''
                plt.figure(13)
                plt.imshow(overlay_im)
                plt.show()
                '''
                plt.imsave(os.path.join('output','p' + str(p)+'_h' + str(h) + 'yaw'+str(y) +'20191010_L1_N_0185.png'),overlay_im)
