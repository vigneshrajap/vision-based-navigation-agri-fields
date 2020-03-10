#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from ocam_camera_model_tools import OcamCalibCameraModel,vec3_normalise
import os
from tqdm import tqdm
from field_mask import *
from PIL import Image
#import cv2

if __name__ == "__main__":
    #%% Demo code: Hot to make an image field mask
    
    #Dummy field mask
    polygon_field_mask = make_field_mask(widths = [0.8,0.7,0.8], labels = [1,0,1], extent = 5)
    
    #%%Transformations 
    #Old camera model with adjustments for testing
    calib_file = os.path.join('../camera_data_collection/basler_2019-09-30-ocam_calib.xml')
    cam_model = OcamCalibCameraModel(calib_file)

    #Camera setup
    # <parent link="$(arg tf_prefix)/base_link"/> <child link="$(arg tf_prefix)/basler_camera" /> <origin xyz="0.75 0.0 1.055" rpy="0 0.279253 0" />
    camera_xyz = [0.0,0.0,1.055] #fixme read from urdf
    camera_rpy = [0,-0.279253,0]
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)

    
    #Robot position 
    #dummy values, should get values from outside
    robot_rpy = [0,0,0]
    robot_xyz = [0,0,0]
    T_robot_to_world = set_up_robot_to_world_transform(rpy = robot_rpy, xyz = robot_xyz)
    
    '''
    input_dim = [700,1000]
    T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
    image_mask = make_image_mask_from_polygons(cam_model, polygon_field_mask, T_cam_to_world, cropped_dims = input_dim)
        
    plt.figure(10)
    plt.imshow(image_mask[:,:,0])  
    plt.figure(11)          
    plt.imshow(image_mask[:,:,1])
    '''
    #Visualize on top of example image
    camera_im = Image.open(os.path.join('..','Frogn_Dataset','images_prepped_train'))
    camera_im = Image.toarray(camera_im)
    #camera_im = plt.imread(os.path.join('..','Frogn_Dataset','images_prepped_train'))
    plt.figure(12)
    plt.imshow(camera_im)

    mask = np.uint8(np.zeros((image_mask.shape[0], image_mask.shape[1],3)))
    mask[:,:,1]=image_mask[:,:,1]
    cropped_im = camera_im[camera_im.shape[0]/2-input_dim[0]/2:camera_im.shape[0]/2+input_dim[0]/2, camera_im.shape[1]/2-input_dim[1]/2:camera_im.shape[1]/2+input_dim[1]/2,:]
    #overlay_im = cv2.addWeighted(src1 = camera_im, alpha=1, src2=mask, beta=0.5, gamma =0, dst = camera_im)
    overlay_im = cropped_im + 0.2*mask
    plt.figure(13)
    plt.imshow(overlay_im)
    plt.show()
