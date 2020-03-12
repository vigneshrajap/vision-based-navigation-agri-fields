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
import csv
import glob

def blend_color_and_image(image,mask,color_code=[0,255,0],alpha=0.5):
    #Blend colored mask and input image
    #Input:
    #   3-channel image (numpy array)
    #   1-channel (integer) mask

    #convert input to uint8 image
    if image.dtype is np.dtype('float32') or np.dtype('float64') and np.max(image) <= 1:
        image = np.uint8(image*255)

    mask = np.tile(mask[:,:,np.newaxis],[1,1,3])
    #convert nan values to zero
    mask = np.nan_to_num(mask)
    blended_im = np.uint8((mask * (1-alpha) * color_code) + (mask * alpha * image) + (np.logical_not(mask) * image)) #mask + image under mask + image outside mask
    return blended_im

def read_robot_offset_from_file(filename,row_ind = None):
    with open(filename) as f:
        a = np.array(list(csv.reader(f, delimiter = '\t')))
        a = a[1:]
        a = np.array(a, dtype=float)

    if row_ind is not None:
        frames = list(a[:,1])
        try:
            ind = frames.index(float(row_ind))
            lateral_offset = a[ind,2]
            angular_offset = a[ind,3]
        except ValueError:
            print('Frame index ', str(row_ind), ' not in list')
            lateral_offset = None
            angular_offset = None
    else:
        lateral_offset = a[:,2]
        angular_offset = a[:,3]
        
    return lateral_offset, angular_offset

if __name__ == "__main__":
    #Make image mask for a folder of images and their robot position data
    #--- Common setup
    #Directories
    image_dir = os.path.join('../Frogn_Dataset/images_prepped_train')
    output_dir = os.path.join('output','field_mask_all_images')
   
    #Camera model
    calib_file = os.path.join('../camera_data_collection/realsense_model_cropped.xml')
    cam_model = RectiLinearCameraModel(calib_file)

    #Tmp: debug without robot position
    use_robot_offset = False 

    #--- Per prefix
    #Set up field mask
    rec_prefix = '20191010_L1_N'
    robot_offset_file = os.path.join('../Frogn_Dataset/training_images_offset_20191010_L1_N.txt')
    lane_spacing = 1.4
    lane_duty_cycle = 0.5

    #Define field mask
    polygon_field_mask = make_field_mask(lane_spacing = lane_spacing, lane_duty_cycle = lane_duty_cycle, labels = [0,1,0,1,0], extent = 5) #read from file?

    #--- For each image
    for im_path in glob.iglob(os.path.join(image_dir,rec_prefix+'*')):
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        frame_ind = im_name[-4:]
        lateral_offset, angular_offset = read_robot_offset_from_file(robot_offset_file,frame_ind)

        #Camera setup #fixme read from urdf
        camera_xyz = np.array([0.749, 0.033, 1.242])
        camera_rpy = np.array([0.000, -0.332, 0.000]) 
        
        #Robot position 
        if use_robot_offset is True:
            robot_rpy = [0,0,angular_offset] 
            robot_xyz = [0,lateral_offset,0]
        else:
            robot_rpy = [0,0,0]
            robot_xyz = [0,0,0]

        #Set up transforms
        T_robot_to_world = set_up_robot_to_world_transform(rpy = robot_rpy, xyz = robot_xyz)
        T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)
        T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
        
        #Make image mask
        mask_with_index_and_label = make_image_mask_from_polygons(cam_model, polygon_field_mask, T_cam_to_world)
        label_mask = mask_with_index_and_label [:,:,1]
        #fixme replace nan values with "no_value" class

        #Visualize on top of example image
        camera_im = plt.imread(im_path)
        overlay_im = blend_color_and_image(camera_im,label_mask,color_code = [0,255,0],alpha=0.7) 

        #Save visualization and numpy array
        plt.imsave(os.path.join(output_dir,im_name) + '.png', overlay_im)
        np.save(os.path.join(output_dir,im_name),mask_with_index_and_label)
