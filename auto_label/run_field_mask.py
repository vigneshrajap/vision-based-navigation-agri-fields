#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from rectilinear_camera_model_tools import RectiLinearCameraModel
from utilities import read_robot_offset_from_file, read_row_spec_from_file, blend_color_and_image
import os
from tqdm import tqdm
from field_mask import *
import csv
import glob

def run_field_mask(dataset_dir = os.path.join('../Frogn_Dataset'),
    image_dir = 'images_prepped_train',
    output_dir = 'output',
    calib_file = os.path.join('../camera_data_collection/realsense_model_cropped.xml'),
    use_robot_offset = True):

    #setup
    cam_model = RectiLinearCameraModel(calib_file)

    #--- Per prefix
    #Set up field mask
    for robot_offset_file in glob.iglob(os.path.join(dataset_dir,'robot_offsets/*')):
        rec_prefix = os.path.basename(robot_offset_file)[:13]
        lane_duty_cycle, lane_spacing = read_row_spec_from_file(os.path.join(dataset_dir,'row_spec',rec_prefix)+'_row_spec.txt',rec_prefix)
        #Define field mask
        polygon_field_mask = make_field_mask(lane_spacing = lane_spacing, lane_duty_cycle = lane_duty_cycle, labels = [0,1,0,1,0], extent = 5) #read from file?

        #--- For each image in the specified folder
        for im_path in glob.iglob(os.path.join(image_dir,rec_prefix+'*')):
            im_name = os.path.splitext(os.path.basename(im_path))[0]
            frame_ind = im_name[-4:]
            lateral_offset, angular_offset,_ = read_robot_offset_from_file(robot_offset_file,frame_ind)

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
            plt.imsave(os.path.join(output_dir,'visualisation',im_name) + '.png', overlay_im)
            plt.imsave(os.path.join(output_dir,'annotations',im_name)+'.png',label_mask)
            np.save(os.path.join(output_dir,'arrays',im_name),mask_with_index_and_label)

if __name__ == "__main__":
    #Make image mask for a folder of images and their robot position data
    #Setup
    dataset_dir = os.path.join('../Frogn_Dataset')
    image_dir = os.path.join(dataset_dir,'images_prepped_train')
    output_dir = os.path.join('output')
    #Camera model
    calib_file = os.path.join('../camera_data_collection/realsense_model_cropped.xml')

    #Run field mask on the specified images, camera model and dataset directory:
    run_field_mask(dataset_dir=dataset_dir, image_dir=image_dir, output_dir=output_dir,
    calib_file=calib_file,use_robot_offset=False)



