#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from rectilinear_camera_model_tools import RectiLinearCameraModel
from utilities import read_robot_offset_from_file, read_row_spec_from_file
import os
from field_mask import *
import csv
import glob
import re
from tqdm import tqdm
import argparse
import sys
sys.path.append('..')
from visualization_utilities import blend_color_and_image

def run_field_mask(image_dir = 'images_prepped_train',
    output_dir = 'output',
    calib_file = os.path.join('../camera_data_collection/realsense_model_cropped.xml'),
    robot_offset_file = None,
    row_spec_file = None,
    use_robot_offset = True,
    sampling_step = None,
    debug = False,
    visualize = False):

    #setup
    cam_model = RectiLinearCameraModel(calib_file)

    ann_dir =  os.path.join(output_dir,'annotation_images')
    os.makedirs(ann_dir, exist_ok = True)
    vis_dir = os.path.join(output_dir,'visualization')
    os.makedirs(vis_dir, exist_ok = True)
    arr_dir = os.path.join(output_dir,'annotation_arrays')
    os.makedirs(arr_dir, exist_ok = True)
    
    #Corrections
    #lateral_correction = 0.13
    delay_correction = 30 #L3S slalom #image frame vs offset correction

    #--- Per prefix
    #Set up field mask
    #for robot_offset_file in glob.iglob(robot_offset_dir):
    rec_prefix = os.path.basename(robot_offset_file)[:-12]
    print('Processing ', rec_prefix)
    try:
        crop_duty_cycle, lane_spacing = read_row_spec_from_file(row_spec_file,rec_prefix)
    except:
        print('Error: Could not read rowspec file')
    if lane_spacing is None:
        print('Error: Could not read lane spec from rowspec file')
    #Define field mask
    polygon_field_mask = make_field_mask(lane_spacing = lane_spacing, crop_duty_cycle = crop_duty_cycle, labels = [0,1,0,1,0], extent = 5) #read from file?

    #--- For each image in the specified folder
    files = os.listdir(image_dir)
    pat = re.compile(rec_prefix + '_\d\d\d\d.png',re.UNICODE)  
    for im_file in tqdm(filter(pat.match, files)):         
    #for im_path in glob.iglob(os.path.join(image_dir,rec_prefix+'*')):
        print(im_file)
        im_name = os.path.splitext(os.path.basename(im_file))[0]
        frame_ind = int(im_name[-4:])

        frame_ind = frame_ind - delay_correction #subtract delay

        try: 
            lateral_offset, angular_offset,_ = read_robot_offset_from_file(robot_offset_file,frame_ind)

            #lateral_offset = lateral_offset - lateral_correction #south #should be fixed in camera coordinate frame
            #lateral_offset = lateral_offset + lateral_correction #north

            #Camera setup #fixme read from urdf
            #camera_xyz = np.array([0.749, 0.033, 1.242]) #measured
            #camera_xyz = np.array([0.749, 0.033, 1.1]) #adjusted
            camera_xyz = np.array([0, 0.033, 1.1]) #zero y offset
            #camera_xyz = np.array([0, 0.033 - lateral_correction, 1.1]) #zero y offset, including lateral correction
            #camera_xyz = np.array([0.0, 0.0, 1.1]) #zero xy offset
            #camera_rpy = np.array([0.000, -0.332, 0.000]) #measured
            camera_rpy = np.array([0.000, -0.4, 0.0]) #adjusted
            #camera_rpy = np.array([0.000, -0.4, 0]) #zero yaw

            #compensate for wrong sign
            camera_rpy[2] = -camera_rpy[2]

            #Robot position 
            if use_robot_offset is True:
                print(lateral_offset, angular_offset)
                robot_rpy = [0,0,-angular_offset] #compensate for wrong sign
                robot_xyz = [0,lateral_offset,0]
            else:
                robot_rpy = [0,0,0]
                robot_xyz = [0,0,0]

            #Set up transforms
            T_robot_to_world = set_up_robot_to_world_transform(rpy = robot_rpy, xyz = robot_xyz)
            T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)
            T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
            
            #Make image mask
            mask_with_index_and_label = make_image_mask_from_polygons(cam_model, polygon_field_mask, T_cam_to_world,sampling_step = sampling_step)
            #prepare label mask for saving and visualization
            label_mask = mask_with_index_and_label[:,:,1] #extract second channel
            label_mask = label_mask + 1 #shift from 0 and 1 to 1 and 2
            label_mask = np.nan_to_num(label_mask).astype('uint8')

            #Visualize on top of example image
            camera_im = plt.imread(os.path.join(image_dir,im_file))
            #upsample back to original image size
            if label_mask.shape != camera_im.shape[0:2]:
                label_im = Image.fromarray(label_mask,mode='L')
                label_im = label_im.resize((camera_im.shape[1],camera_im.shape[0]))
                label_mask = np.array(label_im)
            
            if debug or visualize:
                overlay_im = blend_color_and_image(camera_im,label_mask,color_codes = [[None,None,None],[0,0,255],[255,255,0]],alpha=0.85) 
                #Save visualization and numpy array
                #plt.imsave(os.path.join(output_dir,'visualisation',im_name) + 'lat' + str(lateral_offset) + 'ang' + str(angular_offset) + '.png', overlay_im)
                plt.imsave(os.path.join(vis_dir,im_name)+'.png', overlay_im)
            
            if not debug:
                #Save annotiations alone as images and arrays
                plt.imsave(os.path.join(ann_dir,im_name)+'.png',label_mask)
                np.save(os.path.join(arr_dir,im_name),label_mask)
        except ValueError:
            print('Frame index ', str(frame_ind), ' not in list')

def main():
    parser = argparse.ArgumentParser(description="Make field mask for specified image and position data")
    parser.add_argument("--dataset_dir", default = './output', help = "Base data directory")
    parser.add_argument("--image_dir",default = 'images_only', help = "Input image directory. Relative to dataset dir")
    parser.add_argument("--output_dir", default = 'automatic_annotations', help = "Output directory for results. Relative to dataset_dir")
    parser.add_argument("--robot_offset_file", help = "Mandatory, even if use_robot_offset is False. Specify row to process through robot offset file. Relative to dataset dir"),
    parser.add_argument("--use_robot_offset", default = True, help = "If off: Assume straight driving."),
    parser.add_argument("--camera_calib_file", default = '../camera_data_collection/realsense_model.xml', help = "Camera model used for projecting the mask")
    parser.add_argument("--sampling_step", default = 4, help = "Subsampling factor of image mask to increase speed")
    parser.add_argument("--debug", default = False, help = "Debug flag: When in debug mode, only save visualisation")
    parser.add_argument("--visualize", default = True, help = "Enable saving of visualization outside debug mode.")

    args = parser.parse_args()
    #Run field mask on the specified images, camera model and dataset directory:
    run_field_mask(image_dir = os.path.join(args.dataset_dir, args.image_dir), 
    output_dir = os.path.join(args.dataset_dir, args.output_dir),
    calib_file = os.path.join(args.camera_calib_file),
    robot_offset_file = os.path.join(args.dataset_dir, args.robot_offset_file),
    use_robot_offset = args.use_robot_offset,
    row_spec_file = os.path.join(args.dataset_dir,'row_spec.txt'),
    sampling_step = np.uint8(args.sampling_step),
    debug = args.debug,
    visualize = args.visualize,
    )

if __name__ == "__main__":
    '''
    #Make image mask for a folder of images and their robot position data
    #Setup
    dataset_dir = os.path.join('/media/marianne/Seagate Expansion Drive/data/Frogn_Dataset')
    #dataset_dir = os.path.join('../Frogn_Dataset')
    image_dir = os.path.join(dataset_dir,'images_only')
    output_dir = os.path.join('output/robot_offset_experiments/offsetfix3_widercrop_delay30_smoothing50_meancomp')
    robot_offset_dir = os.path.join('output/robot_offset_experiments/offsetfix3_widercrop_delay30_smoothing50_meancomp/20191010*')
    #Camera model
    calib_file = os.path.join('../camera_data_collection/realsense_model.xml') #realsense model for "images only", cropped model for "prepped" images
    #Turn robot offset on/off
    use_robot_offset = True
    #Turn on subsampling of image mask
    sampling_step = 8
    '''

    main()



