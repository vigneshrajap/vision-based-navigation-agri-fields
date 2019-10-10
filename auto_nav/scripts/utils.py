#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:20:10 2018

@author: marianne
"""
# ROS utilities

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension, Float32

def convert_numpy_to_Float32MultiArray(np_array):
    #Convert from numpy array to Float32MultiArray
    dim_arr = []
    print('pred_shape: ', np_array)
    for d in np_array.shape:
        dim_arr.append(MultiArrayDimension(size=d))
    layout = MultiArrayLayout(dim_arr,0)
    data = np_array.flatten()
    return Float32MultiArray(layout,data) 

# Open CV stuff

def disp_img(image,window_name = "image"):
    cv2.imshow(window_name,image)
    cv2.waitKey(1)
  
def recv_image_msg(image_msg,format = "passthrough"):
    cv_br = CvBridge()
    #rospy.loginfo('Receiving image')
    image = cv_br.imgmsg_to_cv2(image_msg,format)
    return image

def crop_img(image,crop_factor):
    #Crop image with crop factor (centered)
    height, width, channels = image.shape
    new_im_size = (np.round(np.array([height*crop_factor[0], width*crop_factor[1]]))).astype(int)
    offset = (np.round((np.array([height, width])-new_im_size)/2)).astype(int)
    crop_im = image[offset[0]:new_im_size[0]+offset[0],offset[1]:new_im_size[1]+offset[1]]
    return crop_im
