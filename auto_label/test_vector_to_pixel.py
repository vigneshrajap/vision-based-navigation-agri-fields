#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:46:42 2019

@author: marianne
"""
import pytest
from ocam_camera_model_tools import OcamCalibCameraModel
import os
import numpy as np

@pytest.fixture
def example_calib_file():
    calib_file = os.path.join('/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/auto_nav/scripts/input_cam_model_campus_2018-08-31.xml')
    return calib_file

@pytest.fixture
def example_upper_right_quadrant_point():
    point = [1.5,-0.5,1]
    return point

def in_upper_right_quadrant(pixel,width,height):
    return (pixel[0] > float(width)/2 and pixel[1] < float(height)/2)

def test_vector_to_pixel_correct_quadrant(example_calib_file,example_upper_right_quadrant_point):
    cam_model = OcamCalibCameraModel(example_calib_file)
    pixel = cam_model.vector_to_pixel(example_upper_right_quadrant_point)
    assert in_upper_right_quadrant(pixel,cam_model.width,cam_model.height)
    
def test_pixel_to_vector_correct_quadrant(example_calib_file):
    cam_model = OcamCalibCameraModel(example_calib_file)
    width = cam_model.width
    height = cam_model.height
    middle_pixel =np.round(np.array([width/2, height/2]))
    lr_pixel = [width,height]
    ul_pixel = [0,0]
    
    middle_vector = cam_model.pixel_to_vector(middle_pixel[0],middle_pixel[1])
    lr_vector = cam_model.pixel_to_vector(lr_pixel[0],lr_pixel[1])
    ul_vector = cam_model.pixel_to_vector(ul_pixel[0],ul_pixel[1])
    
    print('Middle point', middle_vector)
    print('Lower right point', lr_vector)
    print('Upper left point', ul_vector)
    
    
    assert lr_vector[0] > middle_vector [0] and ul_vector[0] < middle_vector[0] and lr_vector[1] > middle_vector[1] and ul_vector[1] < middle_vector[1]
    
def test_pixel_to_vector_positive_z(example_calib_file):
    cam_model = OcamCalibCameraModel(example_calib_file)
    width = cam_model.width
    height = cam_model.height
    middle_pixel =np.round(np.array([width/2, height/2]))
    lr_pixel = [width,height]
    ul_pixel = [0,0]
    
    middle_vector = cam_model.pixel_to_vector(middle_pixel[0],middle_pixel[1])
    lr_vector = cam_model.pixel_to_vector(lr_pixel[0],lr_pixel[1])
    ul_vector = cam_model.pixel_to_vector(ul_pixel[0],ul_pixel[1])
    
    print('Middle point', middle_vector)
    print('Lower right point', lr_vector)
    print('Upper left point', ul_vector)
    assert middle_vector[2] > 0 and lr_vector[2] > 0 and ul_vector[2] > 0
    