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
def example_ur_quadrant_vector():
    vector = [0.1, -0.1, 1]
    return vector
@pytest.fixture
def example_lr_quadrant_vector():
    vector = [0.1, 0.1, 1]
    return vector
@pytest.fixture
def example_ul_quadrant_vector():
    vector = [-0.1, -0.1, 1]
    return vector
@pytest.fixture
def example_ll_quadrant_vector():
    vector = [-0.1, 0.1, 1]
    return vector


def in_ur_quadrant(pixel,width,height): #upper right
    print('UR pixel', pixel)
    print(width/2)
    print(height/2)
    return (pixel[0] > float(width)/2 and pixel[1] < float(height)/2)

def in_lr_quadrant(pixel,width,height): #lower right
    print('LR pixel', pixel)
    return (pixel[0] > float(width)/2 and pixel[1] > float(height)/2)

def in_ul_quadrant(pixel,width,height): #upper left
    print('UL pixel', pixel)
    return (pixel[0] < float(width)/2 and pixel[1] < float(height)/2)

def in_ll_quadrant(pixel,width,height): #lower left
    print('LL pixel', pixel)
    return (pixel[0] < float(width)/2 and pixel[1] > float(height)/2)

def test_vector_to_pixel_correct_quadrant(example_calib_file,
                                          example_ur_quadrant_vector,
                                          example_lr_quadrant_vector,
                                          example_ul_quadrant_vector,
                                          example_ll_quadrant_vector):
    cam_model = OcamCalibCameraModel(example_calib_file)
    ur_pixel = cam_model.vector_to_pixel(example_ur_quadrant_vector)
    lr_pixel = cam_model.vector_to_pixel(example_lr_quadrant_vector)
    ul_pixel = cam_model.vector_to_pixel(example_ul_quadrant_vector)
    ll_pixel = cam_model.vector_to_pixel(example_ll_quadrant_vector)
    assert in_ur_quadrant(ur_pixel,cam_model.width,cam_model.height) and in_lr_quadrant(lr_pixel,cam_model.width,cam_model.height) and in_ul_quadrant(ul_pixel,cam_model.width,cam_model.height) and in_ll_quadrant(ll_pixel,cam_model.width,cam_model.height)
    

    
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
    
    print('Middle vector', middle_vector)
    print('Lower right vector', lr_vector)
    print('Upper left vector', ul_vector)
    
    
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
    
    print('Middle vector', middle_vector)
    print('Lower right vector', lr_vector)
    print('Upper left vector', ul_vector)
    assert middle_vector[2] > 0 and lr_vector[2] > 0 and ul_vector[2] > 0
    