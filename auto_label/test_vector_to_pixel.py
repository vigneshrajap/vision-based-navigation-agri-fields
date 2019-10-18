#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:46:42 2019

@author: marianne
"""
import pytest
from ocam_camera_model_tools import OcamCalibCameraModel
import os

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

def test_vector_to_pixel_correct_quadrant(example_calib_file,example_point):
    cam_model = OcamCalibCameraModel(example_calib_file)
    pixel = cam_model.vector_to_pixel(example_point)
    assert in_upper_right_quadrant(pixel,cam_model.width,cam_model.height)