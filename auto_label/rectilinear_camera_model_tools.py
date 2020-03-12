#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on March 10 2020

@author: marianne

Translated from C++ code by Richard Moore

"""
import os
import numpy as np
import xmltodict
import math
        
class RectiLinearCameraModel:

    def __init__(self,calib_file):
        param_dict = self.read_opencv_storage_from_file(calib_file)
        self.set_params(param_dict)

        #Derived parameters
        self.focalLengthPixels = (self.height * 0.5) / math.tan(self.verticalFOV * 0.5)
        R = self.focalLengthPixels * math.tan(self.imageCircleFOV * 0.5)
        if (self.imageCircleFOV <= 0):
            R = self.width + self.height; # allows everything
        self.imageCircleR2 = R * R

    def set_params(self,opencv_storage_dict):
        #Extract parameters from opencv storage dictionary object
        d = opencv_storage_dict
        self.xc = float(d['centreX'])
        self.yc = float(d['centreY'])
        self.imageCircleFOV = float(d['imageCircleFOV'])
        self.verticalFOV = float(d['verticalFOV'])
        self.width = int(d['width'])
        self.height = int(d['height'])

    def read_opencv_storage_from_file(self,calib_file):
        with open(calib_file) as fd:
            dict_ = xmltodict.parse(fd.read())
            model_dict = dict_['opencv_storage']['cam_model']
        return model_dict

    def vector_to_pixel(self, point):
        '''
        Go from vector (in camera coordinates) to pixel (image coordinates)
        input: point (list) - x,y,z in camera frame
        '''
        s = self.focalLengthPixels / point[2]
        dx = point[0] * s
        dy = point[1] * s
        x = dx + self.xc
        y = dy + self.yc
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)

        R_squared = dx**2 + dy**2
        return x, y, R_squared

    def pixel_to_vector(self, x, y):
        #NOTE: model (x,y) is (height,width) so we swap
        dx = x - self.xc
        dy = y - self.yc
        direction = np.array([0,0,0])
        direction[0] = dx
        direction[1] = dy
        direction[2] = self.focalLengthPixels
        
        return direction


if __name__ == "__main__":
    calib_file = os.path.join('../camera_data_collection/realsense_model.xml')
    cam_model = RectiLinearCameraModel(calib_file)
    #vector to point debug
    point = [0,1,5]
    pixel = cam_model.vector_to_pixel(point)
    print(pixel)
    
