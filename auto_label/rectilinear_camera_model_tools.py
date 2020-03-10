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
        
class RectiLinearCameraModel:
    '''
    def __init__(self, width, height, verticalFOV,
		centreX=None, centreY=None, imageCircleFOV=0):
        self.imageCircleFOV = imageCircleFOV
        self.verticalFOV = verticalFOV
        if(centreX is None): 
        self.xc = width*0.5
        self.yc = height*0.5
        self.focalLengthPixels = (height * 0.5) / tan(verticalFOV * 0.5)
        R = _focalLengthPixels * tan(imageCircleFOV * 0.5)
        if (imageCircleFOV <= 0):
            R = width + height; // allows everything
        self.imageCircleR2 = R * R
    '''
    def __init__(self,calib_file):
        param_dict = self.read_opencv_storage_from_file(calib_file)
        self.set_params(param_dict)

        #Derived parameters
        self.focalLengthPixels = (self.height * 0.5) / tan(self.verticalFOV * 0.5)
        R = self.focalLengthPixels * tan(self.imageCircleFOV * 0.5)
        if (self.imageCircleFOV <= 0):
            R = self.width + self.height; # allows everything
        self.imageCircleR2 = R * R

    def set_params(self,opencv_storage_dict):
        #Extract parameters from opencv storage dictionary object
        d = opencv_storage_dict
        self.xc = float(d['centreX'])
        self.yc = float(d['centreX'])
        self.imageCircleFOV = float(d['imageCircleFOV'])
        self.verticalFOV = float(d['verticalFOV'])
        self.width = float(d['width'])
        self.height = float(d['height'])

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
        s = self.focalLengthPixels / point[3]
        dx = point[0] * s
        dy = point[1]* s
        x = dx + self.xc
        y = dy + self.yc
        R_squared = dx**2 + dy**2
        return x, y, R_squared

    def pixel_to_vector(self, x, y):
        dx = x - self.xc
        dy = y - self.yc
        direction = np.array([0,0,0])
        direction[0] = dy
        direction[1] = dx
        direction.z = self.focalLengthPixels
        
        return direction


if __name__ == "__main__":
    calib_file = os.path.join('../camera_data_collection/basler_2019-09-30-ocam_calib.xml')
    cam_model = RectiLinearCameraModel(calib_file)
    #vector to point debug
    point = [0.1,0.1,1]
    pixel = cam_model.vector_to_pixel(point)
    print(pixel)
    
