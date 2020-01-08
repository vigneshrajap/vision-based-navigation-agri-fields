#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:29:13 2019

@author: marianne
"""
from ocam_camera_model_tools import OcamCalibCameraModel
import numpy as np
import os
import matplotlib.pyplot as plt
    
def make_line_samples(start,stop,numsamples):
    samples = []
    for coord in zip(start,stop):
        samples.append(np.linspace(coord[0],coord[1],numsamples))
    samples_per_point = map(list, zip(*samples)) #"transpose the coordinate list"
    return samples_per_point

def lines_to_camera_pixels(cam_model,xyz_start, xyz_stop):
    line_points = make_line_samples(xyz_start,xyz_stop,10000)
    print('line_points', line_points)
    
    pixels = []
    for p in line_points:
        new_pixel = np.array(cam_model.vector_to_pixel(p)[0:2])
        
        if(new_pixel[0] < cam_model.height and new_pixel[1] < cam_model.width): #if within image boundaries
            pixels.append(new_pixel)
    return np.array(pixels)
if __name__ == "__main__":
     calib_file = os.path.join('/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/auto_nav/scripts/input_cam_model_campus_2018-08-31.xml')
     cam_model = OcamCalibCameraModel(calib_file)
     #FIXME: need to transform the points from world coordinates to camera coordinates!
     xyz_start = [0,1,0]
     xyz_stop = [0,1,10]
     line_pixels = lines_to_camera_pixels(cam_model,xyz_start,xyz_stop)
     print('line_pixels',line_pixels)
     
     plt.figure(2)
     plt.plot(line_pixels[:,0],line_pixels[:,1])
     plt.show()
     
     mask = np.zeros((cam_model.height,cam_model.width))
     ind = line_pixels
     mask[ind[:,0],ind[:,1]] = 1
     
     print(mask)
     plt.figure(1)
     plt.colorbar
     plt.imshow(mask*255) #too thin, not visible...
     
     #plot section only
     plt.figure(2)
     plt.imshow(mask[1000:1100,2000:2100])
     

         
     
