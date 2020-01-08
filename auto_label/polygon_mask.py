#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from ocam_camera_model_tools import OcamCalibCameraModel
import os

'''
class Rectangle():
    def __init__(self,width,height,position):
        #position: x,y coordinate of lower left corner
        self.w = width
        self.h = height
        self.points = np.array([position,
                                 position + np.array([0,self.h]), 
                                 position + np.array([self.w,self.h]),
                                 position + np.array([self.w,0])])
'''

class Polygon():
    #Polygon represented by list of points
    #Points are ordered counter clockwise
    def __init__(self,points,label=1):
        self.label = label
        self.points = points
        
    def plot(self):
        #plt.figure()
        plt_indeces = np.append(range(self.points.shape[0]),0)
        plt.plot(self.points[plt_indeces,0],self.points[plt_indeces,1])
        plt.axis('scaled')
        
    def is_inside():
        #check if a point is inside
        #geometrisk predikat (orient2D)
        pass
               

def make_field_mask(widths,labels,extent):
    '''
    self.points = np.array([position,
                         position + np.array([0,self.h]), 
                         position + np.array([self.w,self.h]),
                         position + np.array([self.w,0])])
    ''' 
        
def camera_to_world_transform(robot_pose,camera_pose,point):
    #transform xyz point from camera coordinate system to world coordinate system
    pass
    


'''
def make_line_samples(start,stop,num_samples):
    samples = []
    for coord in zip(start,stop):
        samples.append(np.linspace(coord[0],coord[1],num_samples))
    samples_per_point = map(list, zip(*samples)) #"transpose the coordinate list"
    return samples_per_point

def lines_to_camera_pixels(cam_model,xyz_start, xyz_stop,num_samples):
    line_points = make_line_samples(xyz_start,xyz_stop,num_samples)
    #debug
    plt.figure(6)
    plt.plot(np.array(line_points)[:,0],np.array(line_points)[:,2],'*')
    plt.axis('scaled')
    
    pixels = []
    for p in line_points:
        new_pixel = np.array(cam_model.vector_to_pixel(p)[0:2])
        
        if(new_pixel[0] < cam_model.height and new_pixel[1] < cam_model.width): #if within image boundaries
            pixels.append(new_pixel)
    return np.array(pixels)
'''


    

if __name__ == "__main__":
    #Simple test
    points = np.array([[0,0],[0,1],[1,1],[1,0]])
    box = Polygon(points)
    plt.figure(1)
    box.plot()
    
    #make a "crop row"
    row_width = 0.5
    row_extent = 5
    row_offset = 0
    row_label = 1
    row = Rectangle(width = row_width, height = row_extent, position = [0,0])#position = [row_offset-row_width/2,0])

    box2 = Polygon(row.points)
    plt.figure(2)
    box2.plot()
    
    '''
    #Transform box
    calib_file = os.path.join('/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/auto_nav/scripts/input_cam_model_campus_2018-08-31.xml')
    cam_model = OcamCalibCameraModel(calib_file)

    x = 0 # horizontal plane
    box_line_pixels = []
    for pp in box2.point_pairs:
        xyz_start = [pp[0,0],0,pp[0,1]] #fixme change back to x,y,z order when camera transform is implemented
        xyz_stop = [pp[1,0],0,pp[1,1]] #fixme ...
        box_line_pixels.append(lines_to_camera_pixels(cam_model,xyz_start, xyz_stop,num_samples = 1000))
    
    for line_pixels in box_line_pixels:
        plt.figure(3)
        plt.plot(line_pixels[:,0],line_pixels[:,1])
        plt.show()
    '''