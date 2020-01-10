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

def line_XY_intersection(point, direction):
    """
    Finds intersection (x,y) between XY plane and a line.

    point: some point on the line (e.g. camera position)
    direction: some vector pointing along the line

    Assumes numpy arrays.
    """
    r = point[2]/direction[2]
    xy = point[0:2] - r*direction[0:2]
    return xy

def orient2d(a, b, c):
    """
    The Orient2D geometric predicate.

    The output is a scalar number which is:
        > 0 if abc forms an angle in (0, pi), turning left,
        < 0 if abc forms an angle in (0, -pi), turning right,
        = 0 if abc forms an angle equal to 0 or pi, or is straight.

    Alternatively, it can be interpreted as:
        > 0 if c is to the left of the line ab,
        < 0 if c is to the right of the line ab,
        = 0 if c is on the line ab,
    in all cases seen from above, and along ab.

    The algorithm do not use exact arithmetics, and may fail within
    machine tolerances.
    """
    return (a[0]-c[0])*(b[1]-c[1]) - (a[1]-c[1])*(b[0]-c[0])

class Polygon():
    '''
    Polygon represented by list of points (corners)
    Points must be in counter clockwise order
    '''
    
    def __init__(self,points,label=1):
        self.label = label
        self.points = points
        
    def plot(self):
        #plt.figure()
        plt_indeces = np.append(range(self.points.shape[0]),0)
        plt.plot(self.points[plt_indeces,0],self.points[plt_indeces,1])
        plt.axis('scaled')
        
    def make_pointpairs(self):
        return zip(self.points,np.roll(self.points,-1,axis=0))
        
    def check_if_inside(self,q):
        '''
        Check if a point q (x,y) is inside the polygon with geometric predicate for each side
        '''
        is_inside = True
        for p, p_next in self.make_pointpairs():
            if orient2d(p,p_next,q) < 0: 
                is_inside = False
                break
        return is_inside
               

def make_field_mask(widths,labels,extent):
    #Create adjacent rectangles with row/crop labels and save as polygons
    position = [0,0]
    list_of_polygons = []
    h = extent
    for w,label in zip(widths,labels):
        points = np.array([position,
                           position + np.array([w,0]),
                           position + np.array([w,h]),
                           position + np.array([0,h])])
        list_of_polygons.append(Polygon(points,label))
        position = points[0] #position of next rectangle
    return list_of_polygons
        
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
    #make a "crop row"
    list_of_polygons = make_field_mask(widths = [0.5], labels = [1], extent = 5)#position = [row_offset-row_width/2,0])
    box1 = list_of_polygons[0]
    plt.figure(2)
    box1.plot()
    
    is_inside = box1.check_if_inside([0.45,1])
    print(is_inside)
    
    #Transform box
    calib_file = os.path.join('../auto_nav/scripts/input_cam_model_campus_2018-08-31.xml')
    cam_model = OcamCalibCameraModel(calib_file)
    