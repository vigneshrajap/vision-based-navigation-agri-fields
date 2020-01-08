#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt

'''
class Rectangle():
    def __init__(self,width,height,position):
        #position: x,y coordinate of lower left corner
        self.w = width
        self.h = height
        self.corners = np.array([position,
                                 position + [0,self.h], 
                                 position + [self.w,self.h],
                                 position + [self.w,0]])
'''

class Shape():
    def __init__(self,points,label=1):
        self.label = label
        self.points = points
        #Make point pairs
        shifted_points = np.roll(points,-1,axis = 0)
        self.point_pairs = np.array([points.T, shifted_points.T]).T
        
    def plot_shape(self):
        plt.figure()
        for pp in self.point_pairs:
            plt.plot(pp[0,:],pp[1,:])

'''
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
'''

if __name__ == "__main__":
    points = np.array([[0,0],[0,1],[1,1],[1,0]])
    box = Shape(points)
    print(box.point_pairs)
    box.plot_shape()
    
    