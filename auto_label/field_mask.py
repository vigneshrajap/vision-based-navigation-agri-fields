#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from ocam_camera_model_tools import OcamCalibCameraModel,vec3_normalise
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate
import cv2

#%% General transformation stuff
'''
Roll, pitch, yaw is rotation around x,y,z axis. 
To combine rotation matrices, rotate in x,y,z order around x axis first: R=RzRyRx

'''

def x_rotation_matrix(theta):
    Rx = np.array([
            np.array([1, 0, 0]),
            np.array([0, np.cos(theta), np.sin(theta)]),
            np.array([0, -np.sin(theta), np.cos(theta)])
            ])
    return Rx

def y_rotation_matrix(theta):
    Ry = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
            ])
    return Ry
    
def z_rotation_matrix(theta):
    Rz = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
            ])
    return Rz

def create_transformation_matrix(r,t):
    '''
    Make a homogeneous 4x4 translation matrix from rotation angles (in radians) rx,ry,rz and translations (in meteres) tx,ty,tz
    Transformation order:
        1. x axis rotation
        2. y axis rotation
        3. z axis rotation
        4. translation
    '''
    rx,ry,rz = r 
    tx,ty,tz = t
    Rx = x_rotation_matrix(rx)
    Ry = y_rotation_matrix(ry)
    Rz = z_rotation_matrix(rz)
    R = Rz.dot(Ry).dot(Rx) #combine rotation matrices: x rotation first
    t = np.array([tx,ty,tz])
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def transform_xyz_point(T,point):
    P = np.eye(4)
    P[0:3,3] = point
    P_transformed = T.dot(P)
    return P_transformed[0:3,3]
    
#%% Utilities
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
    
#%% Core functionality
'''
Assumptions:
    Robot coordinate system: Aligned with robot base, moving with the robot. X ahead, Y to the left, Z up
    World coordinate system: The local ground truth coordinate system. 
    Camera coordinate system: X right, Y down, Z into the image
'''

def make_field_mask(labels = [1,0,1], lane_spacing = None, crop_duty_cycle = None, widths = None, extent=5):
    #Create adjacent rectangles with row/crop labels and save as polygons
    #Origo is at x = 0 and y = w/2 (in the middle of the center row)
    # label = 1 is lane = 0 is crop
    #x ahead, y to the left

    position = np.array([0,0])
    list_of_polygons = []
    h = extent
    #shift to get the desired origo
    if widths is None:
        total_width = lane_spacing*(np.sum(labels) * (1-crop_duty_cycle) + np.sum(np.logical_not(labels)) * crop_duty_cycle)
    else:
        total_width = np.sum(widths)
    shift = -np.array([0,total_width/2])

    for ind,label in enumerate(labels):
        if widths is None:
            w = lane_spacing*((1-crop_duty_cycle)*label + crop_duty_cycle*int(not(label)))
        else:
            w = widths[ind]
        
        points = np.array([position + shift,
                           position + shift + np.array([h,0]),
                           position + shift + np.array([h,w]),
                           position + shift + np.array([0,w])
                           ])
        list_of_polygons.append(Polygon(points,label))
        position = position + np.array([0,w]) #position of next rectangle
    
    return list_of_polygons

'''
def make_field_mask(widths,labels,extent):
    #Create adjacent rectangles with row/crop labels and save as polygons
    #Origo is at x = 0 and y = w/2 (in the middle of the center row)
    #x ahead, y to the left
    widths = np.array(widths)
    
    position = np.array([0,0])
    list_of_polygons = []
    h = extent
    #shift to get the desired origo
    shift = -np.array([0,np.sum(widths)/2]) 
    for w,label in zip(widths,labels):
        points = np.array([position + shift,
                           position + shift + np.array([h,0]),
                           position + shift + np.array([h,w]),
                           position + shift + np.array([0,w])
                           ])
        list_of_polygons.append(Polygon(points,label))
        position = position + np.array([0,w]) #position of next rectangle
    
    return list_of_polygons
'''
def camera_to_world_transform(T_camera_to_robot = np.eye(4), T_robot_to_world = np.eye(4)):
    return T_robot_to_world.dot(T_camera_to_robot)

def set_up_robot_to_world_transform(rpy, xyz):
    return create_transformation_matrix(r=rpy, t=xyz)
        
def set_up_camera_to_robot_transform(rpy = [0,0,0], xyz = [0,0,0]):
    ''' 
    Robot coordinates: x ahead, y left, z up
    Camera coordinaes: x right, y down, z ahead
    
    inputs: camera pose in robot (base) coordinates
    '''
    # Rotation between coordinate systems. Creating a camera coordinate system 
    # aligned with the robot coordinate system (x_robot = z_cam, y_robot = -x_cam, z_robot = -y_cam)
    rx = np.pi/2
    ry = 0
    rz = np.pi/2
    T_cam_to_camaligned = create_transformation_matrix(r = [rx,ry,rz],t = [0,0,0])
    #Camera tilt and position compared to robot coordinate system
    T_camaligned_to_rob = create_transformation_matrix(r = rpy,t = xyz)
    #Combine 
    T_cam_to_rob = T_camaligned_to_rob.dot(T_cam_to_camaligned)
    return T_cam_to_rob

def make_image_mask_from_polygons(cam_model,polygon_mask,T_cam_to_world,sampling_step = None):
    ''' 
    Make an image mask from a polygon_mask in world coordinates, based on camera model and camera to world transform
    Cropped dims is height, width
    Output: 
        channel 0: polygon index
        channel 1: label
    '''
    if sampling_step is None:
        sampling_step = 1

    camera_origo = transform_xyz_point(T_cam_to_world,[0,0,0]) #camera origo in world coordinates
    
    #Generate mask image
    image_dims = np.array([cam_model.height,cam_model.width])
    subsampled_dims = image_dims/sampling_step
    mask_image = np.zeros((subsampled_dims[0],subsampled_dims[1],2))

    for j in tqdm(np.arange(subsampled_dims[1])):
        for i in np.arange(subsampled_dims[0]):
            pixel = np.array([i,j])*sampling_step
            mask_image[i,j,0],mask_image[i,j,1] = check_if_pixel_inside_mask(polygon_mask,cam_model,pixel,T_cam_to_world,camera_origo)
    return mask_image
    
def check_if_pixel_inside_mask(polygon_mask,cam_model,pixel_yx, T_cam_to_world,camera_origo):
    '''
    For a given camera model, check if pixel y,x is inside a set of polygon shapes
    Return polygon index and polygon label
    If not, return NaN
    '''
    v = cam_model.pixel_to_vector(pixel_yx[1],pixel_yx[0])
    v_world = transform_xyz_point(T_cam_to_world,v)
    gp = line_XY_intersection(point=camera_origo,direction = np.array(v_world)-np.array(camera_origo))
    for poly_index,poly in enumerate(polygon_mask):
        if poly.check_if_inside(gp):
            #mask_image[i,j,0] = poly_index
            #mask_image[i,j,1] = poly.label
            return poly_index, poly.label
            break
    return np.NaN, np.NaN

if __name__ == "__main__":
    #%% Demo code: Hot to make an image field mask
    
    #Dummy field mask
    polygon_field_mask = make_field_mask(widths = [0.5,0.3,0.5], labels = [1,0,1], extent = 5)
    
    #%%Transformations 
    #Old camera model with adjustments for testing
    calib_file = os.path.join('../auto_nav/scripts/input_cam_model_campus_2018-08-31.xml')
    cam_model = OcamCalibCameraModel(calib_file)

    #Camera setup    
    #dummy values, should get values from outside
    camera_tilt = np.pi/8 
    camera_height = 1
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = [0,-camera_tilt,0], xyz =[0,0,camera_height])    
    
    
    #Robot position 
    #dummy values, should get values from outside
    robot_rpy = [0,0,0]
    robot_xyz = [0,0,0]
    T_robot_to_world = set_up_robot_to_world_transform(rpy = robot_rpy, xyz = robot_xyz)
    
    T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
    image_mask = make_image_mask_from_polygons(cam_model, polygon_field_mask, T_cam_to_world)
        
    plt.figure(10)
    plt.imshow(image_mask[:,:,0])  
    plt.figure(11)          
    plt.imshow(image_mask[:,:,1])

        
            
