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
    #Origo is at h = 0 and w/2 (in the middle of the center row)
    widths = np.array(widths)
    
    position = [0,0]
    list_of_polygons = []
    h = extent
    #shift to get the desired origo
    shift = -np.array([np.sum(widths)/2, 0])
    for w,label in zip(widths,labels):
        points = np.array([position + shift,
                           position + shift + np.array([w,0]),
                           position + shift + np.array([w,h]),
                           position + shift + np.array([0,h])])
        list_of_polygons.append(Polygon(points,label))
        position = points[0] #position of next rectangle
    
    return list_of_polygons
        

#%% Transformation related stuff (move later?)
'''
Roll, pitch, yaw is rotation around x,y,z axis. 
To combine rotation matrices, use Euler convension and rotate in x,y,z order around x axis first: R=RzRyRx

'''
#camera-robot specific
'''
def transform_camera_to_world_transform(T_wr = eye(4),T_rc,point = [0,0,0]):
    #transform xyz point from camera coordinate system to world coordinate system
    #return transformed_point
    
    pass
'''
def transform_point_camera_to_world(point = [0,0,0], T_camera_to_robot = np.eye(4), T_robot_to_world = np.eye(4)):
    '''
    Transform point from camera coordinates to world coordinates. 
    '''
    T_cam_to_world = T_robot_to_world.dot(T_camera_to_robot)
    return transform_xyz_point(T_cam_to_world,point)

def set_up_robot_to_world_transform(robot_rpy, robot_xyz):
    return create_transformation_matrix(r=robot_rpy, t=robot_xyz)
        
def set_up_camera_to_robot_transform(camera_rpy = [0,0,0],camera_xyz = [0,0,0]):
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
    
    #camera tilt and position compared to robot coordinate system
    T_camaligned_to_rob = create_transformation_matrix(r = camera_rpy,t = camera_xyz)
    
    T_cam_to_rob = T_camaligned_to_rob.dot(T_cam_to_camaligned)
    return T_cam_to_rob


#General
#Rotation matrices independent of coordinate system definition: 
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
    

if __name__ == "__main__":
    #make a "crop row"
    list_of_polygons = make_field_mask(widths = [0.5], labels = [1], extent = 5)#position = [row_offset-row_width/2,0])
    box1 = list_of_polygons[0]
    plt.figure(2)
    box1.plot()
    
    is_inside = box1.check_if_inside([0.45,1])
    print(is_inside)
    
    #Transformations 
    
    #Transform pixel
    calib_file = os.path.join('../auto_nav/scripts/input_cam_model_campus_2018-08-31.xml')
    cam_model = OcamCalibCameraModel(calib_file)
    vector = cam_model.pixel_to_vector(cam_model.width*1/2, cam_model.height*1/2)    
    normalized_vector = vec3_normalise(vector)
    print(vector,normalized_vector)
    
    #Camera robot transformations
    T_camera_to_robot = set_up_camera_to_robot_transform(camera_rpy = [0,-np.pi/8,0],camera_xyz =[0,0,1])
    vector_robot = transform_xyz_point(T_camera_to_robot,normalized_vector)
    print('vector_robot', vector_robot)
    
    T_robot_to_world = set_up_robot_to_world_transform(robot_rpy = [0,0,np.pi/16],robot_xyz = [0,0,0])
    vector_world = transform_point_camera_to_world(point = normalized_vector,T_camera_to_robot = T_camera_to_robot, T_robot_to_world = T_robot_to_world)
    print('point_world',vector_world)
