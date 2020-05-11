import pytest
from pytest import approx
import numpy as np
from homogeneous_transformation import *

def test_stackexchange_example():
    #https://robotics.stackexchange.com/questions/8053/how-to-use-the-homogeneous-transformation-matrix
    #Express point p_b in coordinate system phi_a
    #Coordinate system phi_b is translated with 3,2 and rotated with 90 degrees in phi_a
    translation = [3,2]
    p_b = [3,4]
    theta = np.pi/2
    H_b_to_a = create_transformation_matrix(r=[0,0,theta],t=[translation[0],translation[1],0])
    p_a = transform_xyz_point(H_b_to_a,[p_b[0],p_b[1],0])
    assert p_a == approx([-1, 5, 0])

def test_rotation_only():
    #The robot coordinate system is rotated with a negative angle compared to the map axes.
    #The transformation goes from robot coordinates  to map coordinates
    theta = -np.pi/6
    p_robot = [1,0,0]
    T_robot_to_map = create_transformation_matrix(r=[0,0,theta], t = [0,0,0])
    p_map = transform_xyz_point(T_robot_to_map,p_robot)
    print(p_map)
    assert p_map == approx([0.8660254,-0.5, 0])

def test_rotation_and_translation():
    #The robot coordinate system is translated with positive values from origin in the map and rotated with a negative angle compared to the map axes. 
    #The transformation goes from robot coordinates to map coordinates
    theta = -np.pi/6
    translation = [1,1,0]
    p_robot = [1,0,0]
    T_robot_to_map = create_transformation_matrix(r=[0,0,theta], t = translation)
    p_map = transform_xyz_point(T_robot_to_map,p_robot)
    print(p_map)
    assert p_map == approx([1.8660254,0.5, 0])

def test_rotation_and_translation2():
    #The robot coordinate system is translated with negative x and positive y from origin in the map and rotated with a negative angle compared to the map axes. 
    #The transformation goes from robot coordinates to map coordinates
    theta = -np.pi/6
    translation = [-1,1,0]
    p_robot = [1,0,0]
    T_robot_to_map = create_transformation_matrix(r=[0,0,theta], t = translation)
    p_map = transform_xyz_point(T_robot_to_map,p_robot)
    print(p_map)
    assert p_map == approx([1.8660254-2,0.5, 0])