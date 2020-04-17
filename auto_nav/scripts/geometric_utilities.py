import numpy as np 
import matplotlib.pyplot as plt

#Utility functions for two-dimensional vectors

def direction_sign(v0, v1):
    #Input: unit vectors in 2D
    #Check whether vector 1 has the same direction as vector 2 (the angle between them is smaller than 90 degrees)
    return np.sign(np.dot(v0,v1))

def angle_between_vectors(v0,v1):
    # Input: unit vectors in 2D
    # Compute angle between two vectors. Output range is from -pi/2 to pi/2
    sin_theta = np.cross(v0,v1)
    return np.arcsin(sin_theta)

def signed_distance_point_to_line(point, line_point, line_vector):
    #Input: 2D points, 2D unit vector
    point_vector = point-line_point
    d = np.cross(line_vector,point_vector)
    return d

def closest_point(x0,y0,xs,ys):
    #Compute distance between points, given as lists or np arrays of x and y coordinates
    x0 = np.repeat(x0,len(xs))
    y0 = np.repeat(y0,len(ys))
    sum_squared_error = 0.5*np.sqrt((x0-xs)**2 +(y0-ys)**2)
    ind = np.argmin(sum_squared_error)
    
    return ind

def line_to_next_point(point_ind, xs, ys, step = 1):
    #Line segment directly from current to next point
    #Returns line on point, unit vector form
    point = np.array([xs[point_ind],ys[point_ind]])
    next_point = np.array([xs[point_ind+step],ys[point_ind+step]])
    vector = next_point-point
    vector = vector/np.linalg.norm(vector)

    return point,vector

def line_fit_from_points(point_ind,xs,ys,forward_window=20, backward_window=0):
    xs = np.array(xs)
    ys = np.array(ys)
    
    if backward_window < point_ind:
        xpoints = xs[point_ind - backward_window : point_ind + forward_window+1]
        ypoints = ys[point_ind - backward_window : point_ind + forward_window+1]
    else:
        xpoints = xs[point_ind : point_ind + forward_window+1]
        ypoints = ys[point_ind : point_ind + forward_window+1]
    #Compute linear fit
    coef = np.polyfit(xpoints,ypoints,deg=1)
    fit_fn = np.poly1d(coef) 
    
    # Get fitted line on point,vector form
    line_point = np.array([xs[point_ind],fit_fn(xs[point_ind])])
    next_line_point = np.array([xs[point_ind+1],fit_fn(xs[point_ind+1])])
    line_vector = next_line_point - line_point
    line_vector = line_vector/np.linalg.norm(line_vector)

    return line_point,line_vector