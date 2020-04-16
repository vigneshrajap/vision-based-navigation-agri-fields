import numpy as np 

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

def line_to_next_point(point_ind, xs, ys):
    #Line segment directly from current to next point
    #Returns line on point, unit vector form
    point = np.array([xs[point_ind],ys[point_ind]])
    next_point = np.array([xs[point_ind+1],ys[point_ind+1]])
    vector = next_point-point
    vector = vector/np.linalg.norm(vector)

    print('pints', point, next_point)

    return point,vector

    


