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

    


