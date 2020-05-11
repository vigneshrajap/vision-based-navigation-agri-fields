import numpy as np

def x_rotation_matrix(theta):
    Rx = np.array([
            np.array([1, 0, 0]),
            np.array([0, np.cos(theta), -np.sin(theta)]),
            np.array([0, np.sin(theta), np.cos(theta)])
            ])
    return Rx

def y_rotation_matrix(theta):
    Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
            ])
    return Ry

def z_rotation_matrix(theta):
    Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
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