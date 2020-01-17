#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:50:15 2019

@author: marianne

Translated from C++ code by Richard Moore

"""
import os
import numpy as np
import xmltodict
        

# Utility functions
def vec3_normalise(point):
    norm = np.linalg.norm(point)
    if norm == 0: 
       return point
    return point / norm

#Evaluation of polynomials
#cubic
def eval_poly3(poly,x):
    return ((poly[0]*x + poly[1])*x + poly[2])*x + poly[3];

#quartic
def eval_poly4(poly, x):
    return (((poly[0]*x + poly[1])*x + poly[2])*x + poly[3])*x + poly[4];

class OcamCalibCameraModel:
    '''
    '''
    def __init__(self,calib_file):
        param_dict = self.read_opencv_storage_from_file(calib_file)
        self.set_params(param_dict)
        return None
    
    def set_params(self,opencv_storage_dict):
        #Etract parameters from opencv storage dictionary object
        d = opencv_storage_dict
        self.fx = np.zeros(5)
        self.fx[0] = float(d['ss4'])
        self.fx[1] = float(d['ss3'])
        self.fx[2] = float(d['ss2'])
        self.fx[3] = float(d['ss1'])
        self.fx[4] = float(d['ss0'])
        self.M = np.zeros((2,2))
        self.M[0,0] = float(d['c'])
        self.M[0,1] = float(d['d'])
        self.M[1,0] = float(d['e'])
        self.M[1,1] = 1.0
        self.xc = float(d['xc'])
        self.yc = float(d['yc'])
        self.width = int(d['width'])
        self.height = int(d['height'])
        self.image_circle_FOV = float(d['imageCircleFOV'])
        
        #Derived parameters
        self.dfdx = np.zeros(4)
        self.dfdx[0] = 4 * self.fx[0]
        self.dfdx[1] = 3 * self.fx[1]
        self.dfdx[2] = 2 * self.fx[2]
        self.dfdx[3] = self.fx[3]
        self.inv_M = np.linalg.inv(self.M)
        self.initial_x = self.width/4 #starting point for polynomial solver
        
    
    def read_opencv_storage_from_file(self,calib_file):
        print(calib_file)
        with open(calib_file) as fd:
            dict_ = xmltodict.parse(fd.read())
            model_dict = dict_['opencv_storage']['cam_model']
        return model_dict
            
    def vector_to_pixel(self,point):
        '''
        Go from vector (in camera coordinates) to pixel (image coordinates)
        input: point - x,y,z in camera frame
        '''
        forward = np.array([0,0,1])
        r = vec3_normalise(point)
        alpha = np.arccos(np.dot(r,forward))
        R = self.alpha_to_R(alpha)
        if R <0 :
            x = -1.0
            y = -1.0
            return false #??
        #Scale to get ideal fisheye pixel coordinates:
        mag = np.sqrt(r[0]**2 + r[1]**2)
        if (mag != 0):
            mag = R / mag
        # NOTE: model (x,y) is (height,width) so we swap
        px = r[1] * mag
        py = r[0] * mag
        # Account for non ideal fisheye effects (shear and translation):
        y = self.M[0,0]*px + self.M[0,1]*py + self.xc
        x = self.M[1,0]*px + self.M[1,1]*py + self.yc
        
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        return x, y, R**2
    
    def pixel_to_vector(self,x,y):
        #NOTE: model (x,y) is (height,width) so we swap
        dx = y - self.xc
        dy = x - self.yc;
        px = self.inv_M[0,0]*dx +self.inv_M[0,1]*dy;
        py = self.inv_M[1,0]*dx + self.inv_M[1,1]*dy;
        R2 = px*px + py*py;

        direction = np.array([0,0,0])
        direction[0] = py;
        direction[1] = px;
        direction[2] = -eval_poly4(self.fx, np.sqrt(R2));

        return direction
    
    def alpha_to_R(self,alpha):
        '''
        Solves polynomial to go from alpha (angle between rays) to R (distance from center point on sensor)
        '''
        #Newton-Raphson search for the solution

        newFx3 = self.fx[3] - np.tan(alpha - np.pi/2);
        fx = np.array([self.fx[0], self.fx[1], self.fx[2], newFx3, self.fx[4]])
        dfdx = np.array([self.dfdx[0], self.dfdx[1], self.dfdx[2], newFx3])

        x = self.initial_x;
        while True:
            px = x
            x -= eval_poly4(fx,x) / eval_poly3(dfdx,x)
            if (abs(x - px) > 1e-3):
                break
        R = x    
        return R


if __name__ == "__main__":
    calib_file = os.path.join('/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/auto_nav/scripts/input_cam_model_campus_2018-08-31.xml')
    ocam_obj = OcamCalibCameraModel(calib_file)
    #vector to point debug
    point = [0.1,0.1,1]
    pixel = ocam_obj.vector_to_pixel(point)
    print(pixel)
    
#Original c++ code:

'''
CmReal OcamCalibCameraModel::_alphaToR(CmReal alpha) const

{

                // Newton-Raphson search for the solution

                CmReal newFx3 = _fx[3] - tan(alpha - CM_PI_2);

                CmReal fx[5] = {_fx[0], _fx[1], _fx[2], newFx3, _fx[4]};

                CmReal dfdx[4] = {_dfdx[0], _dfdx[1], _dfdx[2], newFx3};

                CmReal px, x=_initial_x;

                do {

                               px = x;

                               x -= eval_poly4(fx,x) / eval_poly3(dfdx,x);

                } while (fabs(x - px) > 1e-3);

                return x;

}
''' 
'''
bool OcamCalibCameraModel::vectorToPixel(

                const CmPoint &point, CmReal& x, CmReal& y) const

{

                const CmReal forward[3] = {0,0,1};

                CmReal r[3] = {point[0], point[1], point[2]};

                vec3normalise(r);

                CmReal alpha = acos(vec3dot(r,forward));

                CmReal R = _alphaToR(alpha);

                if (R < 0) {

                               // Uh oh, undefined

                               x = -1.0;

                               y = -1.0;

                               return false;

                }

                CmReal mag = sqrt(r[0]*r[0] + r[1]*r[1]);

                if (mag != 0)

                               mag = R / mag;

                // NOTE: model (x,y) is (height,width) so we swap

                CmReal px = r[1] * mag;

                CmReal py = r[0] * mag;

                y = _M[0]*px + _M[1]*py + _xc;

                x = _M[2]*px + _M[3]*py + _yc;

                return _validPixel(x, y, R*R);

}
'''