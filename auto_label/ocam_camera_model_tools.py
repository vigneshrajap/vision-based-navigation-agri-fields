#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:50:15 2019

@author: marianne

Translated from C++ code by Richard Moore

"""

import numpy as np

# Utility functions
def vec3_normalise(point):
   return vec    

def invert_matrix_2d(matrix):
   return inv_matrix
    
class OcamCalibCameraModel:
    '''
    '''
    def __init__(self,calib_file):
        self.set_params()
        return None
    
    def set_params(ss0,ss1,ss2,ss3,ss4,c,d,e,xc,yc):
        self.fx[0] = ss4
        self.fx[1] = ss3
        self.fx[2] = ss2
        self.fx[3] = ss1
        self.fx[4] = ss0
        self.dfdx[0] = 4 * self.fx[0]
        self.dfdx[1] = 3 * self.fx[1]
        self.dfdx[2] = 2 * self.fx[2]
        self.dfdx[3] = self.fx[3];
        self.M[0] = c
        self.M[1] = d
        self.M[2] = e
        self.M[3] = 1.0
        self.invM = invertMatrix2d(self.M)
        self.xc = xc
        self.yc = yc
        return None
    
    def read_ocam_calib_from_file(calib_file):
        #some kind of xml parsing
        return None
            
    def vector_to_pixel(point):
        '''
        Go from vector (in camera coordinates) to pixel (image coordinates)
        '''
        forward = np.array([0,0,1])
        r = vec3normalise(point)
        alpha = np.arccos(np.dot(r,forward))
        R = alphaToR(alpha)
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
        y = self.M[0]*px + self.M[1]*py + self.xc
        x = self.M[2]*px + self.M[3]*py + self.yc
    
        return x, y, R**2
    
     def alpha_to_R(alpha):
        '''
        Solves polynomial to go from alpha (angle between rays) to R (distance from center point on sensor)
        '''
        #some kind of polynomial solver in python
        return R


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