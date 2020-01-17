#!/usr/bin/env python

def pixel_to_vector(self,x,y):
    #NOTE: model (x,y) is (height,width) so we swap
    dx = y - self.xc
    dy = x - self.yc;
    px = self.inv_M[0,0]*dx +self.inv_M[0,1]*dy;
    py = self.inv_M[1,0]*dx + self.inv_M[1,1]*dy;
    R2 = px*px + py*py;
