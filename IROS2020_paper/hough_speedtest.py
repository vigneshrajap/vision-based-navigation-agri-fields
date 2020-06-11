#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:12:26 2020

@author: marianne
"""

import cv2
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import time
# Read Images 
img = mpimg.imread("/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/predicted_outputs/predicted_outputs_resnet50_segnet/frogn_10000.png", cv2.CV_8UC1)
img = cv2.resize(img,None,fx=0.25,fy=0.25)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
minLineLength = 100
maxLineGap = 10
runtime = []
for r in range(100):
    t_start = time.time()
    lines = cv2.HoughLinesP(gray,1,np.pi/180, 100, minLineLength, maxLineGap)
    t_end = time.time()
    for i in range(1,len(lines)):
      for x1,y1,x2,y2 in lines[i]:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    runtime.append(t_end-t_start)
    print('Prediction time: ', t_end-t_start)
# Output Images 
plt.imshow(img)
print('Mean runtime:', np.mean(runtime))
print('Median runtime: ', np.median(runtime))
print('Std runtime: ',np.std(runtime))
    
