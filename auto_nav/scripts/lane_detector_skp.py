#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import numpy as np
import pandas as pd
import cv2
import os
import sys
import roslib
import matplotlib.pyplot as plt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from moviepy.editor import VideoFileClip
from os.path import expanduser
import pickle
import math
import tf
from numpy import linalg as LA
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseArray,Point
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf2_ros
import quaternion
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
from itertools import imap


class lane_finder():
    '''
    A class to find lane points given an image that has been inverse perspective mapped and scrubbed of most features
    other than the lanes.
    '''

    def __init__(self, image, base_size=.2):
    #### Hyperparameters ####
        self.image = image
        self.vis = image # used for visualization
        self.lanes = []
        self.base_size = base_size
        self.roi = [1000, 1200]  #roi_x, roi_y

        max_value_H = 360/2
        max_value = 255
        self.HSV_low = [0, 0, 0] #low_H, low_S, low_V
        self.HSV_high = [max_value_H, max_value, 145] #high_H, high_S, high_V

    def pipeline(self):

        # Convert BGR to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]

        # Getting ROI
        Roi = hsv[self.roi[1]:height-self.roi[1],self.roi[0]:width-self.roi[0]] #-(roi_y+1),-(roi_x+1)

        # define range of blue color in HSV
        lower_t = np.array([self.HSV_low[0],self.HSV_low[1],self.HSV_low[2]])
        upper_t = np.array([self.HSV_high[0],self.HSV_high[1],self.HSV_high[2]])

        # Detect the object based on HSV Range Values
        mask = cv2.inRange(Roi, lower_t, upper_t)
        output = cv2.bitwise_and(Roi, Roi, mask=mask)

        # Opening the image
        kernel = np.ones((3,3),np.uint8)
        eroded = cv2.erode(mask, kernel, iterations = 1) # eroding + dilating = opening
        wscale = cv2.dilate(eroded, kernel, iterations = 1)
        ret, thresh = cv2.threshold(wscale, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # thresholding the image //THRESH_BINARY_INV
        median = cv2.medianBlur(thresh,35)

        # Finding contours for the thresholded image
        im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # # Find the index of the largest contour
        # areas = [cv2.contourArea(c) for c in contours]
        # max_index = np.argmax(areas)
        # cnt=contours[max_index]
        #
        # #cv2.drawContours(thresh, cnt, 0, (255, 0, 0), 2, 8, hierarchy)
        # #find the biggest area
        # c = max(contours, key = cv2.contourArea)

        #x,y,w,h = cv2.boundingRect(c)
        # draw the book contour (in green)
        #cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2)

        # create hull array for convex hull points
        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        # create an empty black image
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        threshold_area = 10000
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull

        # draw contours and hull points
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > threshold_area:
              # draw ith contour
              cv2.drawContours(output, contours, i, color_contours, 1, 8, hierarchy)
              # draw ith convex hull object
              cv2.drawContours(output, hull, i, color, 3, 8)

        return output

if __name__ == '__main__':

  try:
   rospy.init_node('lane_detector_skp', anonymous=True)

   # Load an color image in grayscale
   home = expanduser("~/ICRA_2020/lane_skp_1.jpg") #very_curved.png"
   rgb_img = cv2.imread(home) #'/home/saga/ICRA_2020/curved_lane.jpg')

   while not rospy.is_shutdown():
     lf = lane_finder(rgb_img, base_size=.2)

     output = lf.pipeline()
     #warped_img, centerLine, curve_fit_img, output = lf.pipeline()

     img_dir = expanduser("~/warped_img.jpg")
     cv2.imwrite(img_dir, output)

  except rospy.ROSInterruptException:
   pass
