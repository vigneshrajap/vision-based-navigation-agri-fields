#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import glob
import pyexcel as pe
import tf
import sys
import math
from tf.transformations import euler_from_quaternion

book = pe.get_book(file_name="../config/ground_truth_coordinates_utm.xls")
gt_lat_utm = []
gt_long_utm = []
lane_number = str(2)

#lane_number = rospy.set_param('lane_number', 1)

for row in book["Sheet"+lane_number]:
        gt_lat_utm.append(row[1]) # Latitude
        gt_long_utm.append(row[2]) # Longitude

gt_lat_map = []
gt_long_map = []
origin_map_x = 6614855.745
origin_map_y = 594362.895
for i in range(2, len(gt_lat_utm)):
        gt_lat_map.append(gt_lat_utm[i]- origin_map_x) # Latitude
        gt_long_map.append(gt_long_utm[i]- origin_map_y) # Longitude

if __name__ == '__main__':
    try:
        #Initialize node
        rospy.init_node('lateral_heading_offset')
        listener = tf.TransformListener()
        listener1 = tf.TransformListener()

        while not rospy.is_shutdown():
            try:
               (trans,rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
               continue

            # try:
            #    (trans1,rot1) = listener1.lookupTransform('utm', 'map', rospy.Time(0))
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #    continue

            robot_lat_utm = trans[1] # Latitude
            robot_long_utm = trans[0] # Longitude
            (roll_r,pitch_r,yaw_r) = euler_from_quaternion(rot)

            # (roll_r1,pitch_r1,yaw_r1) = euler_from_quaternion(rot1)
            #
            # yaw_r = yaw_r - yaw_r1
            # #print yaw_r

            a = np.array((robot_lat_utm, robot_long_utm))
            #b = np.array((gt_lat_utm[1], gt_long_utm[1]))
            b = np.array((gt_lat_map[0], gt_long_map[0]))
            dist_0 = np.linalg.norm(a-b)
            #yaw_gt = math.atan2(gt_long_utm[1],gt_lat_utm[1])

            for i in range(1, len(gt_lat_map)):
                b = np.array((gt_lat_map[i], gt_long_map[i]))
                dist = np.linalg.norm(a-b)
                if dist<dist_0:
                    dist_0 = dist
                    #yaw_gt = math.atan2(gt_long_map[i],gt_lat_map[i])
                    #print robot_lat_utm, gt_lat_utm[i], robot_long_utm, gt_long_utm[i]

            #print yaw_r1, yaw_gt
            print dist_0
    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
