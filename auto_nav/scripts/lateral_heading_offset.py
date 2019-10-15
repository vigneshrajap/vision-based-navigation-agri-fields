#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import glob
import pyexcel as pe
import tf
import sys

book = pe.get_book(file_name="../config/coordinates_utm.ods")
gt_lat_utm = []
gt_long_utm = []
lane_number = str(4)

#lane_number = rospy.set_param('lane_number', 1)

for row in book["Sheet"+lane_number]:
        gt_lat_utm.append(row[1]) # Latitude
        gt_long_utm.append(row[2]) # Longitude

# print lat_utm[1:], long_utm[1:]

if __name__ == '__main__':
    try:
        #Initialize node
        rospy.init_node('lateral_heading_offset')
        listener = tf.TransformListener()

        while not rospy.is_shutdown():
            try:
               (trans,rot) = listener.lookupTransform('utm', 'base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
               continue

            robot_lat_utm = trans[1]
            robot_long_utm = trans[0]

            a = np.array((robot_lat_utm, robot_long_utm))
            b = np.array((gt_lat_utm[1], gt_long_utm[1]))
            dist_0 = np.linalg.norm(a-b)

            for i in range(2, len(gt_lat_utm)):
                b = np.array((gt_lat_utm[i], gt_long_utm[i]))
                dist = np.linalg.norm(a-b)
                if dist<dist_0:
                    dist_0 = dist
                    #print robot_lat_utm, gt_lat_utm[i], robot_long_utm, gt_long_utm[i]

            print dist_0

    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
