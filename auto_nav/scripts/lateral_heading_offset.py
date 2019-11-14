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
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped, Point, PoseStamped
from sensor_msgs.msg import NavSatFix
import geo2UTM

class offset_estimation():
    '''
    A class to find fitted center of the lane points along the crop rows given an input RGB image.
    '''
    def __init__(self):
        self.datum = [-6614855.745, -594362.895, 0.0] # Manual Datum (NEGATE if UTM is the child frame)
        self.gt_utm = []
        self.gt_map = []
        self.book = pe.get_book(file_name="/home/vignesh/planner_ws/src/vision-based-navigation-agri-fields/auto_nav/config/ground_truth_coordinates.xls")
        self.lane_number = str(1) #rospy.set_param('lane_number', 1)

        self.map_frame = str('map')
        self.utm_frame = str('utm')
        self.robot_frame = str('base_link')
        self.gps_frame = str('gps')
        self.gps_topic_name = str('/gps/fix')
        self.receive_gps_fix = False
        self.gps_fix = NavSatFix()
        self.north = []
        self.east = []

        for row in self.book["Sheet"+self.lane_number]:
                self.gt_utm.append([row[1],row[2]]) # Latitude, Longitude

        self.listener = tf.TransformListener()
        self.listener1 = tf.TransformListener()
        self.map_transform = TransformStamped()
        self.map_transform.header.stamp = rospy.Time.now()
        self.map_transform.header.frame_id = self.map_frame
        self.map_transform.child_frame_id = self.utm_frame
        self.map_transform.transform.translation = Vector3(self.datum[0], self.datum[1], self.datum[2]);
        self.map_transform.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        #Listen to image messages and publish predictions with callback
        self.fix_sub = rospy.Subscriber(self.gps_topic_name, NavSatFix, self.gpsFixCallback)

    def gpsFixCallback(self, fix_msg):
        # Read fix messages
        self.gps_fix = fix_msg
        self.receive_gps_fix = True

    def ground_truth_utm2map(self):

        for i in range(1, len(self.gt_utm)): # Skip the first row (String)
          pose_stamped = PoseStamped()
          pose_stamped.pose.position = Point(self.gt_utm[i][0], self.gt_utm[i][1], 0)
          pose_stamped.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

          pose_transformed = PoseStamped()
          pose_transformed.header.stamp = rospy.Time.now()
          pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.map_transform) # Transform RTK values w.r.t to "Map" frame
          self.gt_map.append([pose_transformed.pose.position.x, pose_transformed.pose.position.y])

if __name__ == '__main__':
    try:
        #Initialize node
        rospy.init_node('lateral_heading_offset')
        r = rospy.Rate(1)
        oe = offset_estimation()

        # Function to obtain the ground truth values in Map frame
        oe.ground_truth_utm2map()

        while not rospy.is_shutdown():
           if oe.receive_gps_fix== True:
               gps_fix_utm = geo2UTM.geo2UTM(oe.gps_fix.latitude, oe.gps_fix.longitude)
               gps_pose_stamped = PoseStamped()
               gps_pose_stamped.pose.position = Point(gps_fix_utm[0], gps_fix_utm[1], 0)
               gps_pose_stamped.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

               gps_pose_transformed = PoseStamped()
               gps_pose_transformed.header.stamp = rospy.Time.now()
               gps_pose_transformed = tf2_geometry_msgs.do_transform_pose(gps_pose_stamped, oe.map_transform) # Transform RTK values w.r.t to "Map" frame

               print gps_pose_stamped.pose.position.y+oe.datum[1] #gps_pose_transformed

                # try:
                #    (trans,rot) = listener.lookupTransform(map_frame, utm_frame, rospy.Time(0))
                # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                #    continue
                #
                # # try:
                # #    (trans1,rot1) = listener1.lookupTransform('utm', 'map', rospy.Time(0))
                # # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # #    continue
                #
                # robot_lat_utm = trans[1] # Latitude
                # robot_long_utm = trans[0] # Longitude
                # (roll_r,pitch_r,yaw_r) = euler_from_quaternion(rot)
                #
                # # (roll_r1,pitch_r1,yaw_r1) = euler_from_quaternion(rot1)
                # #
                # # yaw_r = yaw_r - yaw_r1
                # # #print yaw_r
                #
                # a = np.array((robot_lat_utm, robot_long_utm))
                # #b = np.array((gt_lat_utm[1], gt_long_utm[1]))
                # b = np.array((gt_lat_map[0], gt_long_map[0]))
                # dist_0 = np.linalg.norm(a-b)
                # #yaw_gt = math.atan2(gt_long_utm[1],gt_lat_utm[1])
                #
                # for i in range(1, len(gt_lat_map)):
                #     b = np.array((gt_lat_map[i], gt_long_map[i]))
                #     dist = np.linalg.norm(a-b)
                #     if dist<dist_0:
                #         dist_0 = dist
                        #yaw_gt = math.atan2(gt_long_map[i],gt_lat_map[i])
                        #print robot_lat_utm, gt_lat_utm[i], robot_long_utm, gt_long_utm[i]

                # print yaw_r1, yaw_gt
                # print dist_0
           r.sleep()
    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
