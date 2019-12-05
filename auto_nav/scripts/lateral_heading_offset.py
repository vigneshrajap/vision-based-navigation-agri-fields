#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import glob
import pyexcel as pe
import tf
import sys
import math
from tf.transformations import euler_from_quaternion, unit_vector, quaternion_multiply, quaternion_conjugate
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped, Point, PoseStamped
from sensor_msgs.msg import NavSatFix, Imu
import geo2UTM

class offset_estimation():
    '''
    A class to find fitted center of the lane points along the crop rows given an input RGB image.
    '''
    def __init__(self):
        self.datum = [-6614855.745, -594362.895, 0.0] # Manual Datum (NEGATE if UTM is the child frame)
        self.gps_robot = [0.425, -0.62, 1.05]
        self.gt_utm = []
        self.book = pe.get_book(file_name="/home/vignesh/planner_ws/src/vision-based-navigation-agri-fields/auto_nav/config/ground_truth_coordinates.xls")
        self.lane_number = str(1) #rospy.set_param('lane_number', 1)

        self.map_frame = str('map')
        self.utm_frame = str('utm')
        self.robot_frame = str('base_link')
        self.gps_frame = str('gps')
        self.imu_frame = str('xsens')
        self.gps_topic_name = str('/gps/fix')
        self.imu_topic_name = str('/imu/data')
        self.receive_gps_fix = False
        self.gps_fix = NavSatFix()
        self.imu_data = Imu()
        self.north = []
        self.east = []
        self.gt_map_x = []
        self.gt_map_y = []
        self.coefficients = []
        self.gt_line = []
        self.orientation_imu = []
        self.rot_vec = []

        for row in self.book["Sheet"+self.lane_number]:
                self.gt_utm.append([row[1],row[2]]) # Latitude, Longitude

        self.listener = tf.TransformListener()
        self.map_transform = TransformStamped()
        self.map_transform.header.stamp = rospy.Time.now()
        self.map_transform.header.frame_id = self.map_frame
        self.map_transform.child_frame_id = self.utm_frame
        self.map_transform.transform.translation = Vector3(self.datum[0], self.datum[1], self.datum[2]);
        self.map_transform.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        #Listen to image messages and publish predictions with callback
        self.fix_sub = rospy.Subscriber(self.gps_topic_name, NavSatFix, self.gpsFixCallback)
        self.imu_sub = rospy.Subscriber(self.imu_topic_name, Imu, self.imuDataCallback)

    def gpsFixCallback(self, fix_msg):
        # Read fix messages
        self.gps_fix = fix_msg
        self.receive_gps_fix = True

    def imuDataCallback(self, imu_msg):
        # Read fix messages
        self.imu_data = imu_msg
        self.orientation_imu = [self.imu_data.orientation.x, self.imu_data.orientation.y, self.imu_data.orientation.z, self.imu_data.orientation.w]
        (roll_imu, pitch_imu, yaw_imu) = euler_from_quaternion(self.orientation_imu)
        #self.receive_imu_fix = True

    def ground_truth_utm2map(self):

        for i in range(1, len(self.gt_utm)): # Skip the first row (String)
          pose_stamped = PoseStamped()
          pose_stamped.pose.position = Point(self.gt_utm[i][0], self.gt_utm[i][1], 0)
          pose_stamped.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

          pose_transformed = PoseStamped()
          pose_transformed.header.stamp = rospy.Time.now()
          pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.map_transform) # Transform RTK values w.r.t to "Map" frame

          self.gt_map_x.append(pose_transformed.pose.position.x)
          self.gt_map_y.append(pose_transformed.pose.position.y)

        # Fitting the straight line and obtain a, b, c co-ordinates
        self.coefficients = np.polyfit(self.gt_map_x, self.gt_map_y, 1)
        self.gt_line = [oe.coefficients[0], -1, oe.coefficients[1]] # a, b, c

    # rotate vector v1 by quaternion q1
    def qv_mult(self):
        q1 = self.orientation_imu
        q2 = list([self.gps_robot[0],self.gps_robot[1],self.gps_robot[2],0.0])

        return quaternion_multiply(
            quaternion_multiply(q1, q2), quaternion_conjugate(q1)
        )

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

               gps_fix_utm = geo2UTM.geo2UTM(oe.gps_fix.latitude, oe.gps_fix.longitude) # Custom Library to convert geo to UTM co-ordinates

               gps_pose_utm = PoseStamped()
               gps_pose_utm.pose.position = Point(gps_fix_utm[0], gps_fix_utm[1], 0)
               gps_pose_utm.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

               gps_pose_map = PoseStamped()
               gps_pose_map.header.stamp = rospy.Time.now()
               gps_pose_map = tf2_geometry_msgs.do_transform_pose(gps_pose_utm, oe.map_transform) # Transform RTK values w.r.t to "Map" frame

               # Min distance from point (current robot gps) to the line (ground truth)
               dist_n = abs(oe.gt_line[0]*gps_pose_map.pose.position.x+oe.gt_line[1]*gps_pose_map.pose.position.y+ oe.gt_line[2])
               dist_d = math.sqrt(pow(oe.gt_line[0],2)+pow(oe.gt_line[1],2))
               dist_0 = dist_n/dist_d

               # try:
               #    (trans,rot) = oe.listener.lookupTransform(oe.robot_frame, oe.gps_frame, rospy.Time(0))
               # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
               #    continue

               #print rot_vec #roll_imu, pitch_imu, yaw_imu,

               oe.rot_vec = oe.qv_mult()

               gps_transform = TransformStamped()
               gps_transform.header.stamp = rospy.Time.now()
               gps_transform.header.frame_id = oe.robot_frame
               gps_transform.child_frame_id = oe.gps_frame
               gps_transform.transform.translation = Vector3(oe.gps_robot[0], oe.gps_robot[1], 0.0) #oe.gps_robot[2]#(oe.rot_vec[0], oe.rot_vec[1], oe.rot_vec[2])  #-0.425, 0.62, -1.05)
               gps_transform.transform.rotation = Quaternion(0,0,0,1) # Set to identity

               robot_pose_map = PoseStamped()
               robot_pose_map.header.stamp = rospy.Time.now()
               robot_pose_map = tf2_geometry_msgs.do_transform_pose(gps_pose_map, gps_transform) # Transform RTK values w.r.t to "Map" frame

               # Min distance from point (current robot gps) to the line (ground truth)
               dist_n_1 = abs(oe.gt_line[0]*robot_pose_map.pose.position.x+oe.gt_line[1]*robot_pose_map.pose.position.y+ oe.gt_line[2])
               dist_d_1 = math.sqrt(pow(oe.gt_line[0],2)+pow(oe.gt_line[1],2))
               dist_0_1 = dist_n_1/dist_d_1

               print dist_0 #gps_pose_map.pose.position, dist_0
               print dist_0_1 #robot_pose_map.pose.position , dist_0_1

           r.sleep()

    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
