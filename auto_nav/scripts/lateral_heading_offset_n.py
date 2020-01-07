#!/usr/bin/env python
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import rospy
import numpy as np
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
        self.image_topic_name = str('/camera/color/image_raw')
        self.receive_gps_fix = False
        self.receive_imu_fix = False
        self.img_receive = False

        self.gps_fix = NavSatFix()
        self.imu_data = Imu()
        self.image = Image()
        self.bridge = CvBridge()

        self.north = []
        self.east = []
        self.gt_map_x = []
        self.gt_map_y = []
        self.coefficients = []
        self.gt_line = []

        self.orientation_imu = []
        self.yaw_imu = []

        self.rot_vec = []

        self.gps_oldTime = []
        self.gps_NewTime = []
        self.imu_oldTime = []
        self.imu_NewTime = []
        self.dt_gps = 0
        self.dt_imu = 0
        self.oneshot = 0

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
        self.image_sub = rospy.Subscriber(self.image_topic_name, Image, self.imageCallback)

    def gpsFixCallback(self, fix_msg):
        # Read fix messages
        self.gps_fix = fix_msg
        self.receive_gps_fix = True

    def imuDataCallback(self, imu_msg):
        # Read fix messages
        self.imu_data = imu_msg
        self.orientation_imu = [self.imu_data.orientation.x, self.imu_data.orientation.y, self.imu_data.orientation.z, self.imu_data.orientation.w]
        (roll_imu, pitch_imu, self.yaw_imu) = euler_from_quaternion(self.orientation_imu)
        self.receive_imu_fix = True

    def recv_image_msg(self, ros_data): #"passthrough"):
        try:
            self.image = self.bridge.imgmsg_to_cv2(ros_data,"bgr8")
            # print self.img_receive
        except CvBridgeError as e:
          print(e)

    def imageCallback(self, ros_data):
        #Read image
        self.recv_image_msg(ros_data)
        if(np.ndim(self.image) !=3 or np.shape(self.image)[2] !=3):
            rospy.logerr('Input image must have 3 dimensions with 3 color channels')

        # Preprocess
        self.img_receive = True

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
        #self.coefficients = np.polyfit(self.gt_map_x, self.gt_map_y, 1)
        #self.gt_line = [oe.coefficients[0], -1, oe.coefficients[1]] # a, b, c

        increment = 5
        inc = 0
        self.coefficients = np.empty([np.int(len(self.gt_map_x)/increment), 2])
        self.gt_line = np.empty([np.int(len(self.gt_map_x)/increment), 3])
        self.gt_yaw = np.empty([np.int(len(self.gt_map_x)/increment), 1])
        for ind in range(increment+1,len(self.gt_map_x),increment):
            self.coefficients[inc] = (np.polyfit(self.gt_map_x[ind-increment:ind], self.gt_map_y[ind-increment:ind], 1)) # SWITCH TO SECOND ORDER
            self.gt_line[inc, 0] = oe.coefficients[inc][0]
            self.gt_line[inc, 1] = -1
            self.gt_line[inc, 2] = oe.coefficients[inc][1] # a, b, c

            inc = inc + 1

        self.gt_yaw = self.coefficients[:,0] - self.coefficients[0][0]
        # print self.coefficients, self.gt_yaw

    # rotate vector v1 by quaternion q1
    def qv_mult(self):
        q1 = self.orientation_imu
        q2 = list([self.gps_robot[0],self.gps_robot[1],self.gps_robot[2],0.0])

        return quaternion_multiply(
            quaternion_multiply(q1, q2), quaternion_conjugate(q1))

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

               # if (oe.oneshot==0): # Set the Start time
               #    oe.gps_oldTime = oe.gps_fix.header.stamp.to_sec()
               #    oe.imu_oldTime = oe.imu_data.header.stamp.to_sec()
               #    oe.oneshot = 1
               #
               # # Current Time and Relative Time "dt"
               # oe.gps_NewTime = oe.gps_fix.header.stamp.to_sec()
               # oe.imu_NewTime = oe.imu_data.header.stamp.to_sec()
               # oe.dt_gps = oe.dt_gps + (oe.gps_NewTime - oe.gps_oldTime)
               # oe.dt_imu = oe.dt_imu + (oe.imu_NewTime - oe.imu_oldTime)

               gps_fix_utm = geo2UTM.geo2UTM(oe.gps_fix.latitude, oe.gps_fix.longitude) # Custom Library to convert geo to UTM co-ordinates

               gps_pose_utm = PoseStamped()
               gps_pose_utm.pose.position = Point(gps_fix_utm[0], gps_fix_utm[1], 0)
               gps_pose_utm.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

               gps_pose_map = PoseStamped()
               gps_pose_map.header.stamp = rospy.Time.now()
               gps_pose_map = tf2_geometry_msgs.do_transform_pose(gps_pose_utm, oe.map_transform) # Transform RTK values w.r.t to "Map" frame

               # # Min distance from point (current robot gps) to the line (ground truth)
               # dist_n = abs(oe.gt_line[0]*gps_pose_map.pose.position.x+oe.gt_line[1]*gps_pose_map.pose.position.y+ oe.gt_line[2])
               # dist_d = math.sqrt(pow(oe.gt_line[0],2)+pow(oe.gt_line[1],2))
               # dist_0 = dist_n/dist_d

               # try:
               #    (trans,rot) = oe.listener.lookupTransform(oe.robot_frame, oe.gps_frame, rospy.Time(0))
               # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
               #    continue

               oe.rot_vec = oe.qv_mult()

               gps_transform = TransformStamped()
               gps_transform.header.stamp = rospy.Time.now()
               gps_transform.header.frame_id = oe.robot_frame
               gps_transform.child_frame_id = oe.gps_frame
               gps_transform.transform.translation = Vector3(oe.gps_robot[0], oe.gps_robot[1], 0.0) #oe.gps_robot[2]#(oe.rot_vec[0], oe.rot_vec[1], oe.rot_vec[2])
               gps_transform.transform.rotation = Quaternion(0,0,0,1) # Set to identity

               robot_pose_map = PoseStamped()
               robot_pose_map.header.stamp = rospy.Time.now()
               robot_pose_map = tf2_geometry_msgs.do_transform_pose(gps_pose_map, gps_transform) # Transform RTK values w.r.t to "Map" frame

               # Min distance from point (current robot gps) to the line (ground truth)
               dist_0 = np.empty([np.int(len(oe.gt_line)),1])

               for c_n in range(0,len(oe.gt_line)):
                 dist_n = abs(oe.gt_line[c_n,0]*robot_pose_map.pose.position.x+oe.gt_line[c_n,1]*robot_pose_map.pose.position.y+ oe.gt_line[c_n,2])
                 dist_d = math.sqrt(pow(oe.gt_line[c_n,0],2)+pow(oe.gt_line[c_n,1],2))
                 dist_0[c_n] = dist_n/dist_d

               lateral_offset = np.min(dist_0)
               segment_index = np.where(dist_0 == np.min(dist_0))

               if oe.receive_imu_fix==True:
                   angular_offset = oe.gt_yaw[segment_index[0][0]]-oe.yaw_imu
                   print "lateral_offset:", lateral_offset, "angular_offset:", angular_offset #, oe.dt_gps, oe.dt_imu #, oe.gps_oldTime, newTime

               if oe.img_receive==True:
                   rospy.loginfo('Received image for prediction')

               oe.gps_oldTime = oe.gps_NewTime
               oe.imu_oldTime = oe.imu_NewTime
               oe.receive_gps_fix = False

    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
