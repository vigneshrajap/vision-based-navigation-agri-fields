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
import timeit
import shapely.geometry as geom
import rospkg
from std_msgs.msg import Header
import csv

class automated_labelling():
    '''
    A class to find fitted center of the lane points along the crop rows given an input RGB image.
    '''
    def __init__(self):
        self.datum = [-6614855.745, -594362.895, 0.0] # Manual Datum (NEGATE if UTM is the child frame)
        self.gps_robot = [0.425, -0.62, 1.05] # Fixed Static Transform

        rospack = rospkg.RosPack()
        self.book = pe.get_book(file_name=rospack.get_path('auto_nav')+"/config/ground_truth_coordinates.xls", start_row=1)

        self.lane_number = str(4) #rospy.set_param('lane_number', 1)
        self.gt_utm = np.empty([self.book["Sheet"+self.lane_number].number_of_rows(), 2])
        self.gt_map = np.empty([self.book["Sheet"+self.lane_number].number_of_rows(), 2])

        row_ind = self.book["Sheet"+self.lane_number].row[0][0] # Get the index the first cell of the row
        for row in self.book["Sheet"+self.lane_number]:
               self.gt_utm[row[0]%row_ind] = ([row[1],row[2]]) # Latitude, Longitude

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
        self.img_timestamp = Header()
        self.bridge = CvBridge()

        self.orientation_imu = []
        self.yaw_imu = []
        self.lateral_offset = 0.0
        self.angular_offset = 0.0
        self.rot_vec = []
        self.pose_map_r = PoseStamped()
        self.increment = 10 # Fit Line segments over increment values
        self.line = geom.LineString()
        self.gt_yaw = 0

        self.gps_oldTime = []
        self.gps_NewTime = []
        self.imu_oldTime = []
        self.imu_NewTime = []
        self.img_oldTime = []
        self.img_NewTime = []
        self.img_oldSeq = []
        self.img_NewSeq = []
        self.dt_gps = 0
        self.dt_imu = 0
        self.dt_img = 0
        self.dt_imgSeq = 0
        self.oneshot = 0

        self.listener = tf.TransformListener()
        self.map_trans = TransformStamped()
        self.map_trans.header.stamp = rospy.Time.now()
        self.map_trans.header.frame_id = self.map_frame
        self.map_trans.child_frame_id = self.utm_frame
        self.map_trans.transform.translation = Vector3(self.datum[0], self.datum[1], self.datum[2]);
        self.map_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        # Subscribers Callback
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
            self.img_timestamp = ros_data.header
        except CvBridgeError as e:
          print(e)

    def imageCallback(self, ros_data):
        # Read image
        self.recv_image_msg(ros_data)
        if(np.ndim(self.image) !=3 or np.shape(self.image)[2] !=3):
            rospy.logerr('Input image must have 3 dimensions with 3 color channels')

        self.img_receive = True

    def qv_mult(self):
        # rotate vector v1 by quaternion q1
        q1 = self.orientation_imu
        q2 = list([self.gps_robot[0],self.gps_robot[1],self.gps_robot[2],0.0])

        return quaternion_multiply(
            quaternion_multiply(q1, q2), quaternion_conjugate(q1))

    def normalizeangle(self, bearing):
      if (bearing < -math.pi):
              bearing += 2 * math.pi
      elif (bearing > math.pi):
              bearing -= 2 * math.pi
      return bearing

    def ground_truth_utm2map(self):

        pose_stamped = PoseStamped()
        pose_trans = PoseStamped()

        for i in range(0, len(self.gt_utm)): # Skip the first row if it is a String
          pose_stamped.pose.position = Point(self.gt_utm[i][0], self.gt_utm[i][1], 0)
          pose_stamped.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

          pose_trans.header.stamp = rospy.Time.now()
          pose_trans = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.map_trans) # Transform RTK values w.r.t to "Map" frame

          self.gt_map[i] = ([pose_trans.pose.position.x, pose_trans.pose.position.y])

    def GNSS_WorldToRobot(self):

        # Custom Library to convert geo to UTM co-ordinates
        gps_fix_utm = geo2UTM.geo2UTM(auto_label.gps_fix.latitude, auto_label.gps_fix.longitude)

        gps_pose_utm = PoseStamped()
        gps_pose_utm.pose.position = Point(gps_fix_utm[0], gps_fix_utm[1], 0)
        gps_pose_utm.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

        # Transform RTK values w.r.t to "Map" frame
        gps_pose_map = PoseStamped()
        gps_pose_map.header.stamp = rospy.Time.now()
        gps_pose_map = tf2_geometry_msgs.do_transform_pose(gps_pose_utm, auto_label.map_trans)

        self.rot_vec = self.qv_mult() # Rotate the static offset value by IMU yaw
        #print self.rot_vec, self.gps_robot

        gps_trans = TransformStamped()
        gps_trans.header.stamp = rospy.Time.now()
        gps_trans.header.frame_id = self.robot_frame
        gps_trans.child_frame_id = self.gps_frame
        gps_trans.transform.translation = Vector3(self.rot_vec[0], self.rot_vec[1], 0.0) #(auto_label.rot_vec[0], auto_label.rot_vec[1], auto_label.rot_vec[2])
        gps_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        # Transform RTK values w.r.t to "Map" frame
        self.pose_map_r.header.stamp = rospy.Time.now()
        self.pose_map_r = tf2_geometry_msgs.do_transform_pose(gps_pose_map, gps_trans)

    def offset_estimation(self):

       dist_0 = np.empty([np.int(len(self.gt_utm)/self.increment),1])
       lines = geom.MultiLineString()
       multilines = []
       lines = []

       # Increment by parameter for multiple line segments along GT points
       for ind in range(self.increment+1,len(self.gt_utm),self.increment):

            self.line = geom.LineString(self.gt_map[ind-self.increment:ind,:])
            point = geom.Point(self.pose_map_r.pose.position.x, self.pose_map_r.pose.position.y) # x, y
            dist_0[np.int(ind/self.increment)-1] = self.line.distance(point)
            lines.append(self.line)

       multilines.append(geom.MultiLineString(lines))

       # Min Lateral Offset and its line segement index
       self.lateral_offset = np.min(dist_0)
       segment_index = np.where(dist_0 == np.min(dist_0))
       print "lateral_offset:", self.lateral_offset

       radius_of_curvature = multilines[0][segment_index[0][0]].length # Total Length of the line segment (min lateral offset)
       self.gt_yaw = 1/radius_of_curvature

       # Angular Offset => IMU with GT yaw (line segement index)
       if self.receive_imu_fix==True:
           self.angular_offset = self.normalizeangle(self.gt_yaw - self.yaw_imu)
           print "angular_offset:", self.angular_offset

       # RGB Image nearest to current GNSS fix msg
       if self.img_receive==True:
           rospy.loginfo('Received image for prediction')
           print

if __name__ == '__main__':
    try:
        #Initialize node
        rospy.init_node('lateral_heading_offset')
        auto_label = automated_labelling()

        # Function to obtain the ground truth values in Map frame
        auto_label.ground_truth_utm2map()

        myfile = open('offset_values.txt', 'a')
        myfile.truncate(0)
        myfile.write("dt(GPS)")
        myfile.write("\t")
        myfile.write("frame")
        myfile.write("\t")
        myfile.write("LO")
        myfile.write("\t")
        myfile.write("AO")
        myfile.write("\n")

        while not rospy.is_shutdown():

           if auto_label.img_receive == True and auto_label.receive_gps_fix== True:

               if (auto_label.oneshot==0): # Set the Start time
                  auto_label.gps_oldTime = auto_label.gps_fix.header.stamp.to_sec()
                  # auto_label.imu_oldTime = auto_label.imu_data.header.stamp.to_sec()
                  auto_label.img_oldTime = auto_label.img_timestamp.stamp.to_sec()
                  auto_label.img_oldSeq = auto_label.img_timestamp.seq
                  auto_label.oneshot = 1

               # # Current Time and Relative Time "dt"
               auto_label.gps_NewTime = auto_label.gps_fix.header.stamp.to_sec()
               # auto_label.imu_NewTime = auto_label.imu_data.header.stamp.to_sec()
               auto_label.img_NewTime = auto_label.img_timestamp.stamp.to_sec()
               auto_label.img_NewSeq = auto_label.img_timestamp.seq

               auto_label.dt_gps = auto_label.dt_gps + (auto_label.gps_NewTime - auto_label.gps_oldTime)
               # auto_label.dt_imu = auto_label.dt_imu + (auto_label.imu_NewTime - auto_label.imu_oldTime)
               auto_label.dt_img = auto_label.dt_img + (auto_label.img_NewTime - auto_label.img_oldTime)
               auto_label.dt_imgSeq = auto_label.dt_imgSeq + (auto_label.img_NewSeq - auto_label.img_oldSeq)

               # RTK Fix from UTM Frame to Robot Frame
               auto_label.GNSS_WorldToRobot()

               # Offset Estimation
               auto_label.offset_estimation()

               auto_label.gps_oldTime = auto_label.gps_NewTime
               # auto_label.imu_oldTime = auto_label.imu_NewTime
               auto_label.img_oldTime = auto_label.img_NewTime
               auto_label.img_oldSeq = auto_label.img_NewSeq

               myfile.write(str( "%.4f" % auto_label.dt_gps))
               myfile.write("\t")
               myfile.write(str(auto_label.dt_imgSeq))
               myfile.write("\t")
               myfile.write(str("%.4f" % auto_label.lateral_offset))
               myfile.write("\t")
               myfile.write(str("%.4f" % auto_label.angular_offset))
               myfile.write("\n")

               # auto_label.receive_gps_fix = False
               auto_label.img_receive = False
               auto_label.receive_imu_fix = False

               # print auto_label.dt_img  # auto_label.dt_gps, auto_label.dt_imu

    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
