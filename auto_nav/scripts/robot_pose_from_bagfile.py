#!/usr/bin/env python
from sensor_msgs.msg import Image
import rospy
import numpy as np
import glob
import pyexcel as pe
import tf
import sys
import math
from tf.transformations import euler_from_quaternion, unit_vector, quaternion_multiply, quaternion_conjugate, quaternion_from_euler
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
import rosbag
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import os.path as osp
import matplotlib.pyplot as plt

class automated_labelling():
    '''
    A class to find fitted center of the lane points along the crop rows given an input RGB image.
    '''
    def __init__(self):
        self.datum = [-6614855.745, -594362.895, 0.0] # Manual Datum (NEGATE if UTM is the child frame)
        self.gps_robot = [-0.425, 0.62, 1.05] # Fixed Static Transform
        self.imu_robot = [0.310, 0.00, 0.80] # Fixed Static Transform

        rospack = rospkg.RosPack()
        self.book = pe.get_book(file_name=rospack.get_path('auto_nav')+"/config/ground_truth_coordinates.xls", start_row=1)

        self.lane_number = str(3) #rospy.set_param('lane_number', 1) #!!!input
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

        self.gps_fix = NavSatFix()
        self.image = Image()
        self.img_timestamp = Header()
        self.magnetic_declination = 0.0523599

        self.orientation_imu = []
        self.lateral_offset = 0.0
        self.angular_offset = 0.0
        self.rot_vec = []
        self.rpos_map = PoseStamped()
        self.prev_rpos_map = PoseStamped()
        self.increment = 5 # Fit Line segments over increment values
        self.line = geom.LineString()
        self.gt_yaw = 0
        self.prev_rpos_map.pose.position.x = 0
        self.prev_rpos_map.pose.position.y = 0
        self.robot_yaw = 0.0
        self.delta_robot_yaw = 0.0
        self.prev_robot_yaw = 0.0
        self.robot_dir = []
        self.robot_imu = []
        self.rpos_x = []
        self.rpos_y = []
        self.count_ind = 0
        self.first_dir = 24
        self.multilines = []

        self.gps_oldTime = []
        self.gps_NewTime = []
        self.imu_oldTime = []
        self.imu_NewTime = []
        self.img_oldTime = []
        self.img_NewTime = []
        self.img_oldSeq = []
        self.img_NewSeq = []

        self.oneshot_gps = 0
        self.oneshot_imu = 0
        self.oneshot_img = 0
        self.dt_gps = 0
        self.dt_imu = 0
        self.dt_img = 0

        self.listener = tf.TransformListener()
        self.map_trans = TransformStamped()
        self.map_trans.header.stamp = rospy.Time.now()
        self.map_trans.header.frame_id = self.map_frame
        self.map_trans.child_frame_id = self.utm_frame
        self.map_trans.transform.translation = Vector3(self.datum[0], self.datum[1], self.datum[2]);
        self.map_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

    def ground_truth_utm2map(self):

        pose_stamped = PoseStamped()
        pose_trans = PoseStamped()

        for i in range(0, len(self.gt_utm)): # Skip the first row if it is a String
          pose_stamped.pose.position = Point(self.gt_utm[i][0], self.gt_utm[i][1], 0)
          pose_stamped.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

          pose_trans.header.stamp = rospy.Time.now()
          pose_trans = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.map_trans) # Transform RTK values w.r.t to "Map" frame

          self.gt_map[i] = ([pose_trans.pose.position.x, pose_trans.pose.position.y])

        lines = []

        # Increment by parameter for multiple line segments along ground truth points
        for ind in range((self.increment/2),len(self.gt_utm),self.increment):
            self.line = geom.LineString(self.gt_map[ind-(self.increment/2):ind+(self.increment/2),:])
            lines.append(self.line)
        self.multilines.append(geom.MultiLineString(lines))

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

        # Rotate the static offset value by IMU yaw
        self.rot_vec = self.qv_mult()

        gps_trans = TransformStamped()
        gps_trans.header.stamp = rospy.Time.now()
        gps_trans.header.frame_id = self.robot_frame
        gps_trans.child_frame_id = self.gps_frame
        gps_trans.transform.translation = Vector3(self.rot_vec[0], self.rot_vec[1], 0.0) #self.rot_vec[0], self.rot_vec[1] #self.gps_robot[0],self.gps_robot[1]
        gps_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        # Transform RTK values w.r.t to "Map" frame
        self.rpos_map.header.stamp = rospy.Time.now()
        self.rpos_map = tf2_geometry_msgs.do_transform_pose(gps_pose_map, gps_trans)

if __name__ == '__main__':
    #Initialize node
    rospy.init_node('lateral_heading_offset')

    auto_label = automated_labelling()

    # Function to obtain the ground truth values in Map frame
    auto_label.ground_truth_utm2map()

    input_dir = os.path.join('/media/marianne/Seagate Expansion Drive/data/20191010_bagfiles/dataset_9') #!!!input
    bag_files = sorted(glob.glob(osp.join(input_dir, '*.bag')))

    cnt = 0 #debug
    for bag_file in bag_files:
        print(bag_file)

        bag = rosbag.Bag(bag_file)

        ##################### Extract IMU Data #####################
        auto_label.imu_fix_ = []
        auto_label.dt_imu_fix_ = []

        for imu_topic, imu_msg, t_imu in bag.read_messages(topics=[auto_label.imu_topic_name]):

                orientation_imu_orginal = [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]
                (roll_r_imu, pitch_r_imu, yaw_r_imu) = euler_from_quaternion(orientation_imu_orginal)
                yaw_r_imu = yaw_r_imu + auto_label.magnetic_declination # Compensate

                if(auto_label.oneshot_imu==0):
                    t0_imu = t_imu.to_sec()
                    auto_label.oneshot_imu = 1
                    auto_label.receive_imu_fix = True
                    prev_yaw_r_imu = yaw_r_imu
                    delta_yaw_r_imu = 0 # Set Initial Yaw

                auto_label.dt_imu = auto_label.dt_imu + (t_imu.to_sec()-t0_imu)

                delta_yaw_r_imu = delta_yaw_r_imu + auto_label.normalizeangle(yaw_r_imu-prev_yaw_r_imu) # Rate of change of IMU yaw

                auto_label.orientation_imu = quaternion_from_euler(0, 0, yaw_r_imu-prev_yaw_r_imu)
                auto_label.dt_imu_fix_.append(auto_label.dt_imu)
                auto_label.imu_fix_.append(delta_yaw_r_imu)

                t0_imu = t_imu.to_sec()
                prev_yaw_r_imu = yaw_r_imu
        
        ##################### Extract GNSS Data #####################
        for gps_topic, gps_msg, t_gps in bag.read_messages(topics=[auto_label.gps_topic_name]):
                auto_label.gps_fix.latitude = gps_msg.latitude
                auto_label.gps_fix.longitude = gps_msg.longitude
                t0 = t_gps.to_sec()

                # RTK Fix from UTM Frame to Robot Frame
                auto_label.GNSS_WorldToRobot()
                print('gps_t0 and rpos_map',t0, auto_label.rpos_map.pose.position.x, auto_label.rpos_map.pose.position.y)

        ##################### Extract Camera Data #####################
        '''
        seq0_img = None
        for topic, img_msg, t_img in bag.read_messages(topics=[auto_label.image_topic_name]):
            if(seq0_img is None): #first frame
                seq0_img = img_msg.header.seq
            seq_img = img_msg.header.seq - seq0_img #relative frame number
            t_img_sec = t_img.to_sec() #aboslute time
            print("t_img",t_img_sec)
            print('dt img seq', seq_img)
        '''
        bag.close()