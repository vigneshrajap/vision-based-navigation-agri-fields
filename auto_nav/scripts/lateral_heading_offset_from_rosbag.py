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

class automated_labelling():
    '''
    A class to find fitted center of the lane points along the crop rows given an input RGB image.
    '''
    def __init__(self):
        self.datum = [-6614855.745, -594362.895, 0.0] # Manual Datum (NEGATE if UTM is the child frame)
        self.gps_robot = [0.425, -0.62, 1.05] # Fixed Static Transform
        self.imu_robot = [0.310, 0.00, 0.80] # Fixed Static Transform

        rospack = rospkg.RosPack()
        self.book = pe.get_book(file_name=rospack.get_path('auto_nav')+"/config/ground_truth_coordinates.xls", start_row=1)

        self.lane_number = str(2) #rospy.set_param('lane_number', 1)
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
        self.prev_pose_map_r = PoseStamped()
        self.increment = 5 # Fit Line segments over increment values
        self.line = geom.LineString()
        self.gt_yaw = 0
        self.prev_pose_map_r.pose.position.x = 0
        self.prev_pose_map_r.pose.position.y = 0
        self.robot_yaw = []

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

        self.imu_trans = TransformStamped()
        self.imu_trans.header.stamp = rospy.Time.now()
        self.imu_trans.header.frame_id = self.robot_frame
        self.imu_trans.child_frame_id = self.imu_frame
        self.imu_trans.transform.translation = Vector3(self.imu_robot[0], self.imu_robot[1], 0);
        self.imu_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

    def ground_truth_utm2map(self):

        pose_stamped = PoseStamped()
        pose_trans = PoseStamped()

        for i in range(0, len(self.gt_utm)): # Skip the first row if it is a String
          pose_stamped.pose.position = Point(self.gt_utm[i][0], self.gt_utm[i][1], 0)
          pose_stamped.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

          pose_trans.header.stamp = rospy.Time.now()
          pose_trans = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.map_trans) # Transform RTK values w.r.t to "Map" frame

          self.gt_map[i] = ([pose_trans.pose.position.x, pose_trans.pose.position.y])

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

        self.rot_vec = self.qv_mult() # Rotate the static offset value by IMU yaw

        gps_trans = TransformStamped()
        gps_trans.header.stamp = rospy.Time.now()
        gps_trans.header.frame_id = self.robot_frame
        gps_trans.child_frame_id = self.gps_frame
        gps_trans.transform.translation = Vector3(self.rot_vec[0], self.rot_vec[1], 0.0) #self.gps_robot[0],self.gps_robot[1]
        gps_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        # Transform RTK values w.r.t to "Map" frame
        self.pose_map_r.header.stamp = rospy.Time.now()
        self.pose_map_r = tf2_geometry_msgs.do_transform_pose(gps_pose_map, gps_trans)

    def offset_estimation(self):

       # Initialize Parameters
       dist_0 = np.empty([np.int(len(self.gt_utm)/self.increment),1])
       lines = geom.MultiLineString()
       multilines = []
       lines = []

       # Increment by parameter for multiple line segments along ground truth points
       for ind in range((self.increment/2),len(self.gt_utm),self.increment):
            self.line = geom.LineString(self.gt_map[ind-(self.increment/2):ind+(self.increment/2),:])
            point = geom.Point(self.pose_map_r.pose.position.x, self.pose_map_r.pose.position.y) # x, y
            dist_0[np.int(ind/self.increment)-1] = self.line.distance(point)
            lines.append(self.line)

       multilines.append(geom.MultiLineString(lines))

       ############################## LATERAL OFFSET ###########################
       # Min Lateral Offset and its line segement index
       self.lateral_offset = np.min(dist_0)
       segment_index = np.where(dist_0 == np.min(dist_0))
       aX, aY, bX, bY = multilines[0][segment_index[0][0]].bounds
       cX, cY = (self.pose_map_r.pose.position.x, self.pose_map_r.pose.position.y)

       if ((bX - aX)*(cY - aY) - (bY - aY)*(cX - aX)) > 0:
           self.lateral_offset = -self.lateral_offset
       # print "lateral_offset:", self.lateral_offset

       ############################## ANGULAR OFFSET ###########################

       # Estimate slope and perpendicular slope of the nearest line segement
       gt_slope = (bY-aY)/(bX-aX)
       gt_yaw = self.normalizeangle(math.atan2(bY-aY,bX-aX))
       # gt_slope_normal = -1/gt_slope
       # # gt_yaw_normal = self.normalizeangle(math.atan(gt_slope_normal))
       # gt_yaw_normal = self.normalizeangle(math.atan2(aX-bX,bY-aY))
       #
       if (self.pose_map_r.pose.position.x!=self.prev_pose_map_r.pose.position.x) and ((self.pose_map_r.pose.position.y!=self.prev_pose_map_r.pose.position.y)):
            self.robot_yaw = self.normalizeangle(math.atan2(self.pose_map_r.pose.position.y-self.prev_pose_map_r.pose.position.y,self.pose_map_r.pose.position.x-self.prev_pose_map_r.pose.position.x))
            self.prev_pose_map_r = self.pose_map_r

       # # Interpolate the nearest point on the line and find slope of that line
       # point = geom.Point(self.pose_map_r.pose.position.x, self.pose_map_r.pose.position.y)
       # nearest_line = geom.LineString(multilines[0][segment_index[0][0]])
       # point_on_line = nearest_line.interpolate(nearest_line.project(point))
       # robot_slope = (point.y-point_on_line.y)/(point.x-point_on_line.x)

       # Angle between two lines as offset
       self.angular_offset = self.normalizeangle(self.robot_yaw-gt_yaw)

       ########################################################################################33

       #math.atan2((gt_slope_normal-robot_slope),(1+(gt_slope_normal*robot_slope))))

       # print AO #self.gt_yaw, self.yaw_imu, AO

       # # Angular Offset => IMU with GT yaw (line segement index)
       # if self.receive_imu_fix==True:
       #     self.angular_offset = self.normalizeangle(AO) #(math.pi-self.yaw_imu)
       #     # print "angular_offset:", self.angular_offset

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

        myfile = open('20191010_L2_N_offsets.txt', 'a')
        myfile.truncate(0)
        myfile.write("dt(cam)")
        myfile.write("\t")
        myfile.write("frame")
        myfile.write("\t")
        myfile.write("LO")
        myfile.write("\t")
        myfile.write("AO")
        myfile.write("\n")

        input_dir = expanduser("~/Third_Paper/Datasets/20191010_L2_N/bag_files/")

        for bag_file in sorted(glob.glob(osp.join(input_dir, '*.bag'))):
            print(bag_file)

            bag = rosbag.Bag(bag_file) #'/home/vignesh/Third_Paper/Datasets/20191010_L1_N/bag_files/dataset_recording_2019-10-10-14-52-14_0.bag')

            # GNSS Data
            auto_label.gps_fix_ = []
            auto_label.dt_gps_fix_ = []

            for topic, msg, t in bag.read_messages(topics=[auto_label.gps_topic_name]):
                 if(auto_label.oneshot_gps==0):
                     t0 = t.to_sec()
                     auto_label.oneshot_gps = 1

                 auto_label.dt_gps = auto_label.dt_gps + (t.to_sec()-t0)
                 auto_label.dt_gps_fix_.append(auto_label.dt_gps)
                 auto_label.gps_fix_.append([msg.latitude, msg.longitude])
                 t0 = t.to_sec()

            # IMU data
            auto_label.imu_fix_ = []
            auto_label.dt_imu_fix_ = []

            for topic, imu_msg, t_imu in bag.read_messages(topics=[auto_label.imu_topic_name]):
                 if(auto_label.oneshot_imu==0):
                     t0_imu = t_imu.to_sec()
                     auto_label.oneshot_imu = 1
                     auto_label.receive_imu_fix = True
                     prev_yaw_map_imu = 0
                     auto_label.first_line = geom.LineString(auto_label.gt_map[0:auto_label.increment,:])
                     first_aX, first_aY, first_bX, first_bY = auto_label.first_line.bounds
                     #delta_yaw_map_imu = auto_label.normalizeangle(math.atan2(first_bY-first_aY,first_bX-first_aX)) # Set Initial Yaw
                     delta_yaw_map_imu = 0 # Set Initial Yaw

                 auto_label.dt_imu = auto_label.dt_imu + (t_imu.to_sec()-t0_imu)

                 imu_data = imu_msg
                 orientation_imu_orginal = [imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w]
                 (roll_imu, pitch_imu, yaw_imu) = euler_from_quaternion(orientation_imu_orginal)

                 # Transform IMU values w.r.t to "base_link" frame
                 pose_imu = PoseStamped()
                 pose_imu.pose.position = Point(0, 0, 0) # IMU position is meaningless, set to zero
                 pose_imu.pose.orientation = imu_data.orientation

                 # Transform IMU values w.r.t to "Map" frame
                 pose_map_imu = PoseStamped()
                 pose_map_imu.header.stamp = rospy.Time.now()
                 pose_map_imu = tf2_geometry_msgs.do_transform_pose(pose_imu, auto_label.imu_trans)
                 orientation_map_imu = [pose_map_imu.pose.orientation.x, pose_map_imu.pose.orientation.y, pose_map_imu.pose.orientation.z, pose_map_imu.pose.orientation.w]

                 (roll_map_imu, pitch_map_imu, yaw_map_imu) = euler_from_quaternion(orientation_map_imu)
                 delta_yaw_map_imu = delta_yaw_map_imu + (yaw_map_imu-prev_yaw_map_imu) # Rate of change of IMU yaw
                 auto_label.orientation_imu = quaternion_from_euler(0,0,yaw_map_imu-prev_yaw_map_imu)

                 # print yaw_imu, yaw_map_imu, delta_yaw_map_imu

                 auto_label.dt_imu_fix_.append(auto_label.dt_imu)
                 auto_label.imu_fix_.append(delta_yaw_map_imu)
                 t0_imu = t_imu.to_sec()
                 prev_yaw_map_imu = yaw_map_imu

            # Image data
            auto_label.img_ = []
            auto_label.dt_img_ = []
            auto_label.dt_imgSeq_ = []

            for topic, img_msg, t_img in bag.read_messages(topics=[auto_label.image_topic_name]):
                 if(auto_label.oneshot_img==0):
                     t0_img = t_img.to_sec()
                     auto_label.img_oldSeq = img_msg.header.seq
                     auto_label.oneshot_img = 1

                 auto_label.dt_img = auto_label.dt_img + (t_img.to_sec()-t0_img)
                 auto_label.img_NewSeq = img_msg.header.seq

                 auto_label.dt_imgSeq = auto_label.img_NewSeq - auto_label.img_oldSeq

                 auto_label.dt_img_.append(auto_label.dt_img)
                 auto_label.dt_imgSeq_.append(auto_label.dt_imgSeq)
                 t0_img = t_img.to_sec()

            for i in range(len(auto_label.dt_img_)):

                gps_idx = (np.abs(np.array(auto_label.dt_gps_fix_) - auto_label.dt_img_[i])).argmin()
                auto_label.gps_fix.latitude = auto_label.gps_fix_[gps_idx][0]
                auto_label.gps_fix.longitude = auto_label.gps_fix_[gps_idx][1]

                imu_idx = (np.abs(np.array(auto_label.dt_imu_fix_) - auto_label.dt_img_[i])).argmin()
                # auto_label.yaw_imu = auto_label.imu_fix_[imu_idx]

                # RTK Fix from UTM Frame to Robot Frame
                auto_label.GNSS_WorldToRobot()

                # Offset Estimation
                auto_label.offset_estimation()

                myfile.write(str( "%.4f" % auto_label.dt_img_[i]))
                myfile.write("\t")
                myfile.write(str("%04d" %auto_label.dt_imgSeq_[i]))
                myfile.write("\t")
                myfile.write(str("%.4f" % auto_label.lateral_offset))
                myfile.write("\t")
                myfile.write(str("%.4f" % auto_label.angular_offset))
                myfile.write("\n")

            bag.close()

    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
