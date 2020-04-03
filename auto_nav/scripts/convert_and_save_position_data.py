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
from sensor_msgs.msg import NavSatFix
import geo2UTM
#import timeit
#import shapely.geometry as geom
import rospkg
from std_msgs.msg import Header
import csv
import rosbag
import os
from namedtuples_csv import write_namedtuples_to_csv

from collections import namedtuple
#os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
#from os.path import expanduser
#import os.path as osp
#import matplotlib.pyplot as plt

#Read and save GPS ground truth, robot pose and camera images for further processing 

class automated_labelling():
    '''
    Remains from the original implementation in lateral heading offset...
    '''
    def __init__(self):
        self.datum = [-6614855.745, -594362.895, 0.0] # Manual Datum (NEGATE if UTM is the child frame)
        self.gps_robot = [-0.425, 0.62, 1.05] # Fixed Static Transform
        self.imu_robot = [0.310, 0.00, 0.80] # Fixed Static Transform

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

        #map transform
        self.listener = tf.TransformListener()
        self.map_trans = TransformStamped()
        self.map_trans.header.stamp = rospy.Time.now()
        self.map_trans.header.frame_id = self.map_frame
        self.map_trans.child_frame_id = self.utm_frame
        self.map_trans.transform.translation = Vector3(self.datum[0], self.datum[1], self.datum[2]);
        self.map_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        #ground truth
        self.gt_utm = []
        self.gt_map = []

        #robot position
        self.rpos_map = PoseStamped()

    def read_utm_gt_from_xls(self,lane_number):
        #Open xls file
        rospack = rospkg.RosPack()
        book = pe.get_book(file_name=rospack.get_path('auto_nav')+"/config/ground_truth_coordinates.xls", start_row=1)
        lane_number = str(lane_number)

        self.gt_utm = np.empty([book["Sheet"+lane_number].number_of_rows(), 2])

        row_ind = book["Sheet"+lane_number].row[0][0] # Get the index the first cell of the row
        for row in book["Sheet"+lane_number]:
               self.gt_utm[row[0]%row_ind] = ([row[1],row[2]]) # Latitude, Longitude

    def ground_truth_utm2map(self):
        self.gt_map = np.empty(self.gt_utm.shape)

        pose_stamped = PoseStamped()
        pose_trans = PoseStamped()

        for i in range(0, len(self.gt_utm)): # Skip the first row if it is a String
          pose_stamped.pose.position = Point(self.gt_utm[i][0], self.gt_utm[i][1], 0)
          pose_stamped.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

          pose_trans.header.stamp = rospy.Time.now()
          pose_trans = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.map_trans) # Transform RTK values w.r.t to "Map" frame

          self.gt_map[i] = ([pose_trans.pose.position.x, pose_trans.pose.position.y])

    def normalizeangle(self, bearing):
        if (bearing < -math.pi):
                  bearing += 2 * math.pi
        elif (bearing > math.pi):
                  bearing -= 2 * math.pi
        return bearing

    def GNSS_WorldToRobot(self):

        # Custom Library to convert geo to UTM co-ordinates
        gps_fix_utm = geo2UTM.geo2UTM(self.gps_fix.latitude, self.gps_fix.longitude)

        gps_pose_utm = PoseStamped()
        gps_pose_utm.pose.position = Point(gps_fix_utm[0], gps_fix_utm[1], 0)
        gps_pose_utm.pose.orientation = Quaternion(0,0,0,1) # GPS orientation is meaningless, set to identity

        # Transform RTK values w.r.t to "Map" frame
        gps_pose_map = PoseStamped()
        gps_pose_map.header.stamp = rospy.Time.now()
        gps_pose_map = tf2_geometry_msgs.do_transform_pose(gps_pose_utm, self.map_trans)

        gps_trans = TransformStamped()
        gps_trans.header.stamp = rospy.Time.now()
        gps_trans.header.frame_id = self.robot_frame
        gps_trans.child_frame_id = self.gps_frame
        gps_trans.transform.translation = Vector3(self.gps_robot[0],self.gps_robot[1], 0.0) #self.rot_vec[0], self.rot_vec[1] #self.gps_robot[0],self.gps_robot[1]
        gps_trans.transform.rotation = Quaternion(0,0,0,1) # Set to identity

        # Transform RTK values w.r.t to "Map" frame
        self.rpos_map.header.stamp = rospy.Time.now()
        self.rpos_map = tf2_geometry_msgs.do_transform_pose(gps_pose_map, gps_trans)


if __name__ == '__main__':
    #user inputs
    input_dir = os.path.join('/media/marianne/Seagate Expansion Drive/data/20191010_bagfiles/dataset_9') #!!!input
    lane_number = 3
    row_prefix = '20191010_L3_S_morning_slaloam'

    output_dir = './output'
    bag_files = sorted(glob.glob(os.path.join(input_dir, '*.bag')))
    #bag_files = ['/media/marianne/Seagate Expansion Drive/data/20191010_bagfiles/dataset_9/dataset_recording_2019-10-10-12-14-31_37.bag'] #debug
    #Initialize node
    rospy.init_node('lateral_heading_offset')

    auto_label = automated_labelling()

    ############ Read and convert ground truth values ###################

    auto_label.read_utm_gt_from_xls(lane_number)
    auto_label.ground_truth_utm2map()
    
    GTPos = namedtuple('GTPos',['x','y'])
    gt_map_positions = []
    for row in auto_label.gt_map:
        gt_map_positions.append(GTPos(x=row[0],y=row[1]))

    ############ Read from bagfiles #################
    RobotPos = namedtuple('RobotPos',['x','y','time'])
    robot_map_positions = []
    ImageMeta = namedtuple('ImageFrame',['frame_num','time','filename'])
    image_meta_list = []
    
    seq0_img = None

    for bag_file in bag_files:
        print('Opening ' + bag_file)

        bag = rosbag.Bag(bag_file)

        ##################### Extract GNSS Data #####################
        for gps_topic, gps_msg, t_gps in bag.read_messages(topics=[auto_label.gps_topic_name]):
                auto_label.gps_fix.latitude = gps_msg.latitude
                auto_label.gps_fix.longitude = gps_msg.longitude

                # RTK Fix from UTM Frame to Robot Frame
                auto_label.GNSS_WorldToRobot()
                
                rp = RobotPos(x = auto_label.rpos_map.pose.position.x, y = auto_label.rpos_map.pose.position.y, time = t_gps.to_sec())
                robot_map_positions.append(rp)
        
        ##################### Extract Camera Data #####################
        
        for topic, img_msg, t_img in bag.read_messages(topics=[auto_label.image_topic_name]):
            if(seq0_img is None): #first frame
                seq0_img = img_msg.header.seq
            seq_img = img_msg.header.seq - seq0_img #relative frame number
            t_img_sec = t_img.to_sec() #aboslute time

            #Read and save image to file
            im_filename = row_prefix + '_' +  str(seq_img) + '.png'
            im_path = os.path.join(output_dir, im_filename)
            #fixme read image and save directly to png file?

            #accumulate metadata
            im_meta = ImageMeta(frame_num = seq_img, time = t_img_sec, filename = im_filename)
            image_meta_list.append(im_meta)
               
        bag.close()

######## Write to csv files

#Ground truth
gt_pos_file = os.path.join(output_dir, row_prefix + '_gt_pos.csv')
print('Writing ground truth positions to '+ gt_pos_file)
write_namedtuples_to_csv(gt_pos_file,gt_map_positions)

# Robot positions
robot_pos_file = os.path.join(output_dir,row_prefix + '_robot_pos_and_timestamps.csv')
print('Writing robot positions to '+ robot_pos_file)
write_namedtuples_to_csv(robot_pos_file,robot_map_positions)

#Image frames
img_meta_file = os.path.join(output_dir,row_prefix + '_image_timestamps.csv')
print('Writing image timestamps to '+ img_meta_file)
write_namedtuples_to_csv(img_meta_file,image_meta_list)


