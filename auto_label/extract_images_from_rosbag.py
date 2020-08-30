#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

#Source: https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9

"""Extract images from a rosbag.
"""

import os
import glob
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def extract_1cam_data(input_dir,bag_file):
    bag = rosbag.Bag(os.path.join(input_dir,bag_file), "r")
    #topics for cam1, cam2 and cam3
    camera_topics = ['/basler_camera/image_raw']
    camera_output_folders = ['basler']

    output_dirs = []
    #Set up output folders
    for cf in camera_output_folders:
        output_dir = os.path.join(input_dir,cf)
        os.mkdir(output_dir)
        output_dirs.append(output_dir)


    bridge = CvBridge()
    for i in range(0,len(camera_topics)):
        count = 0
        for topic, msg, t in bag.read_messages(topics=[camera_topics[i]]):
            output_dir = output_dirs[i]
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
            print "Wrote image %i from topic %s" % (count,camera_topics[i])
            count += 1
    bag.close()

def extract_images(input_dir,bag_file,camera_topic,output_name):
    bag = rosbag.Bag(os.path.join(input_dir,bag_file), "r")
    print os.path.join(input_dir,bag_file)
    #topics for cam1, cam2 and cam3
    #camera_topics = ['/camera/basler_camera/image_raw']
    #camera_output_folders = ['cam2']
    output_dir = os.path.join(input_dir,output_name)
    os.mkdir(output_dir)

    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[camera_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        print "Wrote image %i from topic %s" % (count,camera_topic)
        count += 1
    bag.close()

def main():
    """Extract a folder of images from a rosbag. Assumes one bagfile per input dir
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--input_dir",default = '.', help="Input dir with rosbag.")
    #parser.add_argument("bag_file",help = "Rosbag filename.")
    parser.add_argument("--camera_topic",default = '/camera/image_raw', help= "Camera topic name.")
    parser.add_argument("--output_name",default = 'camera', help = "Name of ouput folder.")

#   parser.add_argument("bag_file", help="Input ROS bag.")
#   parser.add_argument("output_dir", help="Output directory.")
#   parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    bag_file = os.path.basename(glob.glob(args.input_dir+'/*'+'.bag')[0])
    print "Extract images from %s in %s" % (bag_file,args.input_dir)
    extract_images(args.input_dir, bag_file, args.camera_topic, args.output_name)

    return

if __name__ == '__main__':
    main()
