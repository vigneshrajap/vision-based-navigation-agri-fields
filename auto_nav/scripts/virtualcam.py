#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:33:42 2018

@author: marianne
"""

#Simplified virtual camera 
#For extracting reduced central FOV from camera with wide-FOV  lens

import numpy as np
import rospy
import cv2
import rosparam
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from utils import recv_image_msg,disp_img,crop_img
import VirtualCam

image_topic_name = rosparam.get_param('trail_net/virtualcam/camera_topic')
output_RPY = rosparam.get_param('trail_net/virtualcam/RPY')
#image_topic_name = "/image_folder_publisher/image" #"/pylon_camera_node/image_raw"
#Initialize node
rospy.init_node('virtualcam')

#Load camera models and set output angle 
VirtualCam.LoadCameraModels(r"/home/marianne/catkin_ws/src/trailnet_on_thorvald/trail_net/scripts/input_cam_model_campus_2018-09-21.xml", r"/home/marianne/catkin_ws/src/trailnet_on_thorvald/trail_net/scripts/output_cam_model.xml")
#VirtualCam.SetOutputRPY(0, 0, 0) #Roll, pitch, yaw. Always 0,0,0, output angle for live testing
VirtualCam.SetOutputRPY(output_RPY[0],output_RPY[1],output_RPY[2])
remap_image_dims = (480, 640, 3)

#Set up publisher for prediction messages
pub = rospy.Publisher('virtualcam/image_virtualcam', Image, queue_size = 1)
rate = rospy.Rate(10) # 10hz 
cv_br = CvBridge()

def callback(image_msg):
    #Read image
    image = recv_image_msg(image_msg)
    #Remap image
    remapped_image = np.zeros(remap_image_dims, np.uint8)
    VirtualCam.RemapImage(image, remapped_image)
    #Make ROS image message
    image_msg = cv_br.cv2_to_imgmsg(remapped_image, encoding = "bgr8")
    pub.publish(image_msg)
    rate.sleep()
    
    
def virtualcam_simple():
    #Listen to image messages and publish predictions with callback
    rospy.Subscriber(image_topic_name, Image, callback, queue_size = 1) #rename topic in launch?
    rospy.spin()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    try:
        virtualcam_simple()
    except rospy.ROSInterruptException:
        pass
