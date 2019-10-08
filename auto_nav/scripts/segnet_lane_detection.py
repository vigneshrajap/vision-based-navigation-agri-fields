#!/usr/bin/env python

# Predict view angles for an image, using the trailNet CNN.
# Input: camera image
# Output: Array of angles and probabilities

import rospy
import rosparam
#from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
#Keras and tensorflow
import sys
sys.path.insert(1, '../../../image-segmentation-keras/')
from keras_segmentation import predict
sys.path.insert(1, '../../')
from lane_predict import predict_on_image

from std_msgs.msg import String

#--- ROS setup
image_topic_name = rosparam.get_param('auto_nav/segnet_lane_detection/camera_topic')
model_config = rosparam.getparam('auto_nav/segnet_lane_detection/model_config')
model_weights = rosparam.getparam('auto_nav/segnet_lane_detection/model_weights')
#Initialize node
rospy.init_node('segnet_lane_detection')

#Set up publisher for prediction messages
pub = rospy.Publisher('lane_pred', String, queue_size = 1) #fixme other type
rate = rospy.Rate(10) # 10hz 

model = predict.model_from_checkpoint_files(model_config, model_weights)

def recv_image_msg(image_msg,format = "passthrough"):
    cv_br = CvBridge()
    #rospy.loginfo('Receiving image')
    image = cv_br.imgmsg_to_cv2(image_msg,format)
    return image

#%%
def callback(image_msg):
    
    #Read image
    image = recv_image_msg(image_msg)
    if(np.ndim(image) !=3 or np.shape(image)[2] !=3): 
        rospy.logerr('Input image must have 3 dimensions with 3 color channels')

    # Preprocess
    rospy.loginfo('Received image for prediction')

    # Prediction
    lane_predict.predict_on_image(model,inp=image,lane_fit = True, evaluate = False, visualize = "all", output_file = None, display=True) #fixme vignesh output variables
    
    #fixme add ros message
    lane_msg = "hello world %s" #tmp
    #publish prediction
    rospy.loginfo(lane_msg)
    pub.publish(lane_msg)
    
    rate.sleep()
    
#fixme visualization function

def predict_lane():
    #Listen to image messages and publish predictions with callback
    rospy.Subscriber(image_topic_name, Image, callback, queue_size = 1) #rename topic in launch?
    rospy.spin()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    try:
        predict_lane()
    except rospy.ROSInterruptException:
        pass