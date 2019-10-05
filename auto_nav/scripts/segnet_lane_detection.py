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
sys.path.insert(2, '../../')
import lane_predict
from geometry_msgs.msg import Pose, PoseArray

# #--- ROS setup
# image_topic_name = rosparam.get_param('auto_nav/segnet_lane_detection/camera_topic')
# model_config = rosparam.getparam('auto_nav/segnet_lane_detection/model_config')
# model_weights = rosparam.getparam('auto_nav/segnet_lane_detection/model_weights')

#Initialize node
rospy.init_node('segnet_lane_detection')

#Set up publisher for prediction messages
pub = rospy.Publisher('centerline_local', PoseArray, queue_size = 10)
rate = rospy.Rate(1) # 10hz



visualize = "lane_fit" # "all"

#image = cv2.imread("/home/saga/Third_Paper/Datasets/Frogn_Fields/Frogn_004/frogn_40000.png")

#seg_arr, input_image, output_image, fit = lane_predict.predict_on_image(model, image, lane_fit = True, evaluate = False, visualize = None, output_file = None, display=False)

def recv_image_msg(image_msg, format = "bgr8"): #"passthrough"):
    cv_br = CvBridge()
    #rospy.loginfo('Receiving image')
    image = cv_br.imgmsg_to_cv2(image_msg,format)
    return image

def visualize_segmentation(input_img, seg_arr, n_classes, display = False, output_file = None, class_number = 2):
    seg_img = predict.segmented_image_from_prediction(seg_arr, n_classes = n_classes, input_shape = input_img.shape)
    overlay_img = cv2.addWeighted(input_img,0.7,seg_img,0.3,0)

    # Reshaping the Lanes Class into binary array and Upscaling the image as input image
    dummy_img = np.zeros(seg_arr.shape)
    dummy_img += ((seg_arr[:,: ] == class_number)*(255)).astype('uint8') # Class Number 2 belongs to Lanes
    original_h, original_w = overlay_img.shape[0:2]
    upscaled_img = cv2.resize(dummy_img, (original_w,original_h)).astype('uint8')
    upscaled_img_rgb = cv2.cvtColor(upscaled_img, cv2.COLOR_GRAY2RGB)

    # Stack input and segmentation in one video
    vis_img = np.vstack((
       np.hstack((input_img,
                  seg_img)),
       np.hstack((overlay_img,
                  upscaled_img_rgb)) #np.ones(overlay_img.shape,dtype=np.uint8)*128))
    ))

    return vis_img

def visualize_lane_fit(input_img, output_img, n_classes, display = False, output_file = None, class_number = 2):
    # Stack input and segmentation in one video
    vis_img = np.hstack((input_img,np.ones(input_img.shape,dtype=np.uint8)*128)) #output_img))

    return vis_img

def visualize_functions(visualize=None, input_image=None, seg_arr=None, output_image=None):
    # visualize: None, "all" or one of, "segmentation", "lane_fit"
    if visualize == "segmentation":
        vis_img = visualize_segmentation(input_image, seg_arr, model.n_classes, display=False, output_file=None, class_number=2)
        #cv2.imwrite('Segmentation.png', vis_img)

    elif visualize == "lane_fit":
        vis_img = visualize_lane_fit(input_image, output_image, model.n_classes, display=False, output_file=None, class_number=2)
        #cv2.imwrite('Lane_fit.png', vis_img)

    elif visualize == "all":
        vis_img = visualize_segmentation(input_image, seg_arr, model.n_classes, display=False, output_file=None, class_number=2)
        #cv2.imwrite('Segmentation.png', vis_img)
        vis_img = visualize_lane_fit(input_image, output_image, model.n_classes, display=False, output_file=None, class_number=2)
        #cv2.imwrite('Lane_fit.png', vis_img)

    else:
        vis_img = None

    cv2.destroyAllWindows()

def callback(image_msg):
    #Read image
    image = recv_image_msg(image_msg)
    if(np.ndim(image) !=3 or np.shape(image)[2] !=3):
        rospy.logerr('Input image must have 3 dimensions with 3 color channels')

    # Preprocess
    rospy.loginfo('Received image for prediction')

    model_prefix = "/home/saga/Third_Paper/segnet_weights/resnet50_segnet"
    epoch = 21
    #model = predict.model_from_checkpoint_files( model_prefix, epoch)#model_config, model_weights)
    model = predict.model_from_checkpoint_path(model_prefix, epoch = None)

    #Run prediction (and optional, visualization)
    seg_arr, input_image, output_image, fit = lane_predict.predict_on_image(model, image, lane_fit = False, evaluate = False, visualize = None, output_file = None, display=False)

    # while not rospy.is_shutdown():
    #     #print len(fit)
    #     if fit!=None :
    #        fit_points = PoseArray()
    #        fit_points.header.frame_id = "camera_color_optical_frame"
    #        fit_points.header.stamp = rospy.Time.now()
    #        for i in range(0,len(fit),4): # Increment by 4
    #            P = Pose()
    #            P.position.x = fit[i][0]
    #            P.position.y = fit[i][1]
    #            fit_points.poses.append(P)
    #        pub.publish(fit_points)
    #        rate.sleep()
    #
    # final_msg = visualize_functions(visualize, image, seg_arr=None, output_image=None)
    # cv2.startWindowThread()
    # cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('preview', 800,800)
    # cv2.imshow('preview', final_msg)
    #
    # # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #    print('Q pressed, breaking')
       #break
    #Make ros pred message
    #fixme return lane prediction from predict
    #lane_msg = "hello world %s" #tmp
    #publish prediction
    #rospy.loginfo(lane_msg)
    #pub.publish(lane_msg)
    #rate.sleep()



def predict_lane():
    #Listen to image messages and publish predictions with callback
    rospy.Subscriber("/camera/color/image_raw", Image, callback, queue_size = 10) #rename topic in launch? #image_topic_name

    rospy.spin()

if __name__ == '__main__':
    try:
        predict_lane()
    except rospy.ROSInterruptException:
        pass
