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
from os.path import expanduser
sys.path.insert(2, expanduser("~")+'/planner_ws/src/vision-based-navigation-agri-fields/')
import lane_predict
from geometry_msgs.msg import Pose, PoseArray

class lane_finder():
    '''
    A class to find fitted center of the lane points along the crop rows given an input RGB image.
    '''
    def __init__(self):

        #### Hyperparameters ####
        self.image = Image()
        self.output_image = Image()
        self.bridge = CvBridge()

        # ROS setup paramters
        self.image_topic_name = rosparam.get_param('auto_nav/segnet_lane_detection/camera_topic')
        self.model_config = rosparam.get_param('auto_nav/segnet_lane_detection/model_config')
        self.model_weights = rosparam.get_param('auto_nav/segnet_lane_detection/model_weights')
        self.model_prefix = rosparam.get_param('auto_nav/segnet_lane_detection/model_prefix')
        self.visualize = rosparam.get_param('auto_nav/segnet_lane_detection/visualize')

        self.output_video = True
        self.output_video_file_s = rosparam.get_param('auto_nav/segnet_lane_detection/output_video_file_s')
        self.output_video_file_l = rosparam.get_param('auto_nav/segnet_lane_detection/output_video_file_l')

        self.lane_fit = True
        self.evaluate = False
        self.output_file = None
        self.display = False
        self.seg_arr = []
        self.fit = []
        self.class_number = 2

        #self.loaded_weights = False
        self.img_receive = False
        self.epoch = None
        #model = predict.model_from_checkpoint_files( model_prefix, epoch)#model_config, model_weights)
        self.model = predict.model_from_checkpoint_path(self.model_prefix, self.epoch) #, self.loaded_weights

        #Listen to image messages and publish predictions with callback
        self.img_sub = rospy.Subscriber(self.image_topic_name, Image, self.imageCallback)

        #Set up publisher for prediction messages
        self.fit_pub = rospy.Publisher('centerline_local', PoseArray)
        self.rate = rospy.Rate(1) # 10hz

        # Saving outputs as Video file
        self.fourcc = cv2.VideoWriter_fourcc(*"MJPG") # ('I','Y','U','V') #tried('M','J','P','G')
        self.wr = None
        (self.out_h, self.out_w) = (None, None)
        self.isColor = True
        self.fps = 6

        self.fourcc_1 = cv2.VideoWriter_fourcc(*"MJPG") #('I','Y','U','V') #tried('M','J','P','G')
        self.wr_1 = None
        (self.out_h_1, self.out_w_1) = (None, None)
        self.isColor_1 = True
        self.fps_1 = 6

    def recv_image_msg(self, ros_data): #"passthrough"):
        try:
            self.image = self.bridge.imgmsg_to_cv2(ros_data,"bgr8")
            # print self.img_receive
            #return image
        except CvBridgeError as e:
          print(e)

    def imageCallback(self, ros_data):
        #Read image
        self.recv_image_msg(ros_data) #self.image =
        if(np.ndim(self.image) !=3 or np.shape(self.image)[2] !=3):
            rospy.logerr('Input image must have 3 dimensions with 3 color channels')

        # Preprocess
        rospy.loginfo('Received image for prediction')
        self.img_receive = True

    def visualize_segmentation(self): #input_img, seg_arr, n_classes, display = False, output_file = None, class_number = 2
        seg_img = predict.segmented_image_from_prediction(self.seg_arr, n_classes = self.model.n_classes, input_shape = self.image.shape)
        overlay_img = cv2.addWeighted(self.image,0.7,seg_img,0.3,0)

        # Reshaping the Lanes Class into binary array and Upscaling the image as input image
        dummy_img = np.zeros(self.seg_arr.shape)
        dummy_img += ((self.seg_arr[:,: ] == self.class_number)*(255)).astype('uint8') # Class Number 2 belongs to Lanes
        original_h, original_w = overlay_img.shape[0:2]
        upscaled_img = cv2.resize(dummy_img, (original_w,original_h)).astype('uint8')
        upscaled_img_rgb = cv2.cvtColor(upscaled_img, cv2.COLOR_GRAY2RGB)

        # Stack input and segmentation in one video
        vis_img = np.vstack((
           np.hstack((self.image,
                      seg_img)),
           np.hstack((overlay_img,
                      upscaled_img_rgb)) #np.ones(overlay_img.shape,dtype=np.uint8)*128))
        ))

        cv2.imshow('preview', vis_img)
        #cv2.waitKey(0)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
         print('Q pressed, breaking')

        return vis_img

    def visualize_lane_fit(self):
        # Stack input and segmentation in one video
        vis_img = np.hstack((self.image,self.output_image))
        return vis_img

    def visualize_functions(self):

        # visualize: None, "all" or one of, "segmentation", "lane_fit"
        if self.visualize == "segmentation":
            seg_result = self.visualize_segmentation()
            if self.output_video:
                if self.wr is None: #if writer is not set up yet
                   (self.out_h,self.out_w) = seg_result.shape[:2]
                   self.wr = cv2.VideoWriter(self.output_video_file_s,self.fourcc,self.fps,(self.out_w,self.out_h),self.isColor)
                   #cv2.startWindowThread()
                   #cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
                   #cv2.resizeWindow('preview', 800,800)
                    #break
                self.wr.write(seg_result)

        elif self.visualize == "lane_fit":
            lane_fit_result = self.visualize_lane_fit()
            #Write video
            if self.output_video:
                if self.wr_1 is None: #if writer is not set up yet
                   (self.out_h_1,self.out_w_1) = lane_fit_result.shape[:2]
                   self.wr_1 = cv2.VideoWriter(self.output_video_file_l,self.fourcc_1,self.fps_1,(self.out_w_1,self.out_h_1),self.isColor_1)
                self.wr_1.write(lane_fit_result)

        elif self.visualize == "all":
            seg_result = self.visualize_segmentation()
            lane_fit_result = self.visualize_lane_fit()
            if self.output_video:
                if self.wr is None: #if writer is not set up yet
                   (self.out_h,self.out_w) = seg_result.shape[:2]
                   self.wr = cv2.VideoWriter(self.output_video_file_s,self.fourcc,self.fps,(self.out_w,self.out_h),self.isColor)
                if self.wr_1 is None: #if writer is not set up yet
                   (self.out_h_1,self.out_w_1) = lane_fit_result.shape[:2]
                   self.wr_1 = cv2.VideoWriter(self.output_video_file_l,self.fourcc_1,self.fps_1,(self.out_w_1,self.out_h_1),self.isColor_1)
                self.wr.write(seg_result)
                self.wr_1.write(lane_fit_result)

        else:
            vis_img = None

    def pipeline(self):

      if self.img_receive: #and self.loaded_weights
        #Run prediction (and optional, visualization)
        self.seg_arr,self.image,self.output_image,self.fit = lane_predict.predict_on_image(self.model,self.image,self.lane_fit,self.evaluate,self.visualize,self.output_file,self.display)

        final_img = self.visualize_functions()

        #print len(fit)
        if self.fit!=None :
           fit_points = PoseArray()
           fit_points.header.frame_id = "camera_color_optical_frame"
           fit_points.header.stamp = rospy.Time.now()
           for i in range(0,len(self.fit),4): # Increment by 4
               P = Pose()
               P.position.x = self.fit[i][0]
               P.position.y = self.fit[i][1]
               fit_points.poses.append(P)
           self.fit_pub.publish(fit_points)
           self.rate.sleep()

if __name__ == '__main__':
    try:
        #Initialize node
        rospy.init_node('segnet_lane_detection', anonymous=True)
        lf = lane_finder()

        while not rospy.is_shutdown():
           if lf.img_receive==True:
               lf.pipeline()

               lf.img_receive = False

    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
