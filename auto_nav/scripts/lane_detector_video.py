#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import numpy as np
import pandas as pd
import cv2
import os
import sys
import roslib
import matplotlib.pyplot as plt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from moviepy.editor import VideoFileClip
from os.path import expanduser
import pickle
import math
import tf
from numpy import linalg as LA
from os.path import expanduser
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseArray,Point
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf2_ros
import quaternion
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
from itertools import imap

prev_modifiedCenters = []

class lane_finder():
    '''
    A class to find lane points given an image that has been inverse perspective mapped and scrubbed of most features
    other than the lanes.
    '''

    def __init__(self, image, base_size=.2):
        #super(lane_finder, self).__init__()
    #### Hyperparameters ####
        self.image = image
        self.vis = image # used for visualization
        self.lanes = []
        self.base_size = base_size
        self.roi = [0, 300]  #roi_x, roi_y

        max_value_H = 360/2
        max_value = 255
        self.HSV_low = [0, 0, 0] #low_H, low_S, low_V
        self.HSV_high = [max_value_H, max_value, 145] #high_H, high_S, high_V

        #Finds the expected starting points  using K-Means
        self.clusters = 2
        self.base_size = 0.1 # random number
        self.cam_param_receive = True

        self.left_a = []
        self.left_b = []
        self.left_c = []
        self.right_a = []
        self.right_b = []
        self.right_c = []
        self.modifiedCenters = []

        self.Total_Points = 6
        self.Line_Pts = []
        self.listener = tf.TransformListener()
        self.init_transform = geometry_msgs.msg.TransformStamped()
        self.pub_poses = rospy.Publisher('vector_poses', PoseArray)

        #self.CameraInfo_sub = rospy.Subscriber("/kinect2_camera/rgb/camera_info", CameraInfo, self.imagecaminfoCallback)

    # def imagecaminfoCallback(self, data):
    #     global cam_param_receive, K
    #     K = data.K
    #     cam_param_receive = True
    #     print K

    def perspective_warp(self, img, dst_size, src, dst): # Choose the four vertices

        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped, M

    def inv_perspective_warp(self, img, dst_size, src, dst):
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped, M

    def camera2world(self, x_c, t_c, R_c):

     # ray in world coordinates
     x_c_q = np.quaternion(0, x_c[0], x_c[1], x_c[2])
     x_wq = R_c*x_c_q*R_c.conjugate()
     x_w = np.array([x_wq.x,x_wq.y,x_wq.z])

     # distance to the plane
     ## d = dot((t_p - t_c),n_p)/dot(x_w,n_p)
     ## simplified expression assuming plane t_p = [0 0 0]; n_p = [0 0 1];
     d = -t_c[2]/x_w[2]

     # intersection point
     x_wd = np.array([(x_w[0]*d),(x_w[1]*d),(x_w[2]*d)])
     x_p = np.add(x_wd, t_c)

     return x_p

    def normalizeangle(self, bearing): # Normalize the bearing

        if (bearing < -math.pi):
               bearing += 2 * math.pi
        elif (bearing > math.pi):
               bearing -= 2 * math.pi
        return bearing

    def initialPoints(self, img):

         global prev_modifiedCenters
         # Crop the search space
         bottom = (img.shape[0] - int(self.base_size * img.shape[0]))
         base = img[bottom:img.shape[0], 0:img.shape[1]]

         # Find white pixels
         whitePixels = np.argwhere(base == 255)

         # Attempt to run kmeans (the kmeans parameters were not chosen with any sort of hard/soft optimization)
         try:
             kmeans = KMeans(n_clusters=self.clusters, random_state=0, n_init=3, max_iter=150).fit(whitePixels)
         except:
              # If kmeans fails increase the search space unless it is the whole image, then it fails
              if self.base_size  > 1:
                  return None
              else:
                  self.base_size  = self.base_size  * 1.5
                  return self.initialPoints(self.clusters)

         # conver centers to integer values so can be used as pixel coords
         centers = [list(imap(int, center)) for center in kmeans.cluster_centers_]
         # Lamda function to remap the y coordiates of the clusters into the image space
         increaseY = lambda points: [points[0] + int((1 - self.base_size) * img.shape[0]), points[1]]

         # map the centers in terms of the image space
         self.modifiedCenters = [increaseY(center) for center in centers]

         if abs(self.modifiedCenters[0][1]-self.modifiedCenters[1][1])<50:
             #print self.modifiedCenters, prev_modifiedCenters
             self.modifiedCenters = prev_modifiedCenters
             return self.modifiedCenters

         prev_modifiedCenters = self.modifiedCenters

         # return a list of tuples for centers
         return self.modifiedCenters

    def sliding_window(self, img, nwindows=15, margin=50, minpix=1, draw_windows=True):
        left_fit_= np.empty(3)
        right_fit_ = np.empty(3)
        out_img = np.dstack((img, img, img))*255

        modifiedCenters = self.initialPoints(img)
        leftx_base = modifiedCenters[0][1]
        rightx_base = modifiedCenters[1][1]

        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
          # Identify window boundaries in x and y (and right and left)
          win_y_low = img.shape[0] - (window+1)*window_height
          win_y_high = img.shape[0] - window*window_height
          win_xleft_low = leftx_current - margin
          win_xleft_high = leftx_current + margin
          win_xright_low = rightx_current - margin
          win_xright_high = rightx_current + margin

          # Identify the nonzero pixels in x and y within the window
          good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
          good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
          (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

          # Append these indices to the lists
          left_lane_inds.append(good_left_inds)
          right_lane_inds.append(good_right_inds)

          # If you found > minpix pixels, recenter next window on their mean position
          if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
          if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

          # Draw the windows on the visualization image
          if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                (255,0,0), 3)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (255,0,0), 3)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_a.append(left_fit[0])
        self.left_b.append(left_fit[1])
        self.left_c.append(left_fit[2])

        self.right_a.append(right_fit[0])
        self.right_b.append(right_fit[1])
        self.right_c.append(right_fit[2])

        left_fit_[0] = np.mean(self.left_a[-10:])
        left_fit_[1] = np.mean(self.left_b[-10:])
        left_fit_[2] = np.mean(self.left_c[-10:])

        right_fit_[0] = np.mean(self.right_a[-10:])
        right_fit_[1] = np.mean(self.right_b[-10:])
        right_fit_[2] = np.mean(self.right_c[-10:])

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
        right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 100, 255] #[255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

    def pipeline(self):

        # Convert BGR to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]

        # Getting ROI
        Roi = hsv[self.roi[1]:height,self.roi[0]:width] #-(roi_y+1),-(roi_x+1)

        # define range of blue color in HSV
        lower_t = np.array([self.HSV_low[0],self.HSV_low[1],self.HSV_low[2]])
        upper_t = np.array([self.HSV_high[0],self.HSV_high[1],self.HSV_high[2]])

        # Detect the object based on HSV Range Values
        mask = cv2.inRange(Roi, lower_t, upper_t)

        # Opening the image
        kernel = np.ones((7,7),np.uint8)
        eroded = cv2.erode(mask, kernel, iterations = 1) # eroding + dilating = opening
        wscale = cv2.dilate(eroded, kernel, iterations = 1)
        ret, thresh = cv2.threshold(wscale, 128, 255, cv2.THRESH_BINARY_INV) # thresholding the image //THRESH_BINARY_INV

        # # Perspective warp
        rheight, rwidth = thresh.shape[:2]
        dst_size =(rheight,rwidth)
        src=np.float32([(0.05,0), (1,0), (0.05,1), (1,1)])
        dst=np.float32([(0,0), (1,0), (0,1), (1,1)])
        warped_img, M  = self.perspective_warp(thresh, dst_size, src, dst)

        # Sliding Window Search
        out_img, curves, lanes, ploty = self.sliding_window(warped_img)

        # Fitted curves as points
        leftLane = np.array([np.transpose(np.vstack([curves[0], ploty]))])
        rightLane = np.array([np.flipud(np.transpose(np.vstack([curves[1], ploty])))])
        points = np.hstack((leftLane, rightLane))
        curves_m = (curves[0]+curves[1])/2
        midLane = np.array([np.transpose(np.vstack([curves_m, ploty]))])

        leftLane_i = leftLane[0].astype(int)
        rightLane_i = rightLane[0].astype(int)
        midLane_i = midLane[0].astype(int)

        cv2.polylines(out_img, [leftLane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)
        cv2.polylines(out_img, [rightLane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)
        cv2.polylines(out_img, [midLane_i], 0, (255,0,255), thickness=5, lineType=8, shift=0)

        dst_size =(rwidth, rheight)
        invwarp, Minv = self.inv_perspective_warp(out_img, dst_size, dst, src)

        midPoints = []
        for i in midLane_i:
          point_wp = np.array([i[0],i[1],1])
          midLane_io = np.matmul(Minv, point_wp) # inverse-M*warp_pt
          midLane_n = np.array([midLane_io[0]/midLane_io[2],midLane_io[1]/midLane_io[2]]) # divide by Z point
          midLane_n = midLane_n.astype(int)
          midPoints.append(midLane_n)

        # Combine the result with the original image
        self.image[self.roi[1]:height,self.roi[0]:width] = cv2.addWeighted(self.image[self.roi[1]:height,self.roi[0]:width],
                                                                           1, invwarp, 0.9, 0)
        result = self.image

        return warped_img, midPoints, out_img, result

    def publish_vector(self, centerLine, t_c, R_c, K_arr):

         # Used to publish waypoints as pose array so that you can see them in rviz, etc.
         poses = PoseArray()
         poses.header.frame_id = "map"
         poses.header.stamp = rospy.Time.now()

         for pt in range(self.Total_Points):

           # Line segment points
           seg_x = int((centerLine[0][0]*(1-(float(pt)/self.Total_Points))) + (centerLine[len(centerLine)-1][0]*(float(pt)/self.Total_Points)))
           seg_y = int((centerLine[0][1]*(1-(float(pt)/self.Total_Points))) + (centerLine[len(centerLine)-1][1]*(float(pt)/self.Total_Points)))

           # Calcuate 3D World Point from 2D Image Point
           p_c = np.array([seg_x+self.roi[0], seg_y+self.roi[1], 1])
           x_c = np.linalg.inv(K_arr).dot(p_c) # Applying Intrinsic Parameters
           x_c_norm = LA.norm(x_c, axis=0)
           x_c = x_c/x_c_norm # Normalize the vector
           x_p = self.camera2world(x_c, t_c, R_c)
           self.Line_Pts.append([x_p[0],x_p[1]])

           if pt>0:
              position = Point(self.Line_Pts[pt][0], self.Line_Pts[pt][1], 0) # Position
              yaw = math.atan2(self.Line_Pts[pt-1][1]-self.Line_Pts[pt][1],self.Line_Pts[pt-1][0]-self.Line_Pts[pt][0])
              quaternion_c = tf.transformations.quaternion_from_euler(0, 0, self.normalizeangle(yaw))
              orientation = np.quaternion(quaternion_c[3], quaternion_c[0], quaternion_c[1], quaternion_c[2])
              poses.poses.append(Pose(position, orientation))

         return poses

if __name__ == '__main__':
   try:
     rospy.init_node('lane_detector_video', anonymous=True)

     home = expanduser("~/ICRA_2020/wheel_tracks_n.avi")
     cap = cv2.VideoCapture(home)

     # Check if camera opened successfully
     if (cap.isOpened()== False):
      print("Error opening video stream or file")

     # Read until video is completed
     while(cap.isOpened()):

      # Capture frame-by-frame
      ret, rgb_img = cap.read()
      if ret == True:
       while not rospy.is_shutdown():
         lf = lane_finder(rgb_img, base_size=.2)
         warped_img, centerLine, curve_fit_img, output = lf.pipeline()

         if lf.cam_param_receive==True:

             try:
                #wait for the transform to be found
                #lf.listener.waitForTransform("map", "kinect2_rgb_optical_frame", rospy.Time(0)), rospy.Duration(3.0))
                (trans,rot) = lf.listener.lookupTransform("map", "kinect2_rgb_optical_frame", rospy.Time(0))

             except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                  continue

             t_c = np.array([[trans[0]], [trans[1]], [trans[2]]])
             R_c = np.quaternion(rot[3], rot[0], rot[1], rot[2]) # Format: (w,x,y,z)

             K_arr = [[1065, 0, 980],
                      [0, 1065, 540],
                      [0, 0, 1]]

             #lf.publish_vector(centerLine, t_c, R_c, K_arr)
             poses = lf.publish_vector(centerLine, t_c, R_c, K_arr)

             # Publish the vector of poses
             lf.pub_poses.publish(poses)

             cv2.startWindowThread()
             cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
             cv2.resizeWindow('preview', 800,800)
             cv2.imshow('preview', output)

    #       # Plotting the data
    #       # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    #       # ax1.set_title('Original', fontsize=10)
    #       # ax1.xaxis.set_visible(False)
    #       # ax1.yaxis.set_visible(False)
    #       # ax1.imshow(frame, aspect="auto")
    #       # camera = Camera(f)
    #       # ax2.set_title('Filter+Perspective Tform', fontsize=10)
    #       # ax2.xaxis.set_visible(False)
    #       # ax2.yaxis.set_visible(False)
    #       # ax2.imshow(warped_img, aspect="auto")
    #       #
    #       # ax3.plot(curves[0], ploty, color='yellow', linewidth=5)
    #       # ax3.plot(curves[1], ploty, color='yellow', linewidth=5)
    #       # ax3.xaxis.set_visible(False)
    #       # ax3.yaxis.set_visible(False)
    #       # ax3.set_title('Sliding window+Curve Fit', fontsize=10)
    #       # ax3.imshow(out_img, aspect="auto")
    #       #
    #       # ax4.set_title('Overlay Lanes', fontsize=10)
    #       # ax4.xaxis.set_visible(False)
    #       # ax4.yaxis.set_visible(False)
    #       # ax4.imshow(result, aspect="auto")
    #       #
    #       # # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #       # animation.FuncAnimation(f, ax1, interval=2, blit=True)
    #       # animation = camera.animate()
    #       # plt.show()
    #       rospy.sleep(1)  # sleep for one second

         # # Press Q on keyboard to  exit
         # if cv2.waitKey(25) & 0xFF == ord('q'):
         #    break
         # else:  # Break the loop
         #    break
         #
     # When everything done, release the video capture object
     cap.release()
         #
         # # Closes all the frames
         # cv2.destroyAllWindows()

   except rospy.ROSInterruptException:
     pass
