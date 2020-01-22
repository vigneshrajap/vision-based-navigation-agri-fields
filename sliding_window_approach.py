#!/usr/bin/env python
"""
Created on Sat Jan 18 16:31:13 2020
@author: vignesh raja
@description: This node handles the adaptive sliding window functions
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans
from itertools import imap
import matplotlib.pyplot as plt
import math
from sensor_msgs.msg import Image

class sliding_window():

    def __init__(self):

        self.img = []
        self.result_left = []
        self.result_right = []
        self.fitx_ = []
        self.sw_end = []
        self.increment = 5
        self.margin_sw = 20
        self.nwindows = 10
        self.win_y_low = [None]*self.nwindows
        self.win_y_high = [None]*self.nwindows
        self.minpix = 1
        self.whitePixels_thers = 0.50
        self.search_complete = False

        # Semicircle
        self.Leftangle = 0
        self.Rightangle = 180
        self.LeftstartAngle = -90 # Semicircle
        self.LeftendAngle = -270
        self.RightstartAngle = 90 # Semicircle
        self.RightendAngle = 270
        self.color_ellipse = (0,255,0)
        self.thickness_ellipse = 5
        self.radius = [] # Change based on window height
        self.axes = []

        # Decreasing Window Size
        self.win_size = np.arange(1.0, 0.5, -0.05)
        self.margin_l=110 # Varies on each crop row
        self.margin_r=110
        self.draw_windows=True

        self.mask_example = []

    def perspective_warp(self, img, dst_size, src, dst): # Choose the four vertices
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        dst = dst * np.float32(dst_size) # For destination points

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)

        return warped, M

    def Adaptive_sliding_window(self, out_img, window, x_current,  nonzeroy, nonzerox):

        # Create a Search Window
        k = l = 0
        percent_white_pixels_el = percent_white_pixels_er = 1.0
        self.search_complete = False
        col_ind = np.int((self.win_y_low[window]+self.win_y_high[window])/2)
        center_left = center_right = (x_current, col_ind)
        el_area = er_area = float((math.pi*math.pow(self.radius+10,2))/2)

        while (self.search_complete == False) and (center_left[0] >= 0) and (center_right[0] <= self.img.shape[1]):

           if (percent_white_pixels_el > self.whitePixels_thers):

             # create a white filled ellipse
             mask_left = np.zeros_like(self.img)
             center_left = (np.int(x_current-self.increment*k), col_ind)

             mask_left = cv2.ellipse(mask_left, center_left, self.axes, self.Leftangle, self.LeftstartAngle, self.LeftendAngle, 255, -1)

             # Bitwise AND operation to black out regions outside the mask
             self.result_left = np.bitwise_and(self.img, mask_left)

             percent_white_pixels_el = float(cv2.countNonZero(self.result_left)/el_area)
             k += 1

           if (percent_white_pixels_er > self.whitePixels_thers):

             # create a white filled ellipse
             mask_right = np.zeros_like(self.img)
             center_right = (np.int(x_current + self.increment*l), col_ind)

             mask_right = cv2.ellipse(mask_right, center_right, self.axes, self.Rightangle,  self.RightstartAngle, self.RightendAngle, 255, -1)

             # Bitwise AND operation to black out regions outside the mask
             self.result_right = np.bitwise_and(self.img, mask_right)

             percent_white_pixels_er = float(cv2.countNonZero(self.result_right)/er_area)
             l += 1

           if (percent_white_pixels_el <= self.whitePixels_thers) and (percent_white_pixels_er <= self.whitePixels_thers):

               #if (k>2) and (l>2):
               self.search_complete = True
               # else:
               #      percent_white_pixels_el = 1.0
               #      percent_white_pixels_er = 1.0

            # cv2.rectangle(out_img,(sw_xleft_low,win_y_low),(sw_xleft_high,win_y_high), (0,0,255), 5)
            # cv2.rectangle(out_img,(sw_x_low,win_y_low),(sw_x_high,win_y_high), (0,0,255), 5)

        self.mask_example = cv2.ellipse(self.mask_example, center_left, self.axes, self.Leftangle, self.LeftstartAngle, self.LeftendAngle, 255, -1)
        self.mask_example = cv2.ellipse(self.mask_example, center_right, self.axes, self.Rightangle, self.RightstartAngle, self.RightendAngle, 255, -1)

        # cv2.imshow('Bitwise XOR', mask_right)
        # # De-allocate any associated memory usage
        # if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()

        # Update the Window Size based on New Margins
        margin_ll = abs(center_left[0]-x_current)
        margin_rr = abs(center_right[0]-x_current)

        win_x_low = x_current - margin_ll
        win_x_high = x_current + margin_rr

        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy>=self.win_y_low[window]) & (nonzeroy<self.win_y_high[window]) &
                       (nonzerox>=win_x_low) & (nonzerox<win_x_high)).nonzero()[0]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > self.minpix:
              x_current = np.int(np.mean(nonzerox[good_inds]))

        cv2.line(out_img, (int(win_x_low), self.win_y_low[window]), (int(win_x_high), self.win_y_low[window]), (0,255,0), self.thickness_ellipse)
        cv2.line(out_img, (int(win_x_low), self.win_y_high[window]), (int(win_x_high), self.win_y_high[window]), (0,255,0), self.thickness_ellipse)

        cv2.ellipse(out_img, center_left, self.axes, self.Leftangle, self.LeftstartAngle, self.LeftendAngle, self.color_ellipse, self.thickness_ellipse)
        cv2.ellipse(out_img, center_right, self.axes, self.Rightangle, self.RightstartAngle, self.RightendAngle, self.color_ellipse, self.thickness_ellipse)
        cv2.circle(out_img, (x_current, col_ind),0, (0,0,255), thickness=25, lineType=8, shift=0) #+win_y_high[window])/2

        return good_inds, out_img

    def sliding_window(self, img, modifiedCenters):

        # Creates a list containing 3 lists, each of [] items, all set to 0
        self.img = img
        out_img = np.dstack((img, img, img))*255
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        self.mask_example = np.zeros_like(self.img)

        if len(modifiedCenters[0]):

            ############################ Initialize Paramters ############################
            # Set height of windows
            window_height = np.int(img.shape[0]/self.nwindows)

            self.radius = window_height-20
            self.axes = (self.radius+10, self.radius)

            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            template = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
            self.fitx_ = [[]for y in range(len(modifiedCenters[0]))]
            x_ = [[]for y in range(len(modifiedCenters[0]))]
            y_ = [[]for y in range(len(modifiedCenters[0]))]
            total_xpoints = [[]for y in range(len(modifiedCenters[0]))]
            total_ypoints = [[]for y in range(len(modifiedCenters[0]))]
            combined_rect = [[]for y in range(len(modifiedCenters[0]))]

            # Identify window boundaries in x and y (and right and left)
            self.win_y_low = img.shape[0] - np.multiply(range(1,self.nwindows+1), window_height)
            self.win_y_high = img.shape[0] - np.multiply(range(0,self.nwindows), window_height)

            #area = float(window_height*self.margin_sw*2)
            area_img = float(img.shape[0]*img.shape[1])
            ############################ Parameters ############################

            for p_in in range(len(modifiedCenters)):

              # Current positions to be updated for each window
              x_current = modifiedCenters[p_in][0]

              # Create empty lists to receive left and right lane pixel indices
              lane_inds_n = []
              good_inds_el_n = []
              good_inds_er_n = []

              for window in range(self.nwindows):

                win_x_low = int(x_current - self.win_size[window]*self.margin_l) #
                win_x_high = int(x_current + self.win_size[window]*self.margin_r) #
                # print x_current, win_x_low, win_x_high, self.margin_l, self.margin_r

                if win_x_low < 0:
                    win_x_low = 0

                if win_x_high > self.img.shape[0]:
                    win_x_low = self.img.shape[0]

                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= self.win_y_low[window]) & (nonzeroy < self.win_y_high[window]) &
                            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds) > self.minpix:
                      x_current = np.int(np.mean(nonzerox[good_inds]))

                # Proposed Adaptive Sliding Window
                #good_inds, out_img = self.Adaptive_sliding_window(out_img, window, x_current, nonzeroy, nonzerox)

                # Append these indices to the lists
                if len(good_inds):
                     lane_inds_n.append(good_inds)

                if len(self.result_left) and len(self.result_right):
                     if cv2.countNonZero(self.result_left)>0:
                         good_inds_el_n.append(cv2.findNonZero(self.result_left))
                     if cv2.countNonZero(self.result_right)>0:
                         good_inds_er_n.append(cv2.findNonZero(self.result_right))

                if self.draw_windows == True:
                #     if len(self.result_left) and len(self.result_right):
                #         cv2.line(out_img, (win_x_low, self.win_y_low[window]), (win_x_high, self.win_y_low[window]), (0,255,0), self.thickness_ellipse)
                #         cv2.line(out_img, (win_x_low, self.win_y_high[window]), (win_x_high, self.win_y_high[window]), (0,255,0), self.thickness_ellipse)
                #
                        cv2.line(out_img, (0, self.win_y_low[window]), (img.shape[1],  self.win_y_low[window]), (0,0,255), 2)
                #     else:
                        cv2.rectangle(out_img,(win_x_low,self.win_y_low[window]),(win_x_high,self.win_y_high[window]), (0,255,0), 6)

                # self.sw_end.append([p_in, x_current, margin_ll, margin_rr])

              if len(lane_inds_n): # Concatenate the arrays of indices
                  lane_inds_n = np.concatenate(lane_inds_n)

                  # Extract left and right line pixel positions
                  x_[p_in] = nonzerox[lane_inds_n]
                  y_[p_in] = nonzeroy[lane_inds_n]

                  out_img[nonzeroy[lane_inds_n], nonzerox[lane_inds_n]] = [255, 0, 0] #[255, 0, 100]

              if len(good_inds_el_n) and len(good_inds_er_n):
                  # print good_inds_er_n, len(good_inds_el_n)
                  good_inds_el_n = np.concatenate(good_inds_el_n)
                  good_inds_er_n = np.concatenate(good_inds_er_n)

                  out_img[good_inds_el_n[:,:,1], good_inds_el_n[:,:,0]] = [255, 0, 0] #[255, 0, 100]
                  out_img[good_inds_er_n[:,:,1], good_inds_er_n[:,:,0]] = [255, 0, 0] #[255, 0, 255]

              # if len(lane_inds_n) and len(good_inds_el_n) and len(good_inds_er_n): # Concatenate the arrays of indices

                  # total_xpoints1 = [nonzerox[lane_inds_n], good_inds_el_n[:,:,1].ravel(), good_inds_er_n[:,:,1].ravel()]
                  # total_ypoints1 = [nonzeroy[lane_inds_n], good_inds_el_n[:,:,0].ravel(), good_inds_er_n[:,:,0].ravel()]
                  # total_xpoints[p_in] = np.concatenate(total_xpoints1)
                  # total_ypoints[p_in] = np.concatenate(total_ypoints1)
                  # # total_xpoints[p_in] = -np.sort(-total_xpoints[p_in])
                  # # total_ypoints[p_in] = np.sort(total_ypoints[p_in])
                  # print total_xpoints[p_in], total_ypoints[p_in]

              # Fit a first order straight line / second order polynomial
              fit_l = np.polyfit(y_[p_in], x_[p_in], 1, full=True)
              fit_p = np.polyfit(y_[p_in], x_[p_in], 2, full=True)

              # Generate x and y values for plotting
              if (np.argmin([fit_l[1], fit_p[1]])==0):
                  self.fitx_[p_in] = fit_l[0][0]*ploty + fit_l[0][1]
              else:
                  self.fitx_[p_in] = fit_p[0][0]*ploty**2 + fit_p[0][1]*ploty + fit_p[0][2]

              # Obtain Matching Score
            #   combined_rect[p_in] = np.array([np.vstack((x_[p_in], y_[p_in])).T]) # For the rectangle region
            #   cv2.fillPoly( template, combined_rect[p_in], 255 )
            #   cv2.fillPoly( template, good_inds_el_n, 255 )  # For the ellipse 1
            #   cv2.fillPoly( template, good_inds_er_n, 255 )  # For the ellipse 2
            #
            # dest_xor = cv2.bitwise_xor(img, template, mask = None)
            # percent_white_pixels_f = float(cv2.countNonZero(dest_xor)/area_img)
            # matching_score = 1 - percent_white_pixels_f

            # print cv2.countNonZero(dest_xor), percent_white_pixels_f, matching_score
            # cv2.imshow('Bitwise XOR', self.mask_example )
            # # De-allocate any associated memory usage
            # if cv2.waitKey(0) & 0xff == 27:
            #    cv2.destroyAllWindows()

        return out_img , self.fitx_, ploty, self.sw_end  #, right_fitx #, right_fit_

    def visualization_polyfit(self, out_img, curves, ploty, modifiedCenters):

       #cv2.circle(out_img, (modifiedCenters[1][1], modifiedCenters[1][0]), 8, (0, 255, 0), -1)
       Lane_i = []

       if len(curves):
         for c_in in range(len(curves)): #

           # Fitted curves as points
           Lane = np.array([np.transpose(np.vstack([curves[c_in], ploty]))])
           Lane_i = Lane[0].astype(int)

           cv2.polylines(out_img, [Lane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)

       return out_img
