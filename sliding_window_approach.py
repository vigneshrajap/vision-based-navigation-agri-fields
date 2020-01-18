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

class sliding_window():

    def __init__(self):

        self.fitx_ = []
        self.sw_end = []
        self.increment = 10
        self.margin_sw = 20

        self.whitePixels_thers = 0.50
        self.search_complete = False


        # Decreasing Window Size
        self.win_size = np.arange(1.0, 0.0, -0.1)

    def perspective_warp(self, img, dst_size, src, dst): # Choose the four vertices
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        dst = dst * np.float32(dst_size) # For destination points

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)

        return warped, M

    def sliding_window(self, img, modifiedCenters, kmeans=None, nwindows=10, margin_l=50, margin_r=50, minpix=1, draw_windows=True):

        # Creates a list containing 3 lists, each of [] items, all set to 0
        out_img = np.dstack((img, img, img))*255
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

        if len(modifiedCenters[0]):

            ############################ Initialize Paramters ############################
            # Set height of windows
            window_height = np.int(img.shape[0]/nwindows)

            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            template = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
            self.fitx_ = [[]for y in range(len(modifiedCenters[0]))]
            x_ = [[]for y in range(len(modifiedCenters[0]))]
            y_ = [[]for y in range(len(modifiedCenters[0]))]
            combined = [[]for y in range(len(modifiedCenters[0]))]

            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - np.multiply(range(1,nwindows+1), window_height)
            win_y_high = img.shape[0] - np.multiply(range(0,nwindows), window_height)

            area = float(window_height*self.margin_sw*2)
            area_img = float(img.shape[0]*img.shape[1]*2)
            ############################ Paramters ############################

            for p_in in range(len(modifiedCenters)):

              # Current positions to be updated for each window
              x_current = modifiedCenters[p_in][0]

              # Create empty lists to receive left and right lane pixel indices
              # lane_inds = []
              lane_inds_n = []

              for window in range(nwindows):

                win_x_low = int(x_current - self.win_size[window]*margin_r)
                win_x_high = int(x_current + self.win_size[window]*margin_l)

                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= win_y_low[window]) & (nonzeroy < win_y_high[window]) &
                            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds) > minpix:
                      x_current = np.int(np.mean(nonzerox[good_inds]))

                ############################### TEST ############################
                ## Create a Search Window

                sw_x_low_l = sw_x_high_l = sw_x_low_r = sw_x_high_r = x_current

                k = l = 0
                percent_white_pixels_l = percent_white_pixels_r = 1.0
                self.search_complete = False

                while (self.search_complete == False) and (sw_x_low_r > 0) and (sw_x_high_r <= img.shape[1]) and (sw_x_low_l >= 0) and (sw_x_high_l < img.shape[1]):

                   if (percent_white_pixels_l > self.whitePixels_thers):
                     sw_x_low_l = (x_current - self.increment*k) - self.margin_sw
                     sw_x_high_l = (x_current - self.increment*k) + self.margin_sw
                     if sw_x_low_l<0:
                         sw_x_low_l = 0

                     good_inds1 = ((nonzeroy >= win_y_low[window]) & (nonzeroy < win_y_high[window]) &
                     (nonzerox >= sw_x_low_l) & (nonzerox < sw_x_high_l)).nonzero()[0]

                     #print sw_x_low_l, sw_x_high_l, window, cv2.countNonZero(good_inds1)

                     percent_white_pixels_l = float(cv2.countNonZero(good_inds1)/area)
                     k += 1

                   if (percent_white_pixels_r > self.whitePixels_thers):
                     sw_x_low_r = (x_current + self.increment*l) - self.margin_sw
                     sw_x_high_r = (x_current + self.increment*l) + self.margin_sw

                     if sw_x_high_r>img.shape[1]:
                         sw_x_high_r = img.shape[1]

                     good_inds2 = ((nonzeroy >= win_y_low[window]) & (nonzeroy<win_y_high[window]) &
                     (nonzerox >= sw_x_low_r) & (nonzerox < sw_x_high_r)).nonzero()[0]

                     percent_white_pixels_r = float(cv2.countNonZero(good_inds2)/area)
                     l += 1

                   if (percent_white_pixels_l <= 0.50) and (percent_white_pixels_r <= 0.50):
                       self.search_complete = True

                   # print self.search_complete,  percent_white_pixels_l, percent_white_pixels_r

                    # cv2.rectangle(out_img,(sw_xleft_low,win_y_low),(sw_xleft_high,win_y_high), (0,0,255), 5)
                    # cv2.rectangle(out_img,(sw_x_low,win_y_low),(sw_x_high,win_y_high), (0,0,255), 5)

                # Update the Window Size based on New Margins
                #margin_ll = abs((sw_x_low_l+sw_x_high_l)/2-x_current)
                margin_ll = abs(sw_x_low_l-x_current)
                #margin_rr = abs((sw_x_low_r+sw_x_high_r)/2-x_current)
                margin_rr = abs(sw_x_high_r-x_current)
                win_x_low = x_current - margin_ll
                win_x_high = x_current + margin_rr

                # Identify the nonzero pixels in x and y within the window
                good_inds_n = ((nonzeroy>=win_y_low[window]) & (nonzeroy<win_y_high[window]) &
                               (nonzerox>=win_x_low) & (nonzerox<win_x_high)).nonzero()[0]

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds_n) > minpix:
                      x_current = np.int(np.mean(nonzerox[good_inds_n]))

                # Append these indices to the lists
                if len(good_inds_n):
                     lane_inds_n.append(good_inds_n)
                     if draw_windows == True:
                       cv2.rectangle(out_img,(win_x_low,win_y_low[window]),(win_x_high,win_y_high[window]), (0,255,0), 3)
                       lineThickness = 2
                       cv2.line(out_img, (0, win_y_low[window]), (img.shape[1], win_y_low[window]), (0,0,255), lineThickness)

                self.sw_end.append([p_in, x_current, margin_ll, margin_rr])

              # Concatenate the arrays of indices
              lane_inds_n = np.concatenate(lane_inds_n)

              # Extract left and right line pixel positions
              x_[p_in] = nonzerox[lane_inds_n]
              y_[p_in] = nonzeroy[lane_inds_n]
              combined[p_in] = np.array([np.vstack((x_[p_in], y_[p_in])).T])
              cv2.fillPoly( template, combined[p_in], 255 )

              # Fit a first order straight line / second order polynomial
              fit_l = np.polyfit(y_[p_in], x_[p_in], 1, full=True)
              fit_p = np.polyfit(y_[p_in], x_[p_in], 2, full=True)

              # print fit_l[1], fit_p[1], np.argmin([fit_l[1], fit_p[1]])

              # Generate x and y values for plotting
              if (np.argmin([fit_l[1], fit_p[1]])==0):
                  self.fitx_[p_in] = fit_l[0][0]*ploty + fit_l[0][1]
              else:
                  self.fitx_[p_in] = fit_p[0][0]*ploty**2 + fit_p[0][1]*ploty + fit_p[0][2]

              ############################### TEST ############################

              out_img[nonzeroy[lane_inds_n], nonzerox[lane_inds_n]] = [255, 0, 0] #[255, 0, 100]

            # dest_xor = cv2.bitwise_xor(img, template, mask = None)
            # percent_white_pixels_f = float(cv2.countNonZero(dest_xor)/area_img)
            # matching_score = 1 - percent_white_pixels_f

            # print cv2.countNonZero(dest_xor), percent_white_pixels_f, matching_score
            # cv2.imshow('Bitwise XOR', dest_xor)
            # # De-allocate any associated memory usage
            # if cv2.waitKey(0) & 0xff == 27:
            #    cv2.destroyAllWindows()
            #
            # print cv2.countNonZero(dest_xor), area_img, percent_white_pixels_f, matching_score #str(flatten_list1)

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
