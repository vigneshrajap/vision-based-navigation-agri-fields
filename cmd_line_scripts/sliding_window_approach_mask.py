#!/usr/bin/env python
"""
Created on Sat Jan 18 16:31:13 2020
@author: Vignesh Raja
@description: This node handles the adaptive sliding window functions
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import time

class sliding_window():

    def __init__(self):

        self.img = []
        self.result_left = []
        self.result_right = []
        self.fitx_ = []
        self.sw_end = []
        self.increment = 15
        self.margin_sw = 20
        self.nwindows = 10
        self.win_y_low = [None]*self.nwindows
        self.win_y_high = [None]*self.nwindows
        self.minpix = 1
        self.whitePixels_thers = 0.50
        self.search_complete = False
        self.set_AMR = True

        # Semicircle
        self.Leftangle = 0
        self.Rightangle = 180
        self.LeftstartAngle = -90 # Semicircle
        self.LeftendAngle = -270
        self.RightstartAngle = 90 # Semicircle
        self.RightendAngle = 270
        self.color_ellipse = (0,255,0)
        self.thickness_ellipse = 4
        self.semi_major = [] # Change based on window height
        self.semi_minor = [] # Change based on window height
        self.axes = []

        # Decreasing Window Size
        win_max = 1.0
        win_min = 0.5
        self.win_size = np.arange(win_max, win_min, -((win_max-win_min)/self.nwindows))
        self.margin_l = 115 # (1,2: 100) (3: 140)# Varies on each crop row
        self.margin_r = 115
        self.draw_windows = True

        self.prevx_current = 0
        self.prevcol_ind = 384

        self.mask_example = []
        self.area_img = []
        self.fitting_score_avg = []
        self.rect_sub_ROI = True
        self.curr_xpts = []
        self.curr_ypts = []

    def perspective_warp(self, img, dst_size, src, dst): # Choose the four vertices
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        dst = dst * np.float32(dst_size) # For destination points

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)

        return warped, M

    def fitting_score(self, img, template):
        dest_xor = cv2.bitwise_xor(img, template, mask = None)
        percent_white_pixels_f = float(cv2.countNonZero(dest_xor)/self.area_img)
        fitting_score = 1 - percent_white_pixels_f
        self.fitting_score_avg.append(fitting_score)

        # print fitting_score

    def Adaptive_sliding_window(self, out_img, window, x_current,  nonzeroy, nonzerox, test_img):

        # Horizontal Strip
        hori_strip = self.img[self.win_y_low[window]:self.win_y_high[window],0:out_img.shape[1]]

        # Create a Search Window
        k = l = 0
        percent_white_pixels_el = percent_white_pixels_er = 1.0
        self.search_complete = False

        if self.rect_sub_ROI==True:
            el_area = er_area = float(self.semi_major*self.semi_minor)
            col_ind = np.int(self.win_y_high[window]-self.win_y_low[window])
        else:
            el_area = er_area = float(math.pi*self.semi_major*self.semi_minor/2)
            col_ind = int((self.win_y_high[window]-self.win_y_low[window])/2)

        center_left = center_right = (x_current, col_ind)

        while (self.search_complete == False):

          if (center_left[0]- self.semi_major >= 0):

           if (percent_white_pixels_el > self.whitePixels_thers):

             # create a white filled ellipse
             mask_left = np.zeros_like(hori_strip)

             center_left = (np.int(x_current-self.increment*k), col_ind)

             if self.rect_sub_ROI==True:
                 sw_xleft_low = center_left[0]- self.semi_major
                 sw_xleft_high = center_left[0]
                 mask_left = cv2.rectangle(mask_left, (sw_xleft_low, 0),(sw_xleft_high, col_ind), 255, -1)
             else:
                 mask_left = cv2.ellipse(mask_left, center_left, self.axes, self.Leftangle, self.LeftstartAngle, self.LeftendAngle, 255, -1)

             # Bitwise AND operation to black out regions outside the mask
             self.result_left = np.bitwise_and(hori_strip, mask_left)

             percent_white_pixels_el = float(cv2.countNonZero(self.result_left)/el_area)
             k += 1
             if k==3:  #and window==9:
                 # self.mask_example_r = np.zeros_like(hori_strip)
                # cv2.imshow('Bitwise XOR', hori_strip)
                # # De-allocate any associated memory usage
                # if cv2.waitKey(0) & 0xff == 27:
                #    cv2.destroyAllWindows()
                # self.mask_example_r = np.bitwise_or(self.mask_example_r, mask_left)
                # print sw_xleft_low, sw_xleft_high, col_ind
                # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,255,255), -1)
                # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,0,0), 3)
                test_img = cv2.rectangle(test_img, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,255,255), -1)
                test_img = cv2.rectangle(test_img, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,0,0), 3)

          else:
              percent_white_pixels_el = self.whitePixels_thers

          if (center_right[0]+ self.semi_major <= self.img.shape[1]):

           if (percent_white_pixels_er > self.whitePixels_thers):

             # create a white filled ellipse
             mask_right = np.zeros_like(hori_strip)

             center_right = (np.int(x_current + self.increment*l), col_ind)

             if self.rect_sub_ROI==True:
                 sw_xright_low = center_right[0]
                 sw_xright_high = center_right[0] + self.semi_major
                 mask_right = cv2.rectangle(mask_right, (sw_xright_low,0),(sw_xright_high,col_ind), (255), -1)
             else:
                 mask_right = cv2.ellipse(mask_right, center_right, self.axes, self.Rightangle,  self.RightstartAngle, self.RightendAngle, 255, -1)

             # Bitwise AND operation to black out regions outside the mask
             self.result_right = np.bitwise_and(hori_strip, mask_right)

             percent_white_pixels_er = float(cv2.countNonZero(self.result_right)/er_area)
             l += 1
             if l==3: #and window==9:
                 # self.mask_example_r = np.bitwise_or(self.mask_example_r, mask_right)
                 # # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xright_low,0),(sw_xright_high,col_ind), (0,255,255), 3)
                 # # print sw_xright_low, sw_xright_high, col_ind
                 #
                 # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xright_low,self.win_y_low[window]),(sw_xright_high,self.win_y_high[window]), (255,255,255), -1)
                 # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xright_low,self.win_y_low[window]),(sw_xright_high,self.win_y_high[window]), (0,255,255), 3)

                 test_img = cv2.rectangle(test_img, (sw_xright_low,self.win_y_low[window]),(sw_xright_high,self.win_y_high[window]), (255,255,255), -1)
                 test_img = cv2.rectangle(test_img, (sw_xright_low,self.win_y_low[window]),(sw_xright_high,self.win_y_high[window]), (0,255,255), 3)

          else:
              percent_white_pixels_er = self.whitePixels_thers

          if (percent_white_pixels_el <= self.whitePixels_thers) and (percent_white_pixels_er <= self.whitePixels_thers):

               #if (k>2) and (l>2):
               self.search_complete = True
               # else:
               #      percent_white_pixels_el = 1.0
               #      percent_white_pixels_er = 1.0
            # cv2.rectangle(out_img,(sw_x_low,win_y_low),(sw_x_high,win_y_high), (0,0,255), 5)

        # print window, k, l, self.semi_major, self.semi_minor

        # cv2.imshow('Bitwise XOR', self.mask_example_r)
        # # De-allocate any associated memory usage
        # if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()

        # self.mask_example_r = cv2.ellipse(self.mask_example_r, (center_left[0],(self.win_y_low[window]+self.win_y_high[window])/2), self.axes, self.Leftangle, self.LeftstartAngle, self.LeftendAngle, (255,255,255), -1)
        # self.mask_example_r = cv2.ellipse(self.mask_example_r, (center_right[0],(self.win_y_low[window]+self.win_y_high[window])/2), self.axes, self.Rightangle, self.RightstartAngle, self.RightendAngle, (255,255,255), -1)
        # print k, l
        # test_img = cv2.rectangle(test_img, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,255,255), -1)
        # test_img = cv2.rectangle(test_img, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,0,0), 3)



        # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,255,255), -1)
        # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xright_low,self.win_y_low[window]),(sw_xright_high,self.win_y_high[window]), (255,255,255), -1)
        #
        # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (255,0,0), 3)
        # self.mask_example_r = cv2.rectangle(self.mask_example_r, (sw_xright_low,self.win_y_low[window]),(sw_xright_high,self.win_y_high[window]), (0,255,255), 3)

        # cv2.ellipse(self.mask_example_r, (center_left[0],(self.win_y_low[window]+self.win_y_high[window])/2), self.axes, self.Leftangle, self.LeftstartAngle, self.LeftendAngle, (255,255,255), -1)
        # cv2.ellipse(self.mask_example_r, (center_right[0],(self.win_y_low[window]+self.win_y_high[window])/2), self.axes, self.Rightangle, self.RightstartAngle, self.RightendAngle, (255,255,255), -1)

        # Update the Window Size based on New Margins
        margin_ll = abs(center_left[0]-x_current)
        margin_rr = abs(center_right[0]-x_current)

        win_x_low = x_current - margin_ll
        win_x_high = x_current + margin_rr

        # Identify the nonzero pixels in x and y within the window
        good_inds1 = ((nonzeroy>=self.win_y_low[window]) & (nonzeroy<self.win_y_high[window]) &
                       (nonzerox>=win_x_low) & (nonzerox<win_x_high)).nonzero()[0]

        #mask_empty = np.zeros_like(self.img)
        #mask_empty[self.win_y_low[window]:self.win_y_high[window],0:out_img.shape[1]] = cv2.addWeighted(mask_empty[self.win_y_low[window]:self.win_y_high[window],0:out_img.shape[1]],
                                                                                       # 0.1, self.result_left, 1.0, 0)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero1 = self.result_left.nonzero()
        nonzeroy1 = np.array(nonzero1[0])
        nonzeroy1_n = np.array(nonzero1[0]+self.win_y_low[window])
        nonzerox1 = np.array(nonzero1[1])

        if self.rect_sub_ROI==True:
            good_inds2_n = (((nonzeroy1+self.win_y_low[window])>=self.win_y_low[window]) & ((nonzeroy1+self.win_y_low[window])<self.win_y_high[window])
                            &(nonzerox1>=win_x_low-(self.semi_major)) & (nonzerox1<win_x_low)).nonzero()[0]
            good_inds2 = ((nonzeroy1>=self.win_y_low[window]) & (nonzeroy1<(self.win_y_high[window]))
                            &(nonzerox1>=sw_xleft_low) & (nonzerox1<sw_xleft_high)).nonzero()[0]
        else:
            good_inds2_n = (((nonzeroy1+self.win_y_low[window])>=self.win_y_low[window]) & ((nonzeroy1+self.win_y_low[window])<self.win_y_high[window])
                            &(nonzerox1>=win_x_low-(self.semi_major)) & (nonzerox1<win_x_low)).nonzero()[0]
            good_inds2 = ((nonzeroy1>=0) & (nonzeroy1<col_ind)
                            &(nonzerox1>=win_x_low-(self.semi_major)) & (nonzerox1<win_x_low)).nonzero()[0]

        # mask_empty1 = np.zeros_like(self.img)
        # mask_empty1[self.win_y_low[window]:self.win_y_high[window],0:out_img.shape[1]] = cv2.addWeighted(mask_empty1[self.win_y_low[window]:self.win_y_high[window],0:out_img.shape[1]],
        #                                                                                0.1, self.result_right, 1.0, 0)

        nonzero2 = self.result_right.nonzero()
        nonzeroy2 = np.array(nonzero2[0])
        nonzeroy2_n = np.array(nonzero2[0]+self.win_y_low[window])
        nonzerox2 = np.array(nonzero2[1])
        if self.rect_sub_ROI==True:
            good_inds3_n = (((nonzeroy2+self.win_y_low[window])>=self.win_y_low[window]) & ((nonzeroy2+self.win_y_low[window])<self.win_y_high[window])
                            &(nonzerox2>=win_x_high) & (nonzerox2<win_x_high+(self.semi_major))).nonzero()[0]
            good_inds3 = ((nonzeroy2>=self.win_y_low[window]) & (nonzeroy2<self.win_y_high[window])
                            &(nonzerox2>=sw_xright_low) & (nonzerox2<sw_xright_high)).nonzero()[0]
        else:
            good_inds3_n = (((nonzeroy2+self.win_y_low[window])>=self.win_y_low[window]) & ((nonzeroy2+self.win_y_low[window])<self.win_y_high[window])
                            &(nonzerox2>=win_x_high) & (nonzerox2<win_x_high+(self.semi_major))).nonzero()[0]
            good_inds3 = ((nonzeroy2>=0) & (nonzeroy2<col_ind)
                            &(nonzerox2>=win_x_high) & (nonzerox2<win_x_high+(self.semi_major))).nonzero()[0]

        # good_inds = [good_inds1, good_inds2, good_inds3]
        # good_inds = np.concatenate(good_inds)
        #print good_inds

        total_ypoints = [nonzeroy1_n[good_inds2_n], nonzeroy[good_inds1], nonzeroy2_n[good_inds3_n]]
        total_xpoints = [nonzerox1[good_inds2_n], nonzerox[good_inds1], nonzerox2[good_inds3_n]]

        total_xpoints = np.concatenate(total_xpoints)
        total_ypoints = np.concatenate(total_ypoints)


        # out_img[nonzeroy[good_inds1], nonzerox[good_inds1]] = [255, 0, 0] #[255, 0, 100]
        # # # print nonzeroy2+self.win_y_high[window]
        # # # out_img[nonzeroy1[good_inds2]+self.win_y_low[window], nonzerox1[good_inds2]] = [255, 0, 0] #[255, 0, 100]
        # # # out_img[nonzeroy1[good_inds3]+self.win_y_low[window], nonzerox1[good_inds3]] = [255, 0, 0] #[255, 0, 100]
        # out_img[nonzeroy1_n[good_inds2_n], nonzerox1[good_inds2_n]] = [255, 0, 0] #[255, 0, 100]
        # out_img[nonzeroy2_n[good_inds3_n], nonzerox2[good_inds3_n]] = [255, 0, 0] #[255, 0, 100]

        # print nonzeroy1[good_inds2], self.win_y_high[window], (self.win_y_high[window]-1)-nonzeroy1[good_inds2]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(total_xpoints) > self.minpix:
              x_current = np.int(np.mean(total_xpoints)) #nonzerox[good_inds]

        y_center = (self.win_y_low[window]+self.win_y_high[window])/2

        if self.rect_sub_ROI==True:
            cv2.rectangle(out_img, (sw_xleft_low,self.win_y_low[window]),(sw_xleft_high,self.win_y_high[window]), (0,255,0), 5)
            cv2.rectangle(out_img, (sw_xright_low,self.win_y_low[window]),(sw_xright_high,self.win_y_high[window]), (0,255,0), 5)

            # cv2.arrowedLine(out_img, (x_current, y_center), (center_left[0], y_center), (255, 0, 0), 5, 8, 0, 0.3)
            # cv2.arrowedLine(out_img, (x_current, y_center), (center_right[0], y_center), (255, 0, 0), 5, 8, 0, 0.3)

        else:
            #print center_left[0],center_right[0], y_center
            cv2.ellipse(out_img, (int(center_left[0]),y_center), self.axes, self.Leftangle, self.LeftstartAngle, self.LeftendAngle, self.color_ellipse, self.thickness_ellipse)
            cv2.ellipse(out_img, (int(center_right[0]),y_center), self.axes, self.Rightangle, self.RightstartAngle, self.RightendAngle, self.color_ellipse, self.thickness_ellipse)

            # cv2.rectangle(out_img,(win_x_low, self.win_y_low[window]),(win_x_high, self.win_y_high[window]), (0,255,0), 5)
            # cv2.line(out_img, (center_left[0], self.win_y_low[window]), (center_left[0], self.win_y_high[window]), (0,255,0), 5)
            # cv2.line(out_img, (center_right[0], self.win_y_low[window]), (center_right[0], self.win_y_high[window]), (0,255,0), 5)
            # cv2.arrowedLine(out_img, (x_current, y_center), (center_left[0], y_center), (255, 0, 0), 5, 8, 0, 0.3)
            # cv2.arrowedLine(out_img, (x_current, y_center), (center_right[0], y_center), (255, 0, 0), 5, 8, 0, 0.3)

        return out_img, x_current, (win_x_low, win_x_high), total_ypoints, total_xpoints, test_img

    def sliding_window(self, img, modifiedCenters):

        # Creates a list containing 3 lists, each of [] items, all set to 0
        self.img = img
        # out_img = np.dstack((img, img, img))*255
        out_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        test_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        test_img = cv2.Canny(test_img,100,200)
        im2, contours, heirachy = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(test_img, contours, -1, (255,255,0), 3)

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        self.mask_example = np.zeros_like(self.img)
        self.mask_example_r = cv2.cvtColor(self.mask_example, cv2.COLOR_GRAY2RGB)

        self.prevcol_ind = img.shape[1]

        if len(modifiedCenters[0]):

            ############################ Initialize Paramters ############################
            # Set height of windows
            window_height = np.int(img.shape[0]/self.nwindows)

            self.semi_minor = window_height-15
            self.semi_major = self.semi_minor*2
            self.axes = (self.semi_major, self.semi_minor)

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
            total_pts = []

            # Identify window boundaries in x and y (and right and left)
            self.win_y_low = img.shape[0] - np.multiply(range(1,self.nwindows+1), window_height)
            self.win_y_high = img.shape[0] - np.multiply(range(0,self.nwindows), window_height)

            if self.win_y_low[self.nwindows-1] != 0:
                self.win_y_low[self.nwindows-1] = 0

            #area = float(window_height*self.margin_sw*2)
            self.area_img = float(img.shape[0]*img.shape[1])
            ############################ Parameters ############################

            self.mask_example_r = np.zeros_like(self.img[self.win_y_low[9]:self.win_y_high[9],0:out_img.shape[1]])

            for p_in in range(len(modifiedCenters)):

              # Current positions to be updated for each window
              x_current = modifiedCenters[p_in][0]

              # Create empty lists to receive left and right lane pixel indices
              lane_inds_n = []
              good_inds_el_n = []
              good_inds_er_n = []

              xpts = []
              ypts = []

              for window in range(self.nwindows):

                col_ind = np.int((self.win_y_low[window]+self.win_y_high[window])/2)
                # cv2.circle(out_img, (np.int(x_current), col_ind),0, (0,0,255), thickness=25, lineType=8, shift=0) #+win_y_high[window])/2

                if window > 0:
                    heading = math.atan2(x_current-self.prevx_current, self.prevcol_ind-col_ind)

                    x_current =  int(((x_current-self.prevx_current)*math.cos(heading))-((col_ind-self.prevcol_ind)*math.sin(heading)) + self.prevx_current)

                    # y_current_c =  int(((x_current-self.prevx_current)*math.sin(heading))+((col_ind-self.prevcol_ind)*math.cos(heading)) + col_ind)
                    # print window, x_current, self.prevx_current, col_ind, heading #, x_current_c
                    # cv2.circle(out_img, (x_current_c, col_ind),0, (0,255,255), thickness=25, lineType=8, shift=0) #+win_y_high[window])/2
                    #x_current = x_current_c

                self.prevx_current = x_current
                self.prevcol_ind = col_ind

                win_x_low = int(x_current - self.win_size[window]*self.margin_l)
                win_x_high = int(x_current + self.win_size[window]*self.margin_r)
                # print x_current, win_x_low, win_x_high, self.margin_l, self.margin_r

                # Boundary Conditions
                if win_x_low < 0:
                    win_x_high = int(win_x_high + abs(win_x_low-0))
                    win_x_low = 0

                if win_x_high > self.img.shape[1]:
                    win_x_low = win_x_low - abs(win_x_high-self.img.shape[1])
                    win_x_high = int(self.img.shape[1]-1)

                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= self.win_y_low[window]) & (nonzeroy < self.win_y_high[window]) &
                            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds) > self.minpix:
                      x_current = np.int(np.mean(nonzerox[good_inds]))

                if self.set_AMR == True:
                    # Proposed Adaptive Sliding Window
                    out_img, x_current, win_x, total_ypoints, total_xpoints, test_img = self.Adaptive_sliding_window(out_img, window, x_current, nonzeroy, nonzerox, test_img) #out_img, good_inds, total_ypoints, total_xpoints
                    xpts.append(total_xpoints)
                    ypts.append(total_ypoints)

                    self.curr_xpts.append(x_current)
                    self.curr_ypts.append(col_ind)
                    #
                    # cv2.line(out_img, (int(win_x[0]), self.win_y_low[window]), (int(win_x[1]), self.win_y_low[window]), (0,255,0), self.thickness_ellipse)
                    # cv2.line(out_img, (int(win_x[0]), self.win_y_high[window]), (int(win_x[1]), self.win_y_high[window]), (0,255,0), self.thickness_ellipse)

                    # print win_x[0]

                else:
                    if len(good_inds): # Append these indices to the lists
                         lane_inds_n.append(good_inds)
                         # out_img[nonzeroy[good_inds], nonzerox[good_inds]] = [255, 0, 0] #[255, 0, 100]

                    # Plotting
                    cv2.rectangle(out_img,(win_x_low,self.win_y_low[window]),(win_x_high,self.win_y_high[window]), (0,255,0), self.thickness_ellipse)

                if self.draw_windows == True:
                    # Plotting the X center of the windows
                    # cv2.circle(out_img, (int(x_current), int(col_ind)), 0, (0,0,255), thickness=25, lineType=8, shift=0)
                    # cv2.circle(self.mask_example_r, (int(x_current), int(col_ind)), 0, (0,0,255), thickness=25, lineType=8, shift=0)
                    cv2.circle(test_img, (int(x_current), int(col_ind)), 0, (0,0,255), thickness=25, lineType=8, shift=0)

                    # Plotting the horizontal strips
                    cv2.line(out_img, (0, self.win_y_low[window]), (img.shape[1], self.win_y_low[window]), (0,0,255), 2)

                    cv2.line(test_img, (0, self.win_y_low[window]), (img.shape[1], self.win_y_low[window]), (0,0,255), 2)

                    # cv2.line(self.mask_example_r, (0, self.win_y_low[window]), (img.shape[1], self.win_y_low[window]), (0,0,255), 2)
                    # for ind in range(len(self.win_y_low)):
                    #      hori_strip = img[self.win_y_low[ind]:self.win_y_high[ind],0:img.shape[1]]
                    #      cv2.imwrite("/home/vignesh/dummy_folder/"+str(ind)+".png", hori_strip)

                # self.sw_end.append([p_in, x_current, margin_ll, margin_rr])

              if self.set_AMR == True:
                  x_[p_in] = np.concatenate(xpts)
                  y_[p_in] = np.concatenate(ypts)


              else:
                  if len(lane_inds_n): # Concatenate the arrays of indices
                    lane_inds_n = np.concatenate(lane_inds_n)

                  # Extract left and right line pixel positions
                  x_[p_in] = nonzerox[lane_inds_n]
                  y_[p_in] = nonzeroy[lane_inds_n]
                  out_img[nonzeroy[lane_inds_n], nonzerox[lane_inds_n]] = [255, 0, 0]


              # Fit a first order straight line / second order polynomial
              fit_l = np.polyfit(y_[p_in], x_[p_in], 1, full=True)
              fit_p = np.polyfit(y_[p_in], x_[p_in], 2, full=True)

              # Generate x and y values for plotting
              if (np.argmin([fit_l[1], fit_p[1]])==0):
                  self.fitx_[p_in] = fit_l[0][0]*ploty + fit_l[0][1]
              else:
                  self.fitx_[p_in] = fit_p[0][0]*ploty**2 + fit_p[0][1]*ploty + fit_p[0][2]

              #Lane = np.array([np.transpose(np.vstack([self.curr_xpts]))])
              Lane = np.array([np.transpose(np.vstack([self.curr_xpts, self.curr_ypts]))])

              Lane_i = Lane[0].astype(int)

              #print  Lane_i
              total_pts.append(Lane_i)
              self.curr_xpts = []
              self.curr_ypts = []

              # Obtain Matching Score
              combined_rect[p_in] = np.array([np.vstack((x_[p_in], y_[p_in])).T]) # For the rectangle region
              cv2.fillPoly( template, combined_rect[p_in], 255 )
              #cv2.fillPoly( template, good_inds_el_n, 255 )  # For the ellipse 1
              #cv2.fillPoly( template, good_inds_er_n, 255 )  # For the ellipse 2

            self.fitting_score(img, template)



            # self.mask_example_r = cv2.addWeighted( test_img, 0.6,self.mask_example_r, 0.9, 0)
            # out_img_test = cv2.bitwise_and( test_img, out_img)

            # self.mask_example_r = cv2.cvtColor(self.mask_example_r, cv2.COLOR_GRAY2RGB)
            # self.mask_example_r = cv2.rectangle(self.mask_example_r, (156, 0),(192, 39), (255,0,0), 3)
            # self.mask_example_r = cv2.rectangle(self.mask_example_r, (252, 0),(288, 39), (0,255,255), 3)
            # self.mask_example_r = cv2.rectangle(self.mask_example_r, (317, 0),(352, 39), (255,0,0), 3)
            # self.mask_example_r = cv2.rectangle(self.mask_example_r, (413, 0),(449, 39), (0,255,255), 3)

            cv2.imwrite("test.png", test_img )

            # cv2.imshow('Bitwise XOR', self.mask_example_r)
            # # cv2.imshow('Bitwise XOR', out_img)
            # # De-allocate any associated memory usage
            # if cv2.waitKey(0) & 0xff == 27:
            #     cv2.destroyAllWindows()

            # cv2.polylines(out_img, [Lane_i[0,:],Lane_i[1,:]], 0, (0,255,255), thickness=5, lineType=8, shift=0)
            # cv2.polylines(out_img, [self.curr_pts], 0, (0,255,255), thickness=5, lineType=8, shift=0)

            # cv2.imshow('Bitwise XOR', self.mask_example_r)
            # # De-allocate any associated memory usage
            # if cv2.waitKey(0) & 0xff == 27:
            #    cv2.destroyAllWindows()

        return out_img, total_pts, self.fitting_score_avg  #,self.fitx_,self.sw_end, self.fitting_score_avg, ploty, right_fitx #, right_fit_

    def visualization_polyfit(self, out_img, curves, ploty, modifiedCenters, current_Pts):

       #cv2.circle(out_img, (modifiedCenters[1][1], modifiedCenters[1][0]), 8, (0, 255, 0), -1)
       # Lane_i = []
       if len(current_Pts)>1:
        for c_in in range(len(current_Pts)): #
            cv2.polylines(out_img, [current_Pts[c_in]], 0, (0,255,255), thickness=5, lineType=8, shift=0)
            #cv2.circle(out_img, (current_Pts[c_in][0],current_Pts[c_in][1]), 0, (0,0,255), thickness=25, lineType=8, shift=0)

        curves_m = (current_Pts[0]+current_Pts[1])/2
        #midLane = np.array([np.transpose(np.vstack([curves_m, ploty]))])
        midLane_i = curves_m.astype(int)
        cv2.polylines(out_img, [midLane_i], 0, (255,0,255), thickness=5, lineType=8, shift=0)

       # if len(curves):
       #   for c_in in range(len(curves)): #

       #     # Fitted curves as points
       #     Lane = np.array([np.transpose(np.vstack([curves[c_in], ploty]))])
       #     Lane_i = Lane[0].astype(int)
       #     print Lane_i
       #     cv2.polylines(out_img, [Lane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)
       #
       #   curves_m = (curves[0]+curves[1])/2
       #   midLane = np.array([np.transpose(np.vstack([curves_m, ploty]))])
       #   midLane_i = midLane[0].astype(int)
       #   cv2.polylines(out_img, [midLane_i], 0, (255,0,255), thickness=5, lineType=8, shift=0)

       return out_img, midLane_i
