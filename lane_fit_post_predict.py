#!/usr/bin/env python
import os
import glob
from os.path import expanduser
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from sklearn.cluster import KMeans

import sliding_window_approach
import sliding_window_approach_c
from geometry_msgs.msg import Pose, PoseArray
import scipy.signal as signal
import math
import timeit

class lane_finder_post_predict():
    '''
    A class to find fitted center of the lane points along the crop rows given an input RGB image.
    '''
    def __init__(self):
       #### Hyperparameters ####
       self.image = Image()
       self.final_img = Image()
       self.warp_img = Image()
       self.invwarp_img = Image()
       self.polyfit_img = Image()

       self.class_number = 2 # Extract Interesting Class (2 - Lanes in this case) from predictions
       self.crop_ratio = 0.2 # Ratio to crop the background parts in the image from top

       #src=np.float32([(0.1,0), (0.8,0), (0,1), (1,1)])
       #src=np.float32([(0.1,0.5), (0.8,0.5), (0,1), (1,1)])
       #src=np.float32([(0.2,0.5), (0.8,0.5), (0.2,0.8), (0.8,0.8)])
       #src=np.float32([(0,0.4), (1,0.4), (0,0.8), (1,0.8)])
       #self.src=np.float32([(0,0.3), (1,0.3), (0,1), (1,1)])
       self.src=np.float32([(0.1,0.2), (0.9,0.2), (0.1,1), (0.9,1)])

       self.dst=np.float32([(0,0), (1,0), (0,1), (1,1)])

       self.margin_l = 35
       self.margin_r = 35
       self.nwindows = 10

       self.curves = []
       self.ploty = []
       self.kmeans = KMeans()
       self.base_size = 0.1
       self.clusters = 2
       self.modifiedCenters = []
       self.M_t = []
       self.M_tinv = []

       self.centerLine = []
       self.output_file = None
       self.base = None
       self.end_points = []
       self.sw_end = []
       self.weights = np.empty([1, 1])

    # def end_row_detection(self):
    #    top_roi =  self.warp_img[0:self.warp_img.shape[0]/2,0:self.warp_img.shape[1]]
    #    rowdata = np.sum(top_roi, axis=1) # Sum the rows of warped image to determine peaks
    #    num_zeros = (rowdata == 0).sum() # Count the occurence of zero in np array
    #
    #    if num_zeros>top_roi.shape[0]/8:
    #
    #        coldata1 =  np.sum(self.warp_img, axis=0) # Sum the peak cols
    #        end_col_ind = np.argmax(coldata1, axis=0)
    #
    #        #im2, cnt, heirarchy = cv2.findContours(self.warp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #        edges = cv2.Canny(self.warp_img,100,200)
    #        end_row_ind = edges[:,end_col_ind].nonzero()[0]
    #        #indices = np.where(edges != [0])
    #        #coordinates = np.array([indices[0],indices[1]])
    #        #print coordinates[0] #, cv2.countNonZero(coordinates[:][1]==end_col_ind)
    #
    #        self.end_points = [np.argmin(end_row_ind, axis=0), end_col_ind]
    #        # plt.plot(coldata1)
    #        # plt.savefig('/home/vignesh/dummy_folder/e_row'+str(os.path.splitext(self.base)[0][7:11])+'.png')
    #        # plt.close()
    #        # print num_zeros

    def MidPoints_IDW(self):

       if len(self.modifiedCenters[0]):
           # Mid Lane for robot desired trajectory
           self.weights = np.reshape(self.weights, (len(self.modifiedCenters[0]),1), order='F')
           dist_peaks = abs(self.modifiedCenters[0]-self.warp_img.shape[1]/2) # Distance to center of image

           # In IDW, weights are 1 / distance
           if not dist_peaks: # Case where the peak aligned with center of the image
               peak_center = np.where(dist_peaks == 0)
               dist_peaks[peak_center] = 1

           self.weights = 1.0 / dist_peaks

           # Make weights sum to one
           self.weights /= np.sum(self.weights,axis=0)

           # Multiply the weights for each interpolated point by all observed Z-values
           curves_idw = [[]for y in range(len(self.curves))]

           for c_in in range(0, len(self.modifiedCenters[0])):
               curves_idw[c_in] =  (self.weights[c_in]*self.curves[c_in])

           curves_m = np.sum(curves_idw, axis=0)
           midLane = np.array([np.transpose(np.vstack([curves_m, self.ploty]))])
           self.centerLine = midLane.astype(int)
           cv2.polylines(self.polyfit_img, [self.centerLine], 0, (255,0,0), thickness=5, lineType=8, shift=0)

    def lane_fit_on_prediction(self, Roi_img, dst_size):

       self.warp_img, self.M_t  = sliding_window_approach.perspective_warp(Roi_img, dst_size, self.src, self.dst) # Perspective warp

       coldata =  np.sum(self.warp_img, axis=0) # Sum the columns of warped image to determine peaks

       self.modifiedCenters = signal.find_peaks(coldata, height=60000, distance=self.warp_img.shape[1]/3) #, np.arange(1,100), noise_perc=0.1

       # InitialPoints Estimation using K-Means clustering
       #self.kmeans, self.modifiedCenters = sliding_window_approach.initialPoints(self.warp_img,self.base_size,self.clusters)

       # Sliding Window Search
       # polyfit_img, curves, lanes, ploty = sliding_window_approach.sliding_window(self.warp_img, peakidx, self.kmeans, self.nwindows)
       self.polyfit_img, self.curves, self.ploty, self.sw_end  = sliding_window_approach_c.sliding_window(self.warp_img, self.modifiedCenters, self.kmeans, self.nwindows)

    def visualize_lane_fit(self, dst_size):

       # Find the MidPoints using inverse distance weighting and plot the center line
       self.MidPoints_IDW()

       # Visualize the fitted polygonals (One on each lane and on average curve)
       self.polyfit_img = sliding_window_approach_c.visualization_polyfit(self.polyfit_img, self.curves, self.ploty, self.modifiedCenters)

       # Inverse Perspective warp
       self.invwarp_img, self.M_tinv = sliding_window_approach.inv_perspective_warp(self.polyfit_img, (dst_size[1], dst_size[0]), self.dst, self.src)

       if len(self.modifiedCenters[0]):
           points= np.zeros((len(self.modifiedCenters[0]),2))
           for mc_in in range(len(self.modifiedCenters[0])):
               point_wp = np.array([self.modifiedCenters[0][mc_in], self.warp_img.shape[0]-5, 1])
               peakidx_i = np.matmul(self.M_tinv, point_wp) # inverse-M*warp_pt
               peakidx_in = np.array([peakidx_i[0]/peakidx_i[2],peakidx_i[1]/peakidx_i[2]]) # divide by Z point
               peakidx_in = peakidx_in.astype(int)
               points[mc_in] = peakidx_in

               cv2.circle(self.invwarp_img, (peakidx_in[0],peakidx_in[1]), 0, (0,0,255), thickness=25, lineType=8, shift=0)

           ## End Row Check ##
           # self.end_row_detection()

           mask = np.zeros_like(self.image)
           radius = 75
           axes = (radius, radius)
           angle = 0
           startAngle = -180 # Semicircle
           endAngle = 0

           point_wp1 = np.array([self.sw_end[0], self.warp_img.shape[0], 1])
           peakidx_i1 = np.matmul(self.M_tinv, point_wp1) # inverse-M*warp_pt
           peakidx_i1 = peakidx_i1.astype(int)
           #print self.sw_end[0], self.warp_img.shape[0], peakidx_i1

           center = (peakidx_i1[0]+np.int(self.crop_ratio*self.image.shape[0]),peakidx_i1[1]) # divide by Z point ############## CHECK

           #center = (peakidx_i1[0]/peakidx_i1[2],peakidx_i1[1]/peakidx_i1[2]) # divide by Z point
           #center = center.astype(int)

           #col_ind = np.int(self.modifiedCenters[0][0]+ (self.crop_ratio*self.image.shape[0]))
           #center = (self.image.shape[1]/2, col_ind)
           color = (255,255,255)
           thickness = -1
           # create a white filled ellipse
           mask = cv2.ellipse(mask, center, axes, angle, startAngle, endAngle, color, thickness)
           # Bitwise AND operation to black out regions outside the mask
           result = np.bitwise_and(self.image, mask)

           el_area = float((math.pi*math.pow(radius,2))/2)
           percent_white_pixels_el = float(cv2.countNonZero(result)/el_area)
           #print percent_white_pixels_el, center, self.sw_end

           #test = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)

           #cv2.imwrite('/home/vignesh/dummy_folder/test.png',test)

       # Combine the result with the original image
       self.final_img = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)

       rheight, rwidth = self.final_img.shape[:2]
       self.final_img[int(rheight*self.crop_ratio):rheight,0:rwidth] = cv2.addWeighted(self.final_img[int(rheight*self.crop_ratio):int(rheight),0:rwidth],
                                                               0.8, self.invwarp_img, 1.0, 0)

       #x = 10
       #y = 100
       #self.final_img = cv2.rectangle(self.final_img, (x, 0), (y + 10, 0 + 250), (36,255,12), 1)
       #cv2.putText(self.final_img, 'Fedex', (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

       # Draw a rectangle around the text
       self.final_img = cv2.rectangle(self.final_img,(10,10), (270,140), (0,0,255), 4)
       font = cv2.FONT_HERSHEY_SIMPLEX

       cv2.putText(self.final_img,os.path.splitext(self.base)[0][0:13],(20,40), font, 0.6,(255,0,0),2,cv2.LINE_AA)
       cv2.putText(self.final_img, 'Detected Lanes: 1',(20,70), font, 0.6,(255,0,0),2,cv2.LINE_AA)
       cv2.putText(self.final_img, "Window Margins:" + ' ' + str(self.sw_end[1]) + ',' + ' ' + str(self.sw_end[2]),(20,100), font, 0.6,(255,0,0),2,cv2.LINE_AA)
       cv2.putText(self.final_img, 'Central Lane Curvature: 0',(20,130), font, 0.6,(255,0,0),2,cv2.LINE_AA)

       # self.final_img = cv2.ellipse(self.final_img, center, axes, angle, startAngle, endAngle, (0,255,0), 3)

    def run_lane_fit(self):
       # Setting the parameters for upscaling and warping-unwarping
       rheight, rwidth = self.image.shape[:2]
       Roi_img = self.image[int(self.crop_ratio*rheight):rheight,0:rwidth]
       dst_size = Roi_img.shape[:2]
       #print self.image.shape[:2], dst_size
       # Sliding Window Approach on Lanes Class from segmentation Array and fit the poly curves
       self.lane_fit_on_prediction(Roi_img, dst_size)

       # Overlay the inverse warped image on input image
       self.visualize_lane_fit(dst_size)

    def visualization(self, display=False):
        if display:
            cv2.imshow('Prediction', self.final_img)
        if not self.output_file is None:
            cv2.imwrite(self.output_file, self.final_img )

    def lane_fit_on_predicted_image(self, lane_fit = False, display=False): #visualize = None

        if lane_fit:
            self.run_lane_fit()
            self.visualization()
            self.modifiedCenters = [] # reinitialize to zero
        else:
            self.final_img = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example: Run prediction on an image folder. Example usage: python lane_predict.py --model_prefix=models/resnet_3class --epoch=25 --input_folder=Frogn_Dataset/images_prepped_test --output_folder=.")
    parser.add_argument("--input_folder",default = '', help = "(Relative) path to input image file")
    parser.add_argument("--output_folder", default = '', help = "(Relative) path to output image file. If empty, image is not written.")
    parser.add_argument("--display",default = False, help = "Whether to display video on screen (can be slow)")
    args = parser.parse_args()

    print('Output_folder',args.output_folder)
    im_files = sorted(glob.glob(os.path.join(args.input_folder,'*.png')))
    print(os.path.join(args.input_folder+'*.png'))

    lfp = lane_finder_post_predict() #Class object

    for pred_im in im_files:
        if args.output_folder:
            lfp.base = os.path.basename(pred_im)
            lfp.output_file = os.path.join(args.output_folder,os.path.splitext(lfp.base)[0][0:13])+".jpg" #os.path.splitext(base)[0] #+"_osw"
            print(lfp.output_file)
        else:
            output_file = None

        lfp.image = cv2.imread(pred_im, 0)
        lfp.image = cv2.medianBlur(lfp.image, 15)

        lfp.lane_fit_on_predicted_image(lane_fit = True, display=False) #visualize = "segmentation"

        #t = timeit.Timer("d.lane_fit_on_predicted_image()", "from __main__ import lane_finder_post_predict; d = lane_finder_post_predict()")
        #print t.timeit()

    cv2.destroyAllWindows()
