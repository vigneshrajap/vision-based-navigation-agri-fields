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
import sliding_window_approach_crop
from geometry_msgs.msg import Pose, PoseArray
import scipy.signal as signal
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
       self.src=np.float32([(0,0.5), (1,0.5), (0,1), (1,1)])
       self.dst=np.float32([(0,0), (1,0), (0,1), (1,1)])

       self.margin_l = 35
       self.margin_r = 35
       self.nwindows = 12

       self.curves = []
       self.lanes = []
       self.ploty = []
       self.kmeans = KMeans()
       self.base_size = 0.1
       self.clusters = 2
       self.modifiedCenters = []

       self.centerLine = []
       #self.midPoints = PoseArray()
       self.output_file = None

    def lane_fit_on_prediction(self, Roi_img, dst_size):

       self.warp_img, M  = sliding_window_approach.perspective_warp(Roi_img, dst_size, self.src, self.dst) # Perspective warp

       newdata =  np.sum(self.warp_img, axis=0) # Sum the columns of warped image to determine peaks

       # plt.plot(newdata) # plotting by columns
       # window = signal.general_gaussian(40, p=0.5, sig=70)
       # filtered = signal.fftconvolve(window, newdata)
       # filtered = (np.average(newdata) / np.average(filtered)) * filtered
       # filtered = np.roll(filtered, -25)
       # plt.plot(filtered) # plotting by columns

       #peakidx = signal.find_peaks_cwt(newdata, np.arange(1,100), noise_perc=0.1)

       peakidx = signal.find_peaks(newdata, height=80000, distance=self.warp_img.shape[1]/3) #, np.arange(1,100), noise_perc=0.1
       #print peakidx, len(peakidx[0]), peakidx[0][0]

       #print peakidx, newdata[peakidx[0]]
       # for p_in in range(len(peakidx[0])):
       #  plt.plot(peakidx[0][p_in], newdata[peakidx[0][p_in]], marker='o', markersize=10)
       # plt.show()
       #plt.savefig('/home/vignesh/hist_col.png')
       #plt.close()

       # InitialPoints Estimation using K-Means clustering
       #self.kmeans, self.modifiedCenters = sliding_window_approach.initialPoints(self.warp_img,self.base_size,self.clusters)

       # Sliding Window Search
       self.polyfit_img, self.curves, self.lanes, self.ploty = sliding_window_approach.sliding_window(self.warp_img, peakidx, self.kmeans, self.nwindows)

       polyfit_img, curves, lanes, ploty = sliding_window_approach_crop.sliding_window(self.warp_img, peakidx, self.kmeans, self.nwindows)

    def visualize_lane_fit(self, dst_size):
       # Visualize the fitted polygonals (One on each lane and on average curve)
       self.polyfit_img, midLane_i = sliding_window_approach_crop.visualization_polyfit(self.polyfit_img, self.curves, self.lanes, self.ploty, self.modifiedCenters)

       # Inverse Perspective warp
       self.invwarp_img, Minv = sliding_window_approach.inv_perspective_warp(self.polyfit_img, (dst_size[1], dst_size[0]), self.dst, self.src)

       # for i in midLane_i:
       #   point_wp = np.array([i[0],i[1],1])
       #   midLane_io = np.matmul(Minv, point_wp) # inverse-M*warp_pt
       #   midLane_n = np.array([midLane_io[0]/midLane_io[2],midLane_io[1]/midLane_io[2]]) # divide by Z point
       #   midLane_n = midLane_n.astype(int)
       #   midPoints.append(midLane_n)
         #midPoints.poses.append(Pose((midLane_n[0],midLane_n[1],0),(0,0,0,1)))

       # Combine the result with the original image
       self.final_img = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)
       #final_img = input_image.copy()

       rheight, rwidth = self.final_img.shape[:2]
       self.final_img[int(rheight*self.crop_ratio):rheight,0:rwidth] = cv2.addWeighted(self.final_img[int(rheight*self.crop_ratio):int(rheight),0:rwidth],
                                                               0.8, self.invwarp_img, 1.0, 0)
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
            cv2.imwrite(self.output_file, self.polyfit_img )

    def lane_fit_on_predicted_image(self, lane_fit = False, display=False): #visualize = None

        if lane_fit:
            self.run_lane_fit()
            self.visualization()
        else:
            self.final_img = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example: Run prediction on an image folder. Example usage: python lane_predict.py --model_prefix=models/resnet_3class --epoch=25 --input_folder=Frogn_Dataset/images_prepped_test --output_folder=.")
    parser.add_argument("--input_folder",default = '', help = "(Relative) path to input image file")
    parser.add_argument("--output_folder", default = '', help = "(Relative) path to output image file. If empty, image is not written.")
    parser.add_argument("--display",default = False, help = "Whether to display video on screen (can be slow)")
    args = parser.parse_args()

    print('Output_folder',args.output_folder)
    im_files = glob.glob(os.path.join(args.input_folder,'*.png'))
    print(os.path.join(args.input_folder+'*.png'))

    lfp = lane_finder_post_predict() #Class object

    for pred_im in im_files:
        if args.output_folder:
            base = os.path.basename(pred_im)
            lfp.output_file = os.path.join(args.output_folder,os.path.splitext(base)[0][7:11])+".jpg" #os.path.splitext(base)[0]
            print(lfp.output_file)
        else:
            output_file = None
        lfp.image = cv2.imread(pred_im, 0)

        #blur_img = cv2.blur(seg_img,(10,10))
        #gausblur_img = cv2.GaussianBlur(seg_img, (5,5),0)
        lfp.image = cv2.medianBlur(lfp.image, 15)
        #bilFilter = cv2.bilateralFilter(img,9,75,75)

        lfp.lane_fit_on_predicted_image(lane_fit = True, display=False) #visualize = "segmentation"

        #t = timeit.Timer("d.lane_fit_on_predicted_image()", "from __main__ import lane_finder_post_predict; d = lane_finder_post_predict()")
        #print t.timeit()

    cv2.destroyAllWindows()
