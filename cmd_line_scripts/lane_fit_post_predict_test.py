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
# from geometry_msgs.msg import Pose, PoseArray
import scipy.signal as signal
import math

import imutils
import time

start_time = time.time()

DBASW = sliding_window_approach.sliding_window()

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
       self.roi_img = Image()
       self.crop_img = Image()

       self.class_number = 2 # Extract Interesting Class (2 - Lanes in this case) from predictions
       self.crop_ratio = 0.375 # Ratio to crop the background parts in the image from top
       self.warp_ratio = 0.7

       self.src = np.float32([(0,-0.3), (1,-0.3), (-0.4,0.8), (1.4,0.8)])
       # self.src = np.float32([(0,0), (1,0), (-0.2, 0.9), (1.2, 0.9)])
       # self.src = np.float32([(0,0), (1,0), (0,1), (1,1)])
       self.dst = np.float32([(0,0), (1,0), (0,1), (1,1)])

       # self.margin_l = 35
       # self.margin_r = 35

       self.curves = []
       self.ploty = []
       self.modifiedCenters_local = []
       self.modifiedCenters = []
       self.M_t = []
       self.M_tinv = []

       self.centerLine = []
       self.output_file = None
       self.base = None
       self.base_name = None
       self.end_points = []
       self.sw_end = []
       self.weights = np.empty([2, 1])
       self.fitting_score_avg = []
       self.current_Pts = []
       self.total_time = 0
       self.matching_score = []
       self.fit_s_count = 0
       self.fit_p_count = 0
       self.modifiedCenters_global = []

    def peaks_estimation(self):
       coldata = np.sum(self.warp_img, axis=0)/255 # Sum the columns of warped image to determine peaks

       self.modifiedCenters_local = signal.find_peaks(coldata, height=220, distance=self.warp_img.shape[1]/6) #, np.arange(1,100), noise_perc=0.1, distance=self.warp_img.shape[1]/2 , distance=self.warp_img.shape[1]/2

       self.modifiedCenters_global = self.modifiedCenters_local[0]

       # print self.modifiedCenters_local

       # For crop based
       # self.modifiedCenters_global = [self.modifiedCenters_local[0][np.argmin(abs(self.modifiedCenters_local[0]-int(self.warp_img.shape[0]/2)))]]

       # For lane based
       # if (len(self.modifiedCenters_global)>2):
       #     self.modifiedCenters_global = np.delete(self.modifiedCenters_global, np.argwhere(np.max(self.modifiedCenters_global-self.warp_img.shape[0])))

       # print self.modifiedCenters_global, np.argmax(self.modifiedCenters_global), np.delete(self.modifiedCenters_global,np.argmax(self.modifiedCenters_global-int(self.warp_img.shape[0]/2)))

       ## For crop based
       # if ((len(self.modifiedCenters_global)%2)==0):
       #     self.modifiedCenters_global = np.delete(self.modifiedCenters_global,np.argmax(self.modifiedCenters_global-int(self.warp_img.shape[0]/2)))
       #     # self.modifiedCenters_global = np.delete(self.modifiedCenters_global, np.argwhere(np.max(self.modifiedCenters_global-int(self.warp_img.shape[0]/2))))

       # print self.modifiedCenters_global

       # print self.modifiedCenters_global,

       # if (len(self.modifiedCenters_local[0])<1):
       #     new_warp_ratio = self.warp_ratioself.modifiedCenters_local[0]
       #     ######################## CROPS ###############################
       #     while ((len(self.modifiedCenters_local[0])<1)):
       #        new_warp_ratio = new_warp_ratio-0.1
       #        rheight, rwidth = self.image.shape[:2]
       #        self.crop_img = self.image[int(new_warp_ratio*rheight):rheight,0:rwidth]
       #        dst_size = self.crop_img.shape[:2]
       #
       #        self.warp_img, self.M_t  = DBASW.perspective_warp(self.crop_img, (dst_size[1], dst_size[1]), self.src, self.dst) # Perspective warp
       #        coldata = np.sum(self.warp_img, axis=0)/255 # Sum the columns of warped image to determine peaks
       #
       #        self.modifiedCenters_local = signal.find_peaks(coldata, height=320) #, np.arange(1,100), noise_perc=0.1 #, distance=self.warp_img.shape[1]/3
              # print self.modifiedCenters_local, new_warp_ratio
              # if (new_warp_ratio<0.5):
              #     continue

       # Visualize the Peaks Estimation
       # warp_img_c = cv2.cvtColor(self.warp_img, cv2.COLOR_GRAY2RGB)
       # for p_in in range(len(self.modifiedCenters_local[0])):
       #      cv2.circle(warp_img_c, (np.int(self.modifiedCenters_local[0][p_in]), 630), 8, (0, 0, 255), 20)
       #
       # cv2.imwrite("/home/vignesh/Third_Paper/warp_img.png", warp_img_c)

       # plt.plot(self.modifiedCenters_local[0][0], coldata[self.modifiedCenters_local[0][0]], 'ro', markersize=10)
       # plt.legend(['Peaks'])
       # plt.plot(coldata)
       # plt.xlabel("Columns (Pixels)")
       # plt.ylabel("No. Of whitePixels (Pixels)")
       # plt.title("Peak Estimation - Summing of Columns")
       # plt.show()

    def lane_fit_on_prediction(self, dst_size):

       rheight, rwidth = self.image.shape[:2]
       self.crop_img = self.image[int(self.warp_ratio*rheight):rheight,0:rwidth]
       dst_size = self.crop_img.shape[:2]

       self.warp_img, self.M_t  = DBASW.perspective_warp(self.crop_img, (dst_size[1], dst_size[1]), self.src, self.dst) # Perspective warp

       self.peaks_estimation()

       # Inverse Perspective warp
       self.invwarp_img, self.M_tinv = DBASW.perspective_warp(self.warp_img, (dst_size[1], dst_size[1]), self.dst, self.src) #self.polyfit_img

       #self.M_tinv = cv2.getPerspectiveTransform(self.dst, self.src)
       #self.roi_img = cv2.cvtColor(self.roi_img, cv2.COLOR_GRAY2RGB)

       if len(self.modifiedCenters_global):
           self.modifiedCenters = np.zeros((len(self.modifiedCenters_global),2))
           for mc_in in range(len(self.modifiedCenters_global)):
               point_wp = np.array([self.modifiedCenters_global[mc_in], self.image.shape[1], 1])
               peakidx_i = np.matmul(self.M_tinv, point_wp) # inverse-M*warp_pt
               peakidx_in = np.array([peakidx_i[0]/peakidx_i[2],peakidx_i[1]/peakidx_i[2]]) # divide by Z point
               peakidx_in = peakidx_in.astype(int)
               self.modifiedCenters[mc_in] = peakidx_in


    def visualize_lane_fit(self, dst_size):

       t_start = time.time()

       self.roi_img, self.current_Pts, self.fitting_score_avg = DBASW.sliding_window(self.roi_img, self.modifiedCenters)
       t_end = time.time()
       # print 'Prediction time: ', t_end-t_start
       self.total_time = self.total_time + (t_end-t_start)

       # Visualize the fitted polygonals (One on each lane and on average curve)
       self.roi_img, self.centerLine = DBASW.visualization_polyfit(self.roi_img, self.curves, self.ploty, self.modifiedCenters, self.current_Pts)

       la = [x for x,y in self.centerLine]
       lb = [y for x,y in self.centerLine]
       plotyc = np.linspace(0, self.roi_img.shape[0]-1, self.roi_img.shape[0])

       # Fit a first order straight line / second order polynomial
       fit_l = np.polyfit(lb, la, 1, full=True)
       fit_p = np.polyfit(lb, la, 2, full=True)


         # /*
         #     curvature of ax*x + b*x + c, given x
         # */
         # curvaturequad(a, b, c, x)
         # {
         #    dybydx = 2*a*x + b;  // first derivative
         #    d2ybydx2 = 2 * a;    // second derivative
         #
         #    dybydx2 = dybydx*dybybx;
         #    numerator = sqrt(dybydx2 + 1)*(dybydx +1);
         #    return numerator / d2ybydx2;
         # }

       print ((1 + (2*fit_p[0][0]*x + fit_p[0][1])**2)**1.5) / np.absolute(2*fit_p[0][0])

       # Generate x and y values for plotting
       if (np.argmin([fit_l[1], fit_p[1]])==0):
         fitxc_ = fit_l[0][0]*plotyc + fit_l[0][1]
         # print self.fit_s_count
         self.fit_s_count = self.fit_s_count + 1
       else:
         fitxc_ = fit_p[0][0]*plotyc**2 + fit_p[0][1]*plotyc + fit_p[0][2]
         self.fit_p_count = self.fit_p_count + 1

       # print self.centerLine, la, lb
       # fit_c = np.polyfit(lb, la, 2, full=True)
       # fitxc_ = fit_c[0][0]*plotyc**2 + fit_c[0][1]*plotyc + fit_c[0][2]
       pts_left = np.array([np.transpose(np.vstack([fitxc_, plotyc]))])
       pts_left = pts_left[0].astype(int)
       # print len(pts_left)
       cv2.polylines(self.roi_img, [pts_left], 0, (255,255,0), thickness=5, lineType=8, shift=0)

       ################# Discussions #############################
       # dummy = self.centerLine
       # dummy[7][0] = dummy[6][0]
       # dummy[8][0] = dummy[7][0]
       # dummy[9][0] = dummy[8][0]
       # la_d = [x for x,y in dummy]
       # lb_d = [y for x,y in dummy]
       # fit_p_d = np.polyfit(lb_d, la_d, 2, full=True)
       # fitxc_d = fit_p_d[0][0]*plotyc**2 + fit_p_d[0][1]*plotyc + fit_p_d[0][2]
       # pts_left_d = np.array([np.transpose(np.vstack([fitxc_d, plotyc]))])
       # pts_left_d = pts_left_d[0].astype(int)
       # # print len(pts_left)
       # cv2.polylines(self.roi_img, [pts_left_d], 0, (255,0,255), thickness=5, lineType=8, shift=0)
       ################# Discussions #############################

       # cv2.fillPoly(out_img, np.int_(pts_left), (255,0,255))
       #
       # import csv
       # gt_row_x = []
       # gt_row_y = []
       # set_folder = str('/home/vignesh/dummy_folder/test_cases/') #larger_plants
       #
       #
       # rheight1, rwidth1 = self.image.shape[:2]
       #
       # with open(set_folder+'/ground_truth/'+self.base_name+'.csv') as csvfile: #inclined_terrains #larger_plants
       # # with open('/home/vignesh/dummy_folder/test_cases/results/'+base_name+'.csv') as csvfile:
       #       spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
       #       for row in spamreader:
       #           gt_row_x.append(row[1])
       #           gt_row_y.append(row[2])
       #
       #       gt_x = []
       #       gt_y = []
       #       for gt_in in range(len(gt_row_x)):
       #           gt_x.append(int(gt_row_x[gt_in]))
       #           gt_y.append(float(gt_row_y[gt_in])-rheight1*self.crop_ratio)
       #           # cv2.circle(self.roi_img, (int(gt_row_x[gt_in]),int(float(gt_row_y[gt_in])-rheight1*self.crop_ratio)), 0, (255,0,255), thickness=15, lineType=8, shift=0)
       #       # print gt_x, float(gt_row_y[0])-108
       #
       #       fit_c1 = np.polyfit(gt_y, gt_x, 2, full=True)
       #       plotyc1 = np.linspace(0, self.roi_img.shape[0]-1, self.roi_img.shape[0])
       #       fitxc_1 = fit_c1[0][0]*plotyc1**2 + fit_c1[0][1]*plotyc1 + fit_c1[0][2]
       #       pts_left1 = np.array([np.transpose(np.vstack([fitxc_1, plotyc1]))])
       #       pts_left1 = pts_left1[0].astype(int)
       #
       #       # cv2.polylines(self.roi_img, [pts_left1], 0, (255,0,0), thickness=5, lineType=8, shift=0)
       #
       # ms_temp = 0
       # # for mid_in in range(len(self.centerLine)):
       # #     #print np.float(np.float(1)/np.float(mid_in+1))
       # #     ms = self.centerLine[mid_in][0]-int(gt_row_x[mid_in])
       # #     # print math.pow((np.float(ms)/np.float(0.25*self.image.shape[1])),2)
       # #     ms_lite = abs(np.float(ms)/np.float(140-(5*mid_in))) #0.25*self.image.shape[1]
       # #     ms_temp = ms_temp + ms_lite
       # #     # print gt_row[mid_in], self.centerLine[mid_in][0], ms_lite
       # # print pts_left, pts_left1
       # scale = 1
       # nwindows = 10
       # crop_row_spacing = 140
       # strip_height = np.int(self.roi_img.shape[0]/nwindows)
       #
       # for mid_in in range(len(pts_left1)):
       #     #print pts_left[mid_in][0]-pts_left1[mid_in][0]
       #     ms = abs(pts_left[mid_in][0]-pts_left1[mid_in][0])
       #
       #     ms_lite = abs(np.float(ms)/(crop_row_spacing*scale)) #/np.float(self.image.shape[1]-100)) #-(5*mid_in) #0.25*self.image.shape[1]-(scale)
       #
       #     ms_temp = ms_temp + ms_lite
       #     if ((mid_in%strip_height)==0):
       #         scale = scale - 0.075
       #
       # ms_norm = ms_temp/len(pts_left1)
       # self.matching_score.append(1 - math.pow(ms_norm,2))   #1 - (math.pow(ms_norm,2)/crop_row_spacing)) #len(pts_left1)
       #
       # # self.matching_score.append(1 - math.pow((ms_norm/crop_row_spacing),2))   #1 - (math.pow(ms_norm,2)/crop_row_spacing)) #len(pts_left1)
       #
       # print 1 - math.pow(ms_norm,2)


       # Find the MidPoints using inverse distance weighting and plot the center line
       # self.MidPoints_IDW()

       # cv2.imwrite('/home/vignesh/dummy_folder/test.png',self.roi_img)

       # Combine the result with the original image
       self.final_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
       # self.final_img = cv2.imread("/home/vignesh/Third_Paper/Datasets/20191010_L1_N/"+os.path.splitext(self.base)[0][0:18]+".png")
       # self.final_img = cv2.imread("/home/vignesh/Third_Paper/Datasets/20191010_L4_N/"+os.path.splitext(self.base)[0][0:18]+".png")
       # self.final_img = cv2.imread("/home/vignesh/dummy_folder/test_cases/discussions/"+os.path.splitext(self.base)[0][0:18]+".png")
       # self.final_img = cv2.imread(set_folder+"/rgb/"+self.base_name+".png")#inclined_terrains #larger_plants
       rheight, rwidth = self.final_img.shape[:2]

       self.final_img[int(rheight*self.crop_ratio):rheight,0:rwidth] = cv2.addWeighted(self.roi_img, 0.5,self.final_img[int(rheight*self.crop_ratio):int(rheight),0:rwidth], 1, 0)

       if len(self.modifiedCenters[0]):
           for mc_in in range(len(self.modifiedCenters_global)):
               #print self.modifiedCenters[mc_in][0], self.modifiedCenters[mc_in][1], self.final_img.shape[0]-100

               cv2.circle(self.final_img, (int(self.modifiedCenters[mc_in][0]),int(self.final_img.shape[0]-20)), #+rheight*self.warp_ratio
                                                                                                0, (255,0,255), thickness=15, lineType=8, shift=0)


       #x = 10
       #y = 100
       #self.final_img = cv2.rectangle(self.final_img, (x, 0), (y + 10, 0 + 250), (36,255,12), 1)
       #cv2.putText(self.final_img, 'Fedex', (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

       # # Draw a rectangle around the text
       # self.final_img = cv2.rectangle(self.final_img,(10,10), (270,140), (0,0,255), 4)
       # font = cv2.FONT_HERSHEY_SIMPLEX
       #
       # cv2.putText(self.final_img,os.path.splitext(self.base)[0][0:13],(20,40), font, 0.6,(255,0,0),2,cv2.LINE_AA)
       # cv2.putText(self.final_img, 'Detected Lanes: 1',(20,70), font, 0.6,(255,0,0),2,cv2.LINE_AA)
       # cv2.putText(self.final_img, "Window Margins:" + ' ' + str(self.sw_end[1]) + ',' + ' ' + str(self.sw_end[2]),(20,100), font, 0.6,(255,0,0),2,cv2.LINE_AA)
       # cv2.putText(self.final_img, 'Central Lane Curvature: 0',(20,130), font, 0.6,(255,0,0),2,cv2.LINE_AA)

       # self.final_img = cv2.ellipse(self.final_img, center, axes, angle, startAngle, endAngle, (0,255,0), 3)

    def run_lane_fit(self):
       # Setting the parameters for upscaling and warping-unwarping
       rheight, rwidth = self.image.shape[:2]
       self.roi_img = self.image[int(self.crop_ratio*rheight):rheight,0:rwidth]
       dst_size = self.roi_img.shape[:2]
       # print int(self.crop_ratio*rheight)
       # cv2.imwrite("/home/vignesh/roi_img.png", self.roi_img )

       # Sliding Window Approach on Lanes Class from segmentation Array and fit the poly curves
       self.lane_fit_on_prediction(dst_size)

           # Overlay the inverse warped image on input image
       self.visualize_lane_fit(dst_size)

       # print self.centerLine[4], self.roi_img.shape[1]/2, self.roi_img.shape[0]/2

    def visualization(self, display=False):
        if display:
            cv2.imshow('Prediction', self.final_img)
        if not self.output_file is None:
            cv2.imwrite(self.output_file, self.roi_img)

    def lane_fit_on_predicted_image(self, lane_fit = False, display=False): #visualize = None

        if lane_fit:

            self.run_lane_fit()

            self.visualization()
            self.modifiedCenters = []
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
            lfp.base_name = os.path.splitext(lfp.base)[0][0:18]
            # with open('/home/vignesh/dummy_folder/test_cases/inclined_terrains/ground_truth/'+os.path.splitext(self.base)[0][0:11]+'.csv') as csvfile:
            if (os.path.splitext(lfp.base)[0][11] != str('_')):
               lfp.base_name = os.path.splitext(lfp.base)[0][0:19]

            lfp.output_file = os.path.join(args.output_folder,lfp.base_name+".jpg") #os.path.splitext(base)[0] #+"_osw"
            print(lfp.output_file)
        else:
            output_file = None

        lfp.image = cv2.imread(pred_im, 0)
        #lfp.image = cv2.resize(lfp.image,None,fx=0.75,fy=0.75)
        # print lfp.image.shape[:2]
        lfp.image = cv2.medianBlur(lfp.image, 15)

        lfp.lane_fit_on_predicted_image(lane_fit = True, display=False) #visualize = "segmentation"

        # t = timeit.Timer("d.lane_fit_on_predicted_image()", "from __main__ import lane_finder_post_predict; d = lane_finder_post_predict()")
        # print t.timeit()
    # print np.sum(lfp.matching_score)/len(lfp.matching_score)
    # print np.mean(lfp.matching_score), np.std(lfp.matching_score)
    # print np.sum(lfp.fitting_score_avg)/len(lfp.fitting_score_avg)
    # print("--- %s seconds ---" % (lfp.total_time))
    # print lfp.fit_s_count, lfp.fit_p_count, float(lfp.fit_s_count)/float(lfp.fit_s_count+lfp.fit_p_count), float(lfp.fit_p_count)/float(lfp.fit_s_count+lfp.fit_p_count)
    cv2.destroyAllWindows()
