#!/usr/bin/env python
#import keras_segmentation
#from keras_segmentation.predict import model_from_checkpoint_path
import os
import glob
from os.path import expanduser
import argparse
#from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sliding_window_approach
from geometry_msgs.msg import Pose, PoseArray

def upscaling_warping_parameters(pred_img, class_number, crop_ratio):
   # Perspective warp
   rheight, rwidth = pred_img.shape[:2]
   Roi_g = pred_img[int(crop_ratio*rheight):rheight,0:rwidth]
   dst_size = Roi_g.shape[:2]
   # src=np.float32([(0.1,0), (0.8,0), (0,1), (1,1)])
   src=np.float32([(0,0.3), (1,0.3), (0,1), (1,1)])
   dst=np.float32([(0,0), (1,0), (0,1), (1,1)])

   return Roi_g, src, dst, dst_size

def lane_fit_on_prediction(Roi_img, src, dst, dst_size):

   warped_img, M  = sliding_window_approach.perspective_warp(Roi_img, dst_size, src, dst)

   margin=35
   nwindows=12

   # InitialPoints Estimation using K-Means clustering
   margin, margin_1, modifiedCenters = sliding_window_approach.initialPoints(warped_img, margin)

   # Sliding Window Search
   out_img, curves, lanes, ploty = sliding_window_approach.sliding_window(warped_img, modifiedCenters, nwindows, margin, margin_1)
   return warped_img, out_img, curves, lanes, ploty, modifiedCenters

def visualize_lane_fit(input_image, out_img, curves, lanes, ploty, modifiedCenters, src, dst, dst_size, crop_ratio):
   # Visualize the fitted polygonals (One on each lane and on average curve)
   out_img, midLane_i = sliding_window_approach.visualization_polyfit(out_img, curves, lanes, ploty, modifiedCenters)

   # Inverse Perspective warp
   invwarp, Minv = sliding_window_approach.inv_perspective_warp(out_img, (dst_size[1], dst_size[0]), dst, src)

   midPoints = []
   #midPoints = PoseArray()
   for i in midLane_i:
     point_wp = np.array([i[0],i[1],1])
     midLane_io = np.matmul(Minv, point_wp) # inverse-M*warp_pt
     midLane_n = np.array([midLane_io[0]/midLane_io[2],midLane_io[1]/midLane_io[2]]) # divide by Z point
     midLane_n = midLane_n.astype(int)
     midPoints.append(midLane_n)
     #midPoints.poses.append(Pose((midLane_n[0],midLane_n[1],0),(0,0,0,1)))

   # Combine the result with the original image
   final_img = cv2.cvtColor(input_image,cv2.COLOR_GRAY2RGB)
   #final_img = input_image.copy()
   rheight, rwidth = final_img.shape[:2]
   final_img[int(rheight*crop_ratio):rheight,0:rwidth] = cv2.addWeighted(final_img[int(rheight*crop_ratio):int(rheight),0:rwidth],
                                                           0.8, invwarp, 1.0, 0)
   return out_img, midPoints, invwarp, final_img

def run_lane_fit(pred_inp_image, class_number = 2,  crop_ratio = 0.2):
# Extract Interesting Class (2 - Lanes in this case) from predictions
# Ratio to crop the background parts in the image from top

   # Setting the parameters for upscaling and warping-unwarping
   Roi_img, src, dst, dst_size = upscaling_warping_parameters(pred_inp_image, class_number, crop_ratio)

   # Sliding Window Approach on Lanes Class from segmentation Array and fit the poly curves
   warp_img, out_img, curves, lanes, ploty, modifiedCenters = lane_fit_on_prediction(Roi_img, src, dst, dst_size)

   # Overlay the inverse warped image on input image
   polyfit_img, centerLine, invwarp, final_img = visualize_lane_fit(pred_inp_image, out_img, curves, lanes, ploty, modifiedCenters, src, dst, dst_size, crop_ratio)
   return invwarp ,final_img, centerLine

def visualization(vis_img, lane_fit = None, evaluation = None, n_classes=None, visualize = None, display=False, output_file=None):

    if display:
        cv2.imshow('Prediction', vis_img)
    if not output_file is None:
        cv2.imwrite(output_file, vis_img )
    return vis_img

def lane_fit_on_predicted_image(inp = None, lane_fit = False, output_file = None, display=False): #visualize = None

    if lane_fit:
        class_number = 2 # Extract Interesting Class (2 - Lanes in this case) from predictions
        crop_ratio = 0.2 # Ratio to crop the background parts in the image from top
        warp_img, final_img, fit = run_lane_fit(inp, class_number, crop_ratio)
        #cv2.imwrite(output_file, warp_img )

        #visualize: None, "all" or one of, "segmentation", "lane_fit"
        vis_img = visualization(final_img, lane_fit = None, evaluation = None, n_classes=3, visualize = "segmentation", display=False, output_file=output_file)

    else:
        fit = None
        out_img = None
        vis_img = None

    #return vis_img, out_img, fit

def main():
    parser = argparse.ArgumentParser(description="Example: Run prediction on an image folder. Example usage: python lane_predict.py --model_prefix=models/resnet_3class --epoch=25 --input_folder=Frogn_Dataset/images_prepped_test --output_folder=.")
    parser.add_argument("--input_folder",default = '', help = "(Relative) path to input image file")
    parser.add_argument("--output_folder", default = '', help = "(Relative) path to output image file. If empty, image is not written.")
    parser.add_argument("--display",default = False, help = "Whether to display video on screen (can be slow)")

    args = parser.parse_args()

    print('Output_folder',args.output_folder)
    im_files = glob.glob(os.path.join(args.input_folder,'*.png'))
    print(os.path.join(args.input_folder+'*.png'))
    for pred_im in im_files:
        if args.output_folder:
            base = os.path.basename(pred_im)
            output_file = os.path.join(args.output_folder,os.path.splitext(base)[0][0:11])+".png" #os.path.splitext(base)[0]
            print(output_file)
        else:
            output_file = None
        seg_img = cv2.imread(pred_im, 0)
        lane_fit_on_predicted_image(inp = seg_img, lane_fit = True, output_file = output_file, display=True) #visualize = "segmentation"
        #vis_img, out_img, fit =
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
