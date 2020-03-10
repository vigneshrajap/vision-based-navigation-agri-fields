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

import sys
sys.path.insert(1, '../image-segmentation-keras')
from keras_segmentation import predict
import sliding_window_approach
from geometry_msgs.msg import Pose, PoseArray
import time
start_time = time.time()

def upscaling_warping_parameters(rgb_img, seg_arr, class_number, crop_ratio):
   # Reshaping the Lanes Class into binary array and Upscaling the image as input image
   dummy_img = np.zeros(seg_arr.shape)
   dummy_img += ((seg_arr[:,: ] == class_number)*(255)).astype('uint8') # Class Number 2 belongs to Lanes
   original_h, original_w = rgb_img.shape[0:2]
   upscaled_img = cv2.resize(dummy_img, (original_w,original_h)).astype('uint8')

   # Perspective warp
   rheight, rwidth = upscaled_img.shape[:2]
   Roi_g = upscaled_img[int(crop_ratio*rheight):rheight,0:rwidth]
   dst_size = Roi_g.shape[:2]
   # src=np.float32([(0.1,0), (0.8,0), (0,1), (1,1)])
   src=np.float32([(0,0.3), (1,0.3), (0,1), (1,1)])
   dst=np.float32([(0,0), (1,0), (0,1), (1,1)])
   return upscaled_img, Roi_g, src, dst, dst_size

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
   final_img = input_image.copy()
   rheight, rwidth = final_img.shape[:2]
   final_img[int(rheight*crop_ratio):rheight,0:rwidth] = cv2.addWeighted(final_img[int(rheight*crop_ratio):int(rheight),0:rwidth],
                                                           1, invwarp, 0.9, 0)
   return out_img, midPoints, invwarp, final_img

def run_lane_fit(input_image, seg_arr, class_number = 2,  crop_ratio = 0.2):
# Extract Interesting Class (2 - Lanes in this case) from predictions
# Ratio to crop the background parts in the image from top

   # Setting the parameters for upscaling and warping-unwarping
   upscaled_img, Roi_img, src, dst, dst_size = upscaling_warping_parameters(input_image, seg_arr, class_number, crop_ratio)

   # Sliding Window Approach on Lanes Class from segmentation Array and fit the poly curves
   warp_img, out_img, curves, lanes, ploty, modifiedCenters = lane_fit_on_prediction(Roi_img, src, dst, dst_size)

   # Overlay the inverse warped image on input image
   polyfit_img, centerLine, invwarp, final_img = visualize_lane_fit(input_image, out_img, curves, lanes, ploty, modifiedCenters, src, dst, dst_size, crop_ratio)
   return warp_img, final_img, centerLine

def visualize_segmentation(input_img, seg_arr, n_classes, class_number = 2, display = False, output_file = None):
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
                  upscaled_img_rgb)) # np.ones(overlay_img.shape,dtype=np.uint8)*128
    ))

    return upscaled_img #vis_img #vis_img

def visualization(input_img, seg_arr=None, lane_fit = None, evaluation = None, n_classes=None, visualize = None, display=False, output_file=None):
    class_number = 2 # 1 for Crops, 2 for Lanes
    crop_ratio = 0.2

    #visualize: None, "all" or one of, "segmentation", "lane_fit", "evaluation"
    #with or without gt label and IOU result
    if visualize == "segmentation":
        vis_img = visualize_segmentation(input_img, seg_arr, n_classes, class_number, display=display, output_file=output_file)
        cv2.imwrite(output_file, vis_img )
    if visualize == "lane_fit":
        warp_img, vis_img, centerLine = run_lane_fit(input_img, seg_arr, class_number, crop_ratio)
        return vis_img, centerLine ##fixme

    # if display:
    #     cv2.imshow('Prediction', vis_img)
    # if not output_file is None:
    #     cv2.imwrite(output_file, vis_img )
    return vis_img

def predict_on_image(model, inp, lane_fit = False, evaluate = False, visualize = None, output_file = None, display=False):

    #visualize: None, "all" or one of, "segmentation", "lane_fit"

    #Run prediction (and optional, visualization)
    seg_arr, input_image = predict.predict_fast(model, inp)

    if lane_fit:
        class_number = 2 # Extract Interesting Class (2 - Lanes in this case) from predictions
        crop_ratio = 0.2 # Ratio to crop the background parts in the image from top
        out_img, fit = run_lane_fit(input_image, seg_arr, class_number, crop_ratio)
    else:
        fit = None
        out_img = None

    return seg_arr, input_image, out_img, fit

def predict_on_video():
    #fixme
    #video_gen = video_setup()

    #for frame in video_gen:
        #predict_on_image(frame)

    #video_cleanup()
    return None


def main():
    parser = argparse.ArgumentParser(description="Example: Run prediction on an image folder. Example usage: python lane_predict.py --model_prefix=models/resnet_3class --epoch=25 --input_folder=Frogn_Dataset/images_prepped_test --output_folder=.")
    parser.add_argument("--model_prefix", default = '', help = "Prefix of model filename")
    parser.add_argument("--epoch", default = None, help = "Checkpoint epoch number")
    parser.add_argument("--input_folder",default = '', help = "(Relative) path to input image file")
    parser.add_argument("--output_folder", default = '', help = "(Relative) path to output image file. If empty, image is not written.")
    parser.add_argument("--display",default = False, help = "Whether to display video on screen (can be slow)")

    args = parser.parse_args()

    #Load model
    model = predict.model_from_checkpoint_path(args.model_prefix, args.epoch)

    print('Output_folder',args.output_folder)
    im_files = glob.glob(os.path.join(args.input_folder,'*.png'))
    print(os.path.join(args.input_folder+'*.png'))
    for im in im_files:
        if args.output_folder:
            base = os.path.basename(im)
            output_file = os.path.join(args.output_folder,os.path.splitext(base)[0])+"_crop_pred.png" #
            print(output_file)
        else:
            output_file = None

        seg_arr, input_image, out_img, fit = predict_on_image(model,inp = im, lane_fit = False, evaluate = False, visualize = "segmentation", output_file = output_file, display=True)
        vis_img = visualization(input_image, seg_arr=seg_arr, lane_fit = None, evaluation = None, n_classes=3, visualize = "segmentation", display=False, output_file=output_file)
        #
        # print(timeit.timeit(stmt = "for_loop(seq)",
        #                     setup="seq='Pylenin'",
        #                     number=10000))

        #print(timeit.timeit(predict_on_image(model,inp = im, lane_fit = False, evaluate = False, visualize = "segmentation", output_file = output_file, display=True)))
        print("--- %s seconds ---" % (time.time() - start_time))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
