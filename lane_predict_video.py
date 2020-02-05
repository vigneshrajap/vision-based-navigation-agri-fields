#!/usr/bin/env python
#import keras_segmentation
#from keras_segmentation.predict import model_from_checkpoint_path
import os
#os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import argparse
#from moviepy.editor import VideoFileClip
import cv2
import numpy as np

import sys
sys.path.insert(1, '../image-segmentation-keras')
from keras_segmentation import predict
import sliding_window_approach



def main():

    parser = argparse.ArgumentParser(description="Run prediction on a video.")
    parser.add_argument("--model_prefix", default = '', help = "Prefix of model filename")
    parser.add_argument("--epoch", default = None, help = "Checkpoint epoch number")
    parser.add_argument("--input_video_file",default = '', help = "(Relative) path to input video file")
    parser.add_argument("--output_video_file", default = '', help = "(Relative) path to output video file. If empty, video is not written.")
    parser.add_argument("--display",default = False, help = "Whether to display video on screen (can be slow)")
    parser.add_argument("--frame_step",default = 1, help = "Input video frame step")

    args = parser.parse_args()

    #Load model
    model = predict.model_from_checkpoint_path(args.model_prefix, args.epoch)

    #Initialize video capture
    cap = cv2.VideoCapture(args.input_video_file)
    # Check if video opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    #Initialize video writer
    if args.output_video_file:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG") # ('I','Y','U','V') #tried('M','J','P','G')
        wr = None
        (out_h, out_w) = (None, None)
        isColor = True
        fps = 30
        print('Will write to output video file : ', args.output_video_file)

    frame_count = 0
    frame_step = int(args.frame_step)
    # Read until video is completed
    while(cap.isOpened()):

      # Capture frame-by-frame
        ret, rgb_img = cap.read()
        if ret == True:
           print('Frame ',frame_count)
           #Run prediction on video frame
           seg_arr,inp = predict.predict_fast(model,rgb_img)
           #seg_img = predict.segmented_image_from_prediction(seg_arr, n_classes = model.n_classes,input_shape = rgb_img.shape)
           #overlay_img = cv2.addWeighted(rgb_img,0.7,seg_img,0.3,0)

           # Stack input and segmentation in one video

           # Reshaping the Lanes Class into binary array and Upscaling the image as input image
           grey_img = seg_arr
           dummy_img = np.zeros((model.output_height, model.output_width))
           dummy_img += ((grey_img[:,: ] == 2)*(255)).astype('uint8') # Class Number 2 belongs to Lanes
           original_h, original_w = rgb_img.shape[0:2]
           dummy_img = cv2.resize(dummy_img, (original_w,original_h)).astype('uint8')

           # Perspective warp
           rheight, rwidth = dummy_img.shape[:2]
           Roi_g = dummy_img[int(0.3*rheight):rheight,0:rwidth]
           roiheight, roiwidth = Roi_g.shape[:2]
           src=np.float32([(0.1,0), (0.8,0), (0,1), (1,1)])
           dst=np.float32([(0,0), (1,0), (0,1), (1,1)])
           warped_img, M  = sliding_window_approach.perspective_warp(Roi_g, (roiheight, roiwidth), src, dst)

           # InitialPoints Estimation using K-Means clustering
           modifiedCenters = sliding_window_approach.initialPoints(warped_img)

           # Sliding Window Search
           out_img, curves, lanes, ploty = sliding_window_approach.sliding_window(warped_img, modifiedCenters)

           # Visualize the fitted polygonals (One on each lane and on average curve)
           out_img, midLane_i = sliding_window_approach.visualization_polyfit(out_img, curves, lanes, ploty, modifiedCenters)
           # Inverse Perspective warp
           invwarp, Minv = sliding_window_approach.inv_perspective_warp(out_img, (roiwidth, roiheight), dst, src)

           midPoints = []
           for i in midLane_i:
             point_wp = np.array([i[0],i[1],1])
             midLane_io = np.matmul(Minv, point_wp) # inverse-M*warp_pt
             midLane_n = np.array([midLane_io[0]/midLane_io[2],midLane_io[1]/midLane_io[2]]) # divide by Z point
             midLane_n = midLane_n.astype(int)
             midPoints.append(midLane_n)

           print midPoints.shape

           # Combine the result with the original image
           rgb_img[int(rheight*0.3):rheight,0:rwidth] = cv2.addWeighted(rgb_img[int(rheight*0.3):int(rheight),0:rwidth],
                                                                           1, invwarp, 0.9, 0)
           result = rgb_img

           # vis_img = np.vstack((
           #         np.hstack((rgb_img,
           #                    seg_img)),
           #         np.hstack((overlay_img,
           #                    np.zeros(overlay_img.shape)*128))
           # ))

           #vis_img = np.vstack((np.hstack((rgb_img, seg_img)),np.hstack((overlay_img,np.zeros(overlay_img.shape)))))

           #Write video
           # if args.output_video_file:
           #     if wr is None: #if writer is not set up yet
           #         (out_h,out_w) = result.shape[:2]
           #         wr = cv2.VideoWriter(args.output_video_file,fourcc,fps,(out_w,out_h),isColor)
           #     wr.write(result)

           #Display on screen
           if args.display:
               cv2.startWindowThread()
               cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
               cv2.resizeWindow('preview', 800,800)
               cv2.imshow('preview', result)

           # Press Q on keyboard to  exit
           if cv2.waitKey(25) & 0xFF == ord('q'):
               print('Q pressed, breaking')
               break
           #Skip frames
           frame_count +=frame_step
           cap.set(1, frame_count)
        else:
           break

    # cleanup
    cv2.destroyAllWindows()
    cap.release()
    if args.output_video_file: wr.release()

if __name__ == "__main__":
    main()
