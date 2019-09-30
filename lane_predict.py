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


def visualize_prediction(input_img,seg_img):
    #with or without gt label and IOU result
    overlay_img = cv2.addWeighted(input_img,0.7,seg_img,0.3,0)
    # Stack input and segmentation in one video

    vis_img = np.vstack((
       np.hstack((input_img,
                  seg_img)),
       np.hstack((overlay_img,
                  np.ones(overlay_img.shape,dtype=np.uint8)*128))
    ))
    return vis_img

def predict_on_image(model,input_image,visualize = False, output_file = None, display=False):
    #Run prediction (and optional, visualization)
    seg_arr = predict.predict_fast(model,input_img)
    seg_img = predict.segmented_image_from_prediction(seg_arr, n_classes = model.n_classes, input_shape = rgb_img.shape)
    
    if visualize:
        vis_img = visualize_prediction(input_img,seg_img)
    else:
        vis_img = seg_img #Just plot the segmentation directly
        
    if display:
        cv2.imshow('Prediction', vis_img)    
    if not output_file is None:
        cv2.imwrite(  output_file , vis_img )
        
    return seg_arr, vis_img


def predict_on_video(model,input_video_file,visualize = False, output_video_file = None, display = False, frame_step = 1):
    #Run prediction (and optional, visualization) on video input
    
    #Initialize video capture
    cap = cv2.VideoCapture(input_video_file)
    # Check if video opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    #Initialize video writer
    if output_video_file:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG") # ('I','Y','U','V') #tried('M','J','P','G')
        wr = None
        (out_h, out_w) = (None, None)
        isColor = True
        fps = 30
        print('Will write to output video file : ', output_video_file)

    frame_count = 0
    frame_step = int(frame_step)
    # Read until video is completed
    while(cap.isOpened()):

      # Capture frame-by-frame
        ret, rgb_img = cap.read()
        if ret == True:
           print('Frame ',frame_count)
           #Run prediction on video frame
    
           
           seg_arr = predict.predict_fast(model,rgb_img)
           seg_img = predict.segmented_image_from_prediction(seg_arr, n_classes = model.n_classes, input_shape = rgb_img.shape)
           vis_img = visualize_prediction(rgb_img,seg_img)
           
           #Write video
           if output_video_file:
               if wr is None: #if writer is not set up yet
                   (out_h,out_w) = vis_img.shape[:2]
                   wr = cv2.VideoWriter(output_video_file,fourcc,fps,(out_w,out_h),isColor)
               wr.write(vis_img)
           #Display on screen
           if display:
               cv2.startWindowThread()
               cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
               cv2.resizeWindow('preview', 800,800)
               cv2.imshow('preview', vis_img)

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
    if output_video_file: wr.release()


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
    model = predict.model_from_checkpoint_path(args.model_prefix) #, args.epoch

    predict_on_video(model,args.input_video_file, args.output_video_file)

if __name__ == "__main__":
    main()


# parser = argparse.ArgumentParser(description="Run evaluation of model on a set of images and annotations.")
# parser.add_argument("--model_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/pre-trained_weights/segnet_weights/segnet", help = "Prefix of model filename")
# parser.add_argument("--train_images_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/images_prepped_train/")
# parser.add_argument("--train_annotations_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/annotations_prepped_train/")
# parser.add_argument("--inp_dir_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/images_prepped_test/")
# parser.add_argument("--out_dir_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/predicted_outputs_segnet/")
# parser.add_argument("--inp_path", default = os.path.expanduser('~')+"/Third_Paper/Datasets/Frogn_Dataset/images_prepped_test/frogn_10000.png")
# parser.add_argument("--pre_trained", default = "True", type=bool)
# parser.add_argument("--predict_multiple_images", default = "False", type=bool)
# args = parser.parse_args()
#
# model = keras_segmentation.models.segnet.segnet(n_classes=4,  input_height=320 , input_width=640)
# #pre_trained = True
# #predict_multiple_images = False
#
# if args.pre_trained:
#     model = model_from_checkpoint_path(args.model_path)
#
# else:
#     model.train(
#         train_images =  args.train_images_path,
#         train_annotations = args.train_annotations_path,
#         checkpoints_path = args.model_path, epochs=5
#     )
#
# if args.predict_multiple_images:
#     out = model.predict_multiple(
#           inp_dir = args.inp_dir_path,
#           checkpoints_path = args.model_path,
#           out_dir = args.out_dir_path
#      )
#
# else:
#     out = model.predict_segmentation(
#         inp= args.inp_path,
#         checkpoints_path = args.model_path,
#         out_fname=os.path.expanduser('~')+"/out.png"
#     )

# import matplotlib.pyplot as plt
# plt.imshow(out)
# plt.show()
