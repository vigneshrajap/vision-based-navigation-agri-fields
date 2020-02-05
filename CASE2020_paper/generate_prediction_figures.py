#!/usr/bin/env python
import os
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../../image-segmentation-keras') #use our own version of keras image segmentation
from keras_segmentation.predict import predict, predict_fast, model_from_checkpoint_path
from keras_segmentation.data_utils.data_loader import get_segmentation_arr
#from keras_segmentation.predict import predict, predict_fast, model_from_checkpoint_path
from keras_segmentation import metrics
from keras_segmentation.train import find_latest_checkpoint
import numpy as np
from tqdm import tqdm
import argparse
import time
from PIL import Image

def vis_pred_vs_gt_overlay(inp, pr, gt, figure_width_mm):
    #Visualize segmentation prediction and false positives/negatives
    
    #NB 3-class problem with background not handled properly
    input_image = Image.open(inp)

    #Make overlay image
    mask = np.zeros((gt.shape[0],gt.shape[1],3),dtype='uint8')
    im_resized = np.array(input_image.resize((mask.shape[1],mask.shape[0])))
    
    error_mask = pr-gt
    mask[:,:,0] = 255*np.uint8(error_mask > 0) #False positives for class 1
    mask[:,:,2] = 255*np.uint8(error_mask < 0) #False negatives for class 1
    mask[:,:,1] = 255*(gt-1)*np.uint8(error_mask == 0) #class 1 in ground truth encoded green
    
    alpha = 0.4
    vis_img = np.uint8((1-alpha)*im_resized + alpha*mask)
    
    fig = plt.figure(111,figsize = (figure_width_mm/25.4,figure_width_mm/25.4))
    plt.imshow(vis_img)
    plt.axis('off')
    
    return fig


#%% Script for generating prediction mask figures for CASE 2020 paper
    
#Setup: 
#input images
image_folder = os.path.join('../Frogn_Dataset','images_prepped_test')
annotations_folder = os.path.join('../Frogn_Dataset','annotations_prepped_test')
inp_names = ['20190703_LR_N_0785.png','20190913_LR1_S_0965.png']
#model
checkpoints_path = os.path.join('../models','resnet50_segnet')
epoch = 20

figure_dpi = 300

#Load model first
model = model_from_checkpoint_path(checkpoints_path,epoch)

for fname in inp_names:
    #Load images
    inp = os.path.join(image_folder,fname)
    ann = os.path.join(annotations_folder,fname)
    #Run prediction
    pr = predict_fast(model , inp )
    gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height, no_reshape=True)
    gt = gt.argmax(-1)
    iou = metrics.get_iou( gt , pr , model.n_classes )

    #--Make overlay image
    fig1 = vis_pred_vs_gt_overlay(inp,pr,gt,figure_width_mm = 40) #check column widht for CASE
    fig1.canvas.set_window_title("Predicted mask and errors")
    output_name1 = 'prediction_overlay_'+os.path.basename(inp)
    fig1.savefig(output_name1, dpi = figure_dpi)
    print('Saving to: ', output_name1)
    

