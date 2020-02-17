#!/usr/bin/env python
from __future__ import division
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

'''
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
    
    fig = plt.figure(111)#figsize = (figure_width_mm/25.4,figure_width_mm/25.4))
    plt.imshow(vis_img)
    plt.axis('off')    
    return fig
'''

def blend_color_and_image(image,mask,color_code,alpha):
    #Blend colored mask and input image
    #Assuming 3-channel image, one-channel mask
    mask = np.tile(mask[:,:,np.newaxis],[1,1,3])
    return np.uint8(mask * (1-alpha) * color_code + mask * alpha * image + np.logical_not(mask) * image)

def vis_pred_vs_gt_overlay(inp, pr, gt, figure_width_mm):
    input_image = Image.open(inp)

    #Make overlay image
    im_resized = np.array(input_image.resize((gt.shape[1],gt.shape[0])))
    
    error_mask = pr-gt
    #Make non-overlapping masks
    fp_mask = np.uint8(error_mask > 0)  
    fn_mask = np.uint8(error_mask < 0) #False negatives for class 1
    gt_mask = (gt-1)*np.uint8(error_mask == 0) #class 1 in ground truth encoded green
    
    #Blending
    alpha = 0.4
    fp_color_code = np.array([1,0,1])*255 #magenta
    fn_color_code = np.array([0,0,1])*255 #blue
    gt_color_code = np.array([0,1,0])*255 #green
    
    im_vis = blend_color_and_image(im_resized,fp_mask,fp_color_code,alpha)
    im_vis= blend_color_and_image(im_vis,fn_mask,fn_color_code,alpha)
    im_vis= blend_color_and_image(im_vis,gt_mask,gt_color_code,alpha)
    return im_vis


#%% Script for generating prediction mask figures for CASE 2020 paper
    
#Setup: 
#input images
image_folder = os.path.join('../Frogn_Dataset','images_prepped_test')
annotations_folder = os.path.join('../Frogn_Dataset','annotations_prepped_test')
inp_names = ['20190703_LR_N_0785.png',
             '20190913_LR1_S_0965.png',
             '20190609_LR_N_0000.png',
             '20190913_LR2_N_1760.png',
             '20190913_LR3_N_0950.png',
             '20190913_LR4_S_2275.png']

output_name= 'prediction_overlay'
#model
checkpoints_path = os.path.join('../models','resnet50_segnet')
epoch = 20

figure_dpi = 300
figure_width_mm = 88.57
image_spacing_mm = figure_width_mm*0.05
print(len(inp_names))
figure_height_mm = ((176*len(inp_names)*1.25)/(320*2*1.05))*figure_width_mm #Adjusting height to get a tight fit

#Load model first
model = model_from_checkpoint_path(checkpoints_path,epoch)

fig,ax = plt.subplots(nrows=len(inp_names),ncols=2,figsize=(figure_width_mm/25.4,figure_height_mm/25.4))

for ind,fname in enumerate(inp_names):
    #Load images
    inp = os.path.join(image_folder,fname)
    ann = os.path.join(annotations_folder,fname)
    #Run prediction
    pr = predict_fast(model , inp )
    gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height, no_reshape=True)
    gt = gt.argmax(-1)
    iou = metrics.get_iou( gt , pr , model.n_classes )

    #--Make overlay image
    im_overlay = vis_pred_vs_gt_overlay(inp,pr,gt,figure_width_mm = 85) #check column widht for CASE
    
    #Put in subplots
    
    ax[ind][0].axis('off')
    ax[ind][0].imshow(Image.open(inp))
    ax[ind][0].xaxis.set_major_locator(plt.NullLocator())
    ax[ind][0].yaxis.set_major_locator(plt.NullLocator())
    
    ax[ind][1].axis('off')
    ax[ind][1].imshow(im_overlay)
    ax[ind][1].xaxis.set_major_locator(plt.NullLocator())
    ax[ind][1].yaxis.set_major_locator(plt.NullLocator())

fig.set_constrained_layout_pads(w_pad=0, h_pad = 0, hspace=0., wspace=0.)
fig.savefig(output_name, dpi = figure_dpi,bbox_inches='tight',pad_inches=0)
