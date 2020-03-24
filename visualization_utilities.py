import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def blend_color_and_image(image,mask,color_code=[0,255,0],alpha=0.5):
    #Blend colored mask and input image
    #Input:
    #   3-channel image (numpy array)
    #   1-channel (integer) mask

    #convert input to uint8 image
    if image.dtype is np.dtype('float32') or np.dtype('float64') and np.max(image) <= 1:
        image = np.uint8(image*255)

    mask = np.tile(mask[:,:,np.newaxis],[1,1,3])
    #convert nan values to zero
    mask = np.nan_to_num(mask)
    blended_im = np.uint8((mask * (1-alpha) * color_code) + (mask * alpha * image) + (np.logical_not(mask) * image)) #mask + image under mask + image outside mask
    return blended_im

def vis_pred_vs_gt_overlay(inp, pr, gt, figure_width_mm=None,alpha = 0.5):
    #Visualize segmentation prediction and false positives/negatives
    
    #NB 3-class problem with background not handled properly
    input_image = Image.open(inp)

    #Make overlay image
    #mask = np.zeros((gt.shape[0],gt.shape[1],3),dtype='uint8')
    im_resized = np.array(input_image.resize((gt.shape[1],gt.shape[0])))
    
    error_mask = pr-gt
    #Make non-overlapping masks
    fp_mask = np.uint8(error_mask > 0)  
    fn_mask = np.uint8(error_mask < 0) #False negatives for class 1
    gt_mask = (gt-1)*np.uint8(error_mask == 0) #class 1 in ground truth encoded green
    
    #Blending
    fp_color_code = np.array([1,0,1])*255 #magenta
    fn_color_code = np.array([0,0,1])*255 #blue
    gt_color_code = np.array([0,1,0])*255 #green
    
    im_vis = blend_color_and_image(im_resized,fp_mask,fp_color_code,alpha)
    im_vis= blend_color_and_image(im_vis,fn_mask,fn_color_code,alpha)
    im_vis= blend_color_and_image(im_vis,gt_mask,gt_color_code,alpha)
    #vis_img = np.uint8(fp_mask * (1-alpha) * color_code + fp_mask * alpha * im_resized + np.logical_not(fp_mask) * im_resized)
    #vis_img = np.uint8(fp_mask*(1-alpha)*im_resized + (fp_mask*alpha)*color_code))
    
    fig = plt.figure(111)#figsize = (figure_width_mm/25.4,figure_width_mm/25.4))
    plt.imshow(im_vis)
    plt.axis('off')    
    return fig

def vis_pred_vs_gt_separate(inp,pr,gt):   
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(gt-pr)
    #ax1.colorbar()
    ax1.title.set_text("Difference GT-pred")

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(gt)
    ax2.title.set_text('GT')

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(pr)
    ax3.title.set_text('pred')

    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(plt.imread(inp))
    ax4.title.set_text('Input image')
    
    return fig

