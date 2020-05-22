import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def blend_color_and_image(image,mask,color_codes,alpha=0.5):
    #Blend colored mask and input image
    #Input:
    #   3-channel image (numpy array)
    #   1-channel (integer) mask with N different values
    #   Nx3 list of RGB color codes (for each value mask).Example for mask with values 0,1,2:

    mask_values = np.unique(mask)
    color_codes = np.array(color_codes)
    if not color_codes.shape[0] == len(mask_values):
        assert('Number of mask values and color codes does not correspond')
    #convert input to uint8 image
    if image.dtype is np.dtype('float32') or np.dtype('float64') and np.max(image) <= 1:
        image = np.uint8(image*255)

    mask = np.tile(mask[:,:,np.newaxis],[1,1,3])
    #convert nan values to zero
    mask = np.nan_to_num(mask)

    #Add one layer per value (larger than zero) in mask
    blended_im = image
    for ind in range(0,len(mask_values)):
        if color_codes[ind,0] is not None:
            val = mask_values[ind]
            tmp_mask = (mask == val)
            blended_im = np.uint8((tmp_mask * (1-alpha) * color_codes[ind,:]) + (tmp_mask * alpha * blended_im) + (np.logical_not(tmp_mask) * blended_im)) #mask + image under mask + image outside mask
    return blended_im

def vis_pred_overlay(inp,pr, fig = None):
    input_image = Image.open(inp)
    input_image.convert('RGB')
    #Make overlay image
    #mask = np.zeros((gt.shape[0],gt.shape[1],3),dtype='uint8')
    im_resized = np.array(input_image.resize((pr.shape[1],pr.shape[0])))[:,:,:3]
    overlay_im = blend_color_and_image(im_resized,pr,color_codes = [[None,None,None],[255,255,0],[0,0,255]],alpha=0.8) 

    if fig is None: fig = plt.figure(111)
    plt.imshow(overlay_im)
    plt.axis('off')    
    return fig

def vis_pred_vs_gt_overlay_and_separate(inp,pr,gt):
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
    vis_pred_overlay(inp,pr, fig = ax4)
    ax4.title.set_text(os.path.basename(inp))
    
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

if __name__ == "__main__":
    #Test visualization
    inp = os.path.join('/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/auto_label/output/images_only/20191010_L1_N_1093.png')
    input_image = Image.open(inp)
    input_image.convert('RGB')
    mask = np.load(os.path.join('/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/auto_label/output/automatic_annotations/annotation_arrays/20191010_L1_N_1093.npy'))

    im_resized = np.array(input_image.resize((mask.shape[1],mask.shape[0])))[:,:,:3]

    vis = blend_color_and_image(im_resized, mask,color_codes=[[None,None,None],[255,255,0],[0,0,255]],alpha=0.8)
    plt.imshow(vis)
    plt.show()
    plt.imsave('vis.png',vis)


