#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:29:43 2020

@author: marianne
"""

#!/usr/bin/env python
import os
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../image-segmentation-keras')
from keras_segmentation.data_utils.data_loader import get_image_arr , get_segmentation_arr
from keras_segmentation.predict import predict, model_from_checkpoint_path
from keras_segmentation import metrics
#from keras_segmentation.train import find_latest_checkpoint
import numpy as np
from tqdm import tqdm
import argparse
import time
#from scipy.misc import imresize
from PIL import Image

#Temporary code with one image, will be added back to evaluate function when done
#TODO: should be based on colormaps for error images and masks separately, instead of RGB encoding

labels = [0,1,2]

gt = np.load('gt_array.npy')
pr = np.load('pr_array.npy')
inp = r'Frogn_Dataset/images_prepped_test/20190913_LR1_S_0420.png'

image_mode = 'F'
input_image = Image.open(inp)
iou = metrics.get_iou( gt , pr ,3)
ious = []
ious.append( iou )

#visualization

fig = plt.figure()
plt.title("IOU:"+str(iou))
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
ax4.imshow(input_image)
ax4.title.set_text('Input image')

#make overlay image

mask = np.zeros((gt.shape[0],gt.shape[1],3),dtype='uint8')
im_resized = np.array(input_image.resize((mask.shape[1],mask.shape[0])))
#scaling_factor = 255/np.max(labels)

error_mask = pr-gt
mask[:,:,0] = 255*np.uint8(error_mask > 0)
mask[:,:,2] = 255*np.uint8(error_mask < 0)
mask[:,:,1] = 255*(gt-1)*np.uint8(error_mask == 0)#255*np.uint8(gt==2)*np.uint8(error_mask == 0)

#fixme use a different color map?

alpha = 0.4
vis_img = np.uint8((1-alpha)*im_resized + alpha*mask)
#vis_img = Image.blend(mask_image,im_resized,alpha=0.5)
plt.figure(110)
plt.imshow(mask)

plt.figure(111)
plt.imshow(vis_img) # fixme downsample input image

plt.figure(112)
plt.imshow(input_image)



