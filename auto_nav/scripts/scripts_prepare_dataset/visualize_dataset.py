#!/usr/bin/env python
import os
import cv2
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import glob
from PIL import Image
import os.path as osp
import random
import numpy as np

random.seed(0)
class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]

#input_dir = expanduser("~/planner_ws/src/vision-based-navigation-agri-fields/Frogn_Dataset/annotations_prepped_train/")
input_dir = expanduser("~/Third_Paper/Datasets/Frogn_Dataset_32x/annotations_prepped_train/")
output_dir = expanduser("~/Third_Paper/Datasets/Ground_Truth/")
n_classes = 5
colors = class_colors

for label_file in glob.glob(osp.join(input_dir, '*.png')):
        print(label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            seg = cv2.imread(label_file)
            print("Found the following classes" , np.unique( seg ))

            seg_img = np.zeros_like( seg )
            for c in range(n_classes):
            	seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
            	seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
            	seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')

            #cv2.imshow("seg_img" , seg_img )
            #cv2.waitKey()
            # Store the Cropped Images
            cv2.imwrite(output_dir+base+'.png', seg_img)
