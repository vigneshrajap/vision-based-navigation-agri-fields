#!/usr/bin/env python
import os
import cv2
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import glob
from PIL import Image
import os.path as osp

roi_x = 208 #120
roi_y = 120 #208

#input_dir = expanduser("~/Third_Paper/Datasets/Frogn_Dataset/annotations_prepped_test/") #/frogn_2%04d.jpg"%d
#output_dir = expanduser("~/Third_Paper/Datasets/Frogn_Dataset_32x_352/annotations_prepped_test/")

input_dir = expanduser("~/Third_Paper/Datasets/october_10_data_collection_bags/dataset_17_pics/") #/frogn_2%04d.jpg"%d
output_dir = expanduser("~/Third_Paper/Datasets/october_10_data_collection_bags/dataset_17_pics1/")


for label_file in glob.glob(osp.join(input_dir, '*.jpg')):
        print(label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            img = cv2.imread(label_file)
            # cv2.imshow('preview', img)
            # k = cv2.waitKey(1) & 0xFF
            # if k == 27:
            #    break

            # Getting ROI
            iheight, iwidth = img.shape[:2]
            Roi = img[roi_y+8:iheight,0:iwidth]

            # Store the Cropped Images
            cv2.imwrite(output_dir+base+'.png', Roi)
