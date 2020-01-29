#!/usr/bin/env python
import os
import cv2
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hides the pygame version, welcome msg
from os.path import expanduser
import glob
from PIL import Image
import os.path as osp

input_dir = expanduser("~/Third_Paper/Datasets/Frogn_Dataset_32x_352/") #/frogn_2%04d.jpg"%dannotations
output_dir = expanduser("~/Third_Paper/Datasets/Frogn_Dataset_32x_352/images_prepped_test/")

# input_dir = expanduser("~/Third_Paper/Datasets/Frogn_Dataset/") #/frogn_2%04d.jpg"%dannotations
# output_dir = expanduser("~/Third_Paper/Datasets/Frogn_Dataset/")

for label_file in glob.glob(osp.join(input_dir, '*.png')):
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
            Roi = img[0:iheight,0:iwidth]
            print base[13]
            if base[13]==str('S'):
              #print 1
              #base[12] = str('N')
              print base[0:13]+str('N')+base[14:19]
              # Store the Cropped Images
              cv2.imwrite(output_dir+base[0:13]+str('N')+base[14:19]+'.png', Roi)
