#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:10:53 2019

@author: marianne
"""
import glob
import matplotlib as plt
import numpy as np

def crop_images(input_folder,crop_percentage):
    im_files = glob.glob(os.path.join(input_folder,'*.png'))
    print(im_files)
    for imf in im_files:
        im = plt.imread(imf)
        orig_shape=np.array(im.shape[0:2])
        new_shape = np.round(orig_shape*crop_percentage).astype('int') #crop to 60% of the image size
        offset = np.round(0.5*(orig_shape-new_shape)).astype('int')
        im_out = im[offset[0]:offset[0]+new_shape[0],offset[1]:offset[1]+new_shape[1]]
        plt.imsave(imf,im_out)