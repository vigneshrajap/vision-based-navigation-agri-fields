#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import read_robot_offset_from_file, read_row_spec_from_file, blend_color_and_image
import os
import csv
import glob

def data_correction(robot_offset_file, visualize = False):

    lateral_offset, angular_offset,_ = read_robot_offset_from_file(robot_offset_file)
    angular_offset_mean = np.mean(angular_offset)
    angular_offset_corrected = angular_offset-angular_offset_mean

    if visualize:
        print('Angular offset mean:', angular_offset_mean)
        plt.figure()
        plt.plot(angular_offset)
        plt.plot(angular_offset_corrected)
        plt.plot(np.repeat(angular_offset_mean,len(angular_offset)))
        plt.ylim((-0.5,0.5))
        plt.show()

    return angular_offset_corrected

if __name__ == "__main__":
    dataset_dir = os.path.join('../Frogn_Dataset'),
    robot_offset_file = os.path.join('../Frogn_Dataset','robot_offsets/20191010_L4_N_slaloam_offsets.txt')

    data_correction(robot_offset_file,visualize = True)