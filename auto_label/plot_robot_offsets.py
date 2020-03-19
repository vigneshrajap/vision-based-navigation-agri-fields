#!/usr/bin/env python
import numpy as np
from utilities import read_robot_offset_from_file
import matplotlib.pyplot as plt
import os
import glob

#Plot the angular offsets for all the rows
input_files = glob.glob(os.path.join('..','Frogn_Dataset','robot_offsets','*')) #
#['../Frogn_Dataset/training_images_offset_20191010_L1_N.txt']
plt.figure('a')
plt.figure('l')
for inp in input_files:
    robot_offset_file = os.path.join(inp)
    lateral_offsets, angular_offsets, row_inds = read_robot_offset_from_file(robot_offset_file)
    plt.figure('a')
    plt.plot(row_inds,angular_offsets)
    plt.title('Angular offset')

    plt.figure('l')
    plt.plot(row_inds,lateral_offsets)
    plt.title('Lateral offset')

#print input_files[0][31:44]
plt.figure('a')
plt.legend(input_files)
plt.xlabel('Frame number')
plt.ylabel('Radians')

plt.figure('l')
plt.legend(input_files)
plt.xlabel('Frame number')
plt.ylabel('Meters')

plt.show()
