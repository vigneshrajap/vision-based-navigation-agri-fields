import numpy as np
import csv
#Utility functions for automatic labelling

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

def read_robot_offset_from_file(filename,row_ind = None):
    # Open file with robot offset values and return lateral offset and angular offset.
    #if row not specified, return all rows
    with open(filename) as f:
        a = np.array(list(csv.reader(f, delimiter = '\t')))
        a = a[1:]
        a = np.array(a, dtype=float)

    if row_ind is not None:
        frames = list(a[:,1])
        try:
            ind = frames.index(float(row_ind))
            lateral_offset = a[ind,2]
            angular_offset = a[ind,3]
        except ValueError:
            print('Frame index ', str(row_ind), ' not in list')
            lateral_offset = None
            angular_offset = None
    else: 
        lateral_offset = a[:,2]
        angular_offset = a[:,3]
        row_ind = a[:,1]
        
    return lateral_offset, angular_offset, row_ind