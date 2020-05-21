#Prepare dataset for segNet
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import cv2

input_dir = '/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/auto_label/output'
annotation_dir = os.path.join(input_dir,'automatic_annotations/annotation_arrays') #'straight_annotations'#'automatic_annotations/annotation_arrays'
image_dir = os.path.join(input_dir,'images_only')
visualization_dir = os.path.join(input_dir,'automatic_annotations/visualization') #'straight_annotations #'automatic_annotations/visualization'
output_dir = os.path.join(input_dir,'prepped_data') #'prepped_data'

#Input per row
prefix = '20191010_L1_N'#'20191010_L1_N' #'20191010_L3_N_morning' #'20191010_L3_S_morning_slalom'
start_frame_num = 96 #96 #58 #30
end_frame_num = 2112  #2112 #1293 #3763

#Common setup
sample_step = 10

#Get files with the right prefix
all_ann_files = sorted(os.listdir(annotation_dir))
pat = re.compile(prefix + '_\d\d\d\d.npy',re.UNICODE)  
ann_files = list(filter(pat.match, all_ann_files))

first_frame_num = int(os.path.splitext(os.path.basename(ann_files[0]))[0][-4:]) #frame number of first file
start_frame_ind = start_frame_num-first_frame_num
print('start_frame_ind', start_frame_ind)

for n in range(start_frame_ind,len(ann_files),sample_step): #start on first valid frame, then step by sample step
    ann_file = ann_files[n]
    print(ann_file)
    im_name = os.path.splitext(os.path.basename(ann_file))[0]
    frame_num = int(im_name[-4:])
    if frame_num <= end_frame_num: #skip invalid files at the end
        #Load files
        ann = np.load(os.path.join(annotation_dir,ann_file))
        im = plt.imread(os.path.join(image_dir,im_name + '.png'))
        vis = plt.imread(os.path.join(visualization_dir, im_name + '.png'))
        #Crop annotation and image
        #Height from 480 to 360 from the top, keeping 640 width
        ann_cropped = ann[-360:,-640:]
        im_cropped = im[-360:,-640:]
        vis_cropped = vis[-360:,-640:]

        #-- Save 
        save_dir = os.path.join(output_dir,'train')
        
        print('Saving to ', save_dir)
        plt.imsave(os.path.join(save_dir,'images',im_name + '.png'),im_cropped)
        plt.imsave(os.path.join(save_dir,'visualization',im_name + '.png'),vis_cropped)
        cv2.imwrite(os.path.join(save_dir,'annotations',im_name + '.png'),ann_cropped)
        

