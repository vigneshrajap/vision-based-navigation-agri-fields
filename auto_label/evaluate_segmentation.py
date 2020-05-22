import sys 

from predict import evaluate_and_visualize

import os
import glob
import numpy as np

if __name__ == "__main__":
    model_folder = 'models'
    model_prefix = 'autolabel_L1_N_ignorezero_2020-21-05-1613'
    epoch = 10
    data_folder = 'output/prepped_data/val'
    data_prefix = ''
    output_folder = 'output/segmentation'

    model_path_and_prefix = os.path.join(model_folder,model_prefix)
    im_files = glob.glob(os.path.join(data_folder,'images', data_prefix + '*.png'))
    ann_files = glob.glob(os.path.join(data_folder,'annotations' , data_prefix + '*.png'))
    output_path = os.path.join(output_folder,model_prefix + '_' + str(epoch))
    os.makedirs(output_path, exist_ok = True)

    print('Running evaluation on ',len(im_files),'images, and',len(ann_files),'annotations:')
    ious = evaluate_and_visualize(model=None , #fetched from path and epoch number
    inp_images= im_files , 
    annotations= ann_files, 
    checkpoints_path=model_path_and_prefix,
    epoch = epoch, 
    visualize = True, 
    ignore_zero_class = True,
    output_folder = output_path)

    ious = np.array( ious )
    print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
    print("Total  IoU "  ,  np.mean(ious[1:2] ))

    with open(os.path.join(output_path, "iou.txt"), 'w') as print_file:
        print("Class wise IoU "  ,  np.mean(ious , axis=0 ), file = print_file)
        print("Total  IoU "  ,  np.mean(ious[1:2] ), file = print_file)
