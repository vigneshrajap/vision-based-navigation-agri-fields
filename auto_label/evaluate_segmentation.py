import sys 

from predict import evaluate_and_visualize

import os
import glob
import numpy as np

if __name__ == "__main__":
    model_folder = 'models'
    model_prefix = 'manuallabel_all_2020-02-06-1440'
    epoch = 19
    data_folder = 'output/prepped_data/val'
    data_prefix = ''
    output_folder = 'output/segmentation'

    model_path_and_prefix = os.path.join(model_folder,model_prefix)
    im_files = glob.glob(os.path.join(data_folder,'images_manual', data_prefix + '*.png'))
    ann_files = glob.glob(os.path.join(data_folder,'annotations_manual' , data_prefix + '*.png'))
    output_path = os.path.join(output_folder,model_prefix + '_' + str(epoch))
    os.makedirs(output_path, exist_ok = True)

    print('Running evaluation on ',len(im_files),'images, and',len(ann_files),'annotations:')
    class_wise_iou, mean_iou, frequency_weighted_iou = evaluate_and_visualize(model=None , #fetched from path and epoch number
    inp_images= im_files , 
    annotations= ann_files, 
    checkpoints_path=model_path_and_prefix,
    epoch = epoch, 
    visualize = True, 
    ignore_zero_class = True,
    output_folder = output_path)

    print("Class wise IoU "  ,  class_wise_iou)
    print("Total  IoU "  ,  mean_iou)
    print("Frequency weighted IoU ", frequency_weighted_iou)

    with open(os.path.join(output_path, "iou.txt"), 'w') as print_file:
        print("Class wise IoU "  ,  class_wise_iou, file = print_file)
        print("Total  IoU "  ,  mean_iou, file = print_file)
        print("Frequency weighted IoU ", frequency_weighted_iou, file = print_file)
