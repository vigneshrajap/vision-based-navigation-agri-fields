import sys 
#sys.path.append('../../image-segmentation-keras')
import keras_segmentation
sys.path.append('..')
from lane_detection_segnet_evaluate import evaluate
from lane_detection_segnet_evaluate import evaluate
import os
import glob

if __name__ == "__main__":
    model_folder = 'models'
    model_prefix = 'autolabel_L3_S_slalom_2020-20-05-1646'
    epoch = 5
    data_folder = 'output/prepped_data/val'
    data_prefix = ''
    output_folder = 'output/segmentation'

    model_path_and_prefix = os.path.join(model_folder,model_prefix)
    im_files = glob.glob(os.path.join(data_folder,'images', data_prefix + '*.png'))
    ann_files = glob.glob(os.path.join(data_folder,'annotations' , data_prefix + '*.png'))
    output_path = os.path.join(output_folder,model_prefix)

    print('Running evaluation on ',len(im_files),'images, and',len(ann_files),'annotations:')
    evaluate(model=None , #fetched from path and epoch number
    inp_images= im_files , 
    annotations= ann_files, 
    checkpoints_path=model_path_and_prefix,
    epoch = epoch, 
    visualize = True, 
    output_folder = output_path)
