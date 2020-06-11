#!/usr/bin/env python
import os
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../../image-segmentation-keras') #use our own version of keras image segmentation
from keras_segmentation.predict import predict, predict_fast,model_from_checkpoint_path
from keras_segmentation.data_utils.data_loader import get_segmentation_arr
#from keras_segmentation.predict import predict, predict_fast, model_from_checkpoint_path
from keras_segmentation import metrics
from keras_segmentation.train import find_latest_checkpoint
import numpy as np
from tqdm import tqdm
import time
from PIL import Image

def evaluate( model=None , inp_images=None , annotations=None , checkpoints_path=None, epoch = None):
    #Finished implementation of the evaluate function in keras_segmentation.predict
    #Input: array of paths or nd arrays
    if model is None and ( not checkpoints_path is None ):
        model = model_from_checkpoint_path(checkpoints_path,epoch)

    ious = []
    pred_time = []
    for inp , ann   in zip( inp_images , annotations ):
        t_start = time.time()
        pr = predict_fast(model , inp )
        t_end = time.time()
        pred_time.append(t_end-t_start)
        print('Prediction time: ', t_end-t_start)
        
        gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height, no_reshape=True)
        gt = gt.argmax(-1)
        iou = metrics.get_iou( gt , pr , model.n_classes )
        ious.append( iou )


    print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
    print("Total  IoU "  ,  np.mean(ious ))
    print("Median prediction time:", np.median(pred_time))


def main():
    #Just running prediction to check timing
    data_folder = '../Frogn_Dataset'
    images_folder = 'images_prepped_test'
    annotations_folder = 'annotations_prepped_test'
    model_path = os.path.join('../models','resnet50_segnet')
    epoch = 20

    im_files = glob.glob(os.path.join(data_folder,images_folder, '*.png'))
    ann_files = glob.glob(os.path.join(data_folder,annotations_folder , '*.png'))

    print('Running evaluation on ',len(im_files),'images, and',len(ann_files),'annotations:')
    print(model_path)
    evaluate(model=None , inp_images= im_files , annotations= ann_files, checkpoints_path=model_path,epoch = epoch)

if __name__ == "__main__":
    main()
