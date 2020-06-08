import glob
import random
import json
import os
import six

import cv2
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

from train import find_latest_checkpoint
from keras_segmentation.data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from keras_segmentation.models.all_models import model_from_name
from keras_segmentation.models.config import IMAGE_ORDERING

#Our code
import metrics
import sys
sys.path.append('..')
from visualization_utilities import blend_color_and_image, vis_pred_vs_gt_overlay_and_separate

def model_from_checkpoint_path( checkpoints_path, epoch = None):
    #Adapted from keras_segmentation. With epoch option.
    assert ( os.path.isfile(checkpoints_path+"_config.json" ) ) , "Checkpoint not found."
    model_config = json.loads(open(  checkpoints_path+"_config.json" , "r" ).read())
    if epoch is None:
        weights_file = find_latest_checkpoint( checkpoints_path )
    else:
        weights_file = os.path.join( checkpoints_path + "." + str( epoch ) )
    assert ( not weights_file is None ) , "Checkpoint not found."
    model = model_from_name[ model_config['model_class']  ]( model_config['n_classes'] , input_height=model_config['input_height'] , input_width=model_config['input_width'] )
    print("loaded weights " , weights_file )
    model.load_weights(weights_file)
    return model

def predict_fast( model=None , inp=None, checkpoints_path = None):
    #Adapted from keras_segmentation. Without reloading model and visualization
    if model is None and ( not checkpoints_path is None ):
        model = model_from_checkpoint_path(checkpoints_path)

    assert ( not inp is None )
    assert( (type(inp) is np.ndarray ) or  isinstance( inp , six.string_types)  ) , "Inupt should be the CV image or the input file name"
    
    if isinstance( inp , six.string_types)  :
        inp = cv2.imread(inp )

    assert len(inp.shape) == 3 , "Image should be h,w,3 "

    input_width = model.input_width
    input_height = model.input_height

    x = get_image_array( inp , input_width  , input_height , ordering=IMAGE_ORDERING )
    pr = model.predict( np.array([x]) )[0]
    pr_arr = pr.reshape(( model.output_height ,  model.output_width , model.n_classes ) ).argmax( axis=2 )
    return pr_arr

def predict_verbose( model=None , inp=None, checkpoints_path = None):
    #Like predict_fast but with full prediction array (before max) as output as well
    if model is None and ( not checkpoints_path is None ):
        model = model_from_checkpoint_path(checkpoints_path)

    assert ( not inp is None )
    assert( (type(inp) is np.ndarray ) or  isinstance( inp , six.string_types)  ) , "Inupt should be the CV image or the input file name"
    
    if isinstance( inp , six.string_types)  :
        inp = cv2.imread(inp )

    assert len(inp.shape) == 3 , "Image should be h,w,3 "

    input_width = model.input_width
    input_height = model.input_height

    x = get_image_array( inp , input_width  , input_height , ordering=IMAGE_ORDERING )
    pr = model.predict( np.array([x]) )[0]
    pr_arr = pr.reshape(( model.output_height ,  model.output_width , model.n_classes ) ).argmax( axis=2 )
    softmax = pr.reshape(( model.output_height ,  model.output_width , model.n_classes ) ).max( axis=2 )
    return pr_arr, softmax, pr

def evaluate_and_visualize( model=None , inp_images=None , annotations=None , checkpoints_path=None, epoch = None, visualize = True, output_folder = '',ignore_zero_class = False):
    #Finished implementation of the evaluate function in keras_segmentation.predict (v 0.2.0) and added visualization
    #Input: array of paths or nd arrays
    if model is None and ( not checkpoints_path is None ):
        model = model_from_checkpoint_path(checkpoints_path,epoch)
    if checkpoints_path is None:
        checkpoints_path = ''

    ious = []
    mean_ious = []
    fw_ious = []
    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)
    for inp , ann   in tqdm(zip( inp_images , annotations )):
        pr, pr_softmax, pr_raw = predict_verbose(model , inp )
        t_end = time()
        
        gt = get_segmentation_array( ann , model.n_classes ,  model.output_width , model.output_height, no_reshape=True)
        gt = gt.argmax(-1)
        iou, m_iou, fw_iou = metrics.get_iou( gt , pr , model.n_classes, ignore_zero_class = ignore_zero_class)
        ious.append(iou)
        mean_ious.append(m_iou)
        fw_ious.append(fw_iou)
        if visualize:
            #fig = vis_pred_overlay(inp,pr)
            fig = vis_pred_vs_gt_overlay_and_separate(inp,pr,gt, softmax = pr_softmax)
            plt.suptitle("IoU (bg, crop, lane): {:.2f},{:.2f},{:.2f}, Mean IoU: {:.2f}, FW IoU: {:.2f}".format(iou[0],iou[1],iou[2],m_iou,fw_iou))
            if not output_folder:
                if epoch is None:
                    epoch = ''
                print(checkpoints_path, epoch, os.path.basename(inp))
                fig.savefig(checkpoints_path+epoch+'_IOU_'+os.path.basename(inp))
                print('Saving to: ',checkpoints_path+epoch+'_IOU_'+os.path.basename(inp))
            else:
                fig.savefig(os.path.join(output_folder,os.path.basename(checkpoints_path)+'_IOU_'+os.path.basename(inp)))
    class_wise_iou = np.mean(ious , axis = 0)
    mean_iou = np.mean(mean_ious)
    frequency_weighted_iou = np.mean(fw_ious)
    return class_wise_iou, mean_iou, frequency_weighted_iou
