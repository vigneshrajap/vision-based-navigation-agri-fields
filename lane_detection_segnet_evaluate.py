#!/usr/bin/env python
import os
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../image-segmentation-keras') #use our own version of keras image segmentation
from keras_segmentation.predict import predict, predict_fast,model_from_checkpoint_path
from keras_segmentation.data_utils.data_loader import get_segmentation_arr
#from keras_segmentation.predict import predict, predict_fast, model_from_checkpoint_path
from keras_segmentation import metrics
from keras_segmentation.train import find_latest_checkpoint
import numpy as np
from tqdm import tqdm
import argparse
import time
from PIL import Image

def vis_pred_vs_gt_overlay(inp, pr, gt):
    #Visualize segmentation prediction and false positives/negatives
    
    #NB 3-class problem with background not handled properly
    input_image = Image.open(inp)

    #Make overlay image
    mask = np.zeros((gt.shape[0],gt.shape[1],3),dtype='uint8')
    im_resized = np.array(input_image.resize((mask.shape[1],mask.shape[0])))
    
    error_mask = pr-gt
    mask[:,:,0] = 255*np.uint8(error_mask > 0) #False positives for class 1
    mask[:,:,2] = 255*np.uint8(error_mask < 0) #False negatives for class 1
    mask[:,:,1] = 255*(gt-1)*np.uint8(error_mask == 0) #class 1 in ground truth encoded green
    
    alpha = 0.4
    vis_img = np.uint8((1-alpha)*im_resized + alpha*mask)
    
    fig = plt.figure(111)
    plt.imshow(vis_img)
    
    return fig

def vis_pred_vs_gt_separately(inp,pr,gt):
              
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(gt-pr)
    #ax1.colorbar()
    ax1.title.set_text("Difference GT-pred")

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(gt)
    ax2.title.set_text('GT')

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(pr)
    ax3.title.set_text('pred')

    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(plt.imread(inp))
    ax4.title.set_text('Input image')
    
    return fig


def evaluate( model=None , inp_images=None , annotations=None , checkpoints_path=None, epoch = None, visualize = False, output_folder = ''):
    #Finished implementation of the evaluate function in keras_segmentation.predict
    #Input: array of paths or nd arrays
    if model is None and ( not checkpoints_path is None ):
        model = model_from_checkpoint_path(checkpoints_path,epoch)

    ious = []
    pred_time = []
    for inp , ann   in tqdm(zip( inp_images , annotations )):
        t_start = time.time()
        pr = predict_fast(model , inp )
        t_end = time.time()
        pred_time.append(t_end-t_start)
        #print('Prediction time: ', pred_time)
        
        gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height, no_reshape=True)
        gt = gt.argmax(-1)
        iou = metrics.get_iou( gt , pr , model.n_classes )
        ious.append( iou )
        np.save('gt_array',gt)
        np.save('pr_array',pr)

        if visualize:
            fig = vis_pred_vs_gt_overlay(inp,pr,gt)
            plt.title("Predicted mask and errors. " "IOU (bg, crop, lane):"+str(iou))
            if not output_folder:
                if epoch is None:
                    epoch = ''
                print(checkpoints_path, epoch, os.path.basename(inp))
                fig.savefig(checkpoints_path+epoch+'_IOU_'+os.path.basename(inp))
                print('Saving to: ',checkpoints_path+epoch+'_IOU_'+os.path.basename(inp))
            else:
                fig.savefig(os.path.join(output_folder,os.path.basename(checkpoints_path)+'_IOU_'+os.path.basename(inp)))
    print ious
    ious = np.array( ious )
    print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
    print("Total  IoU "  ,  np.mean(ious ))
    print("Mean prediction time:", np.mean(pred_time))

def main():
    parser = argparse.ArgumentParser(description="Run evaluation of model on a set of images and annotations.")
    parser.add_argument("--model_prefix", default = 'resnet50_segnet', help = "Prefix of model filename")
    parser.add_argument("--epoch", default = None, help = "Checkpoint epoch number")
    parser.add_argument("--data_folder", default = 'Frogn_Dataset', help = "Relative path of data folder")
    parser.add_argument("--model_folder", default = 'models', help = "Relative path of model folder")
    parser.add_argument("--images_folder",default = 'images_prepped_test', help = "Name of image folder")
    parser.add_argument("--annotations_folder",default = 'annotations_prepped_test', help = "Name of annotations folder")
    parser.add_argument("--visualize",default = False, help = "Turn visualization on/off")
    parser.add_argument("--output_folder",default = '', help = "Ouput folder for figures.")
    args = parser.parse_args()

    model_path = os.path.join(args.model_folder,args.model_prefix)
    im_files = glob.glob(os.path.join(args.data_folder,args.images_folder, '*.png'))
    ann_files = glob.glob(os.path.join(args.data_folder,args.annotations_folder , '*.png'))

    print('Running evaluation on ',len(im_files),'images, and',len(ann_files),'annotations:')
    evaluate(model=None , inp_images= im_files , annotations= ann_files, checkpoints_path=model_path,epoch = args.epoch, visualize = args.visualize, output_folder = args.output_folder)

if __name__ == "__main__":
    main()
