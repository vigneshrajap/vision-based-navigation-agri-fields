#!/usr/bin/env python
import os
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../image-segmentation-keras')
from keras_segmentation.data_utils.data_loader import get_image_arr , get_segmentation_arr
from keras_segmentation.predict import predict, model_from_checkpoint_path
from keras_segmentation import metrics
#from keras_segmentation.train import find_latest_checkpoint
import numpy as np
from tqdm import tqdm 
import argparse

def evaluate( model=None , inp_images=None , annotations=None , checkpoints_path=None, epoch = None, visualize = False, output_folder = ''):
    #Finished implementation of the evaluate function in keras_segmentation.predict
    #Input: array of paths or nd arrays
    if model is None and ( not checkpoints_path is None ):
        model = model_from_checkpoint_path(checkpoints_path,epoch)
        
    ious = []
    for inp , ann   in tqdm(zip( inp_images , annotations )):
        pr = predict(model , inp )
        gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height, no_reshape=True)
        gt = gt.argmax(-1)
        iou = metrics.get_iou( gt , pr , model.n_classes )
        ious.append( iou )
        
        if visualize:
            print(iou)
            print('gt',np.min(gt),np.max(gt))
            print('pr',np.min(pr),np.max(pr))
            fig = plt.figure()
            plt.title("IOU:"+str(iou))
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
            #plt.show()
            #print(checkpoints_path, inp)
            if not output_folder:
                plt.savefig(checkpoints_path+'_IOU_'+os.path.basename(inp))
            else: 
                plt.savefig(os.path.join(output_folder,os.path.basename(checkpoints_path)+'_IOU_'+os.path.basename(inp)))
    ious = np.array( ious )
    print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
    print("Total  IoU "  ,  np.mean(ious ))

def main():
    parser = argparse.ArgumentParser(description="Run evaluation of model on a set of images and annotations.")
    parser.add_argument("--model_prefix", default = '', help = "Prefix of model filename")
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