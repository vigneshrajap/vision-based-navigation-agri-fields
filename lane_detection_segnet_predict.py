#!/usr/bin/env python
import keras_segmentation
import os
import glob
import argparse

'''
out = model.predict_segmentation(
    inp=os.path.join(data_folder,'images_prepped_test/frogn_10008.png'), 
    out_fname="out.png"
)
print('out shape',out.shape)
print(out)

#Run from command line:
python -m keras_segmentation predict \
 --checkpoints_path="/home/marianne/Code/vision-based-navigation-agri-fields/models/segnet_from_scatch.0" \
 --input_path="/home/marianne/Code/vision-based-navigation-agri-fields/Frogn_Dataset/images_prepped_test" \
 --output_path="."
'''
#out = keras_segmentation.predict.predict_multiple( model=None , inps=None , inp_dir=None, out_dir=None , checkpoints_path=None  )
def main():
    parser = argparse.ArgumentParser(description="Run evaluation of model on a set of images and annotations.")
    parser.add_argument("--model_prefix", default = '', help = "Prefix of model filename")
    parser.add_argument("--output_folder",default = '', help = "Relative path output folder")
    parser.add_argument("--data_folder", default = 'Frogn_Dataset', help = "Relative path of data folder")
    parser.add_argument("--model_folder", default = 'models', help = "Relative path of model folder")
    parser.add_argument("--images_folder",default = 'images_prepped_test', help = "Name of image folder")

    args = parser.parse_args()
    
    model_path = os.path.join(args.model_folder,args.model_prefix)
    im_dir = os.path.join(args.data_folder,args.images_folder)
    output_folder = args.output_folder

    keras_segmentation.predict.predict_multiple(model=None , inps=None, inp_dir = im_dir,  out_dir = output_folder, checkpoints_path=model_path)

if __name__ == "__main__":
    main()