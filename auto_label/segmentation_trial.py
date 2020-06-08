#!/usr/bin/env python
import sys
import keras_segmentation
#sys.path.append('..')
#from lane_detection_segnet_evaluate import evaluate
import os
import glob
from time import strftime
from train import train
from types import MethodType

#--- User input
main_path = os.path.join('.')
data_folder = 'output/prepped_data'
model_folder = 'models'
model_name = 'manuallabel_L1_fewimages'+'_'+strftime("%Y-%d-%m-%H%M") 
prefix = '20191010_L1_N_14' #'20191010_L1_N' #empty prefix = train on all
train_annotations_folder = 'annotations_manual' #'annotations_straight'
val_annotations_folder = 'annotations_manual'
train_images_folder = 'images_manual'
val_images_folder = 'images_manual'

#--- Setup
train_images_path =  os.path.join(main_path, data_folder,'train',train_images_folder, prefix + '*')
train_annotations_path = os.path.join(main_path, data_folder,'train',train_annotations_folder, prefix + '*')
val_images_path = os.path.join(main_path, data_folder,'val',train_images_folder, prefix + '*')
val_annotations_path = os.path.join(main_path, data_folder,'val',val_annotations_folder, prefix + '*')

model_path = os.path.join(main_path,model_folder,model_name)
os.makedirs(model_path, exist_ok = True)

model = keras_segmentation.models.segnet.segnet(n_classes=3,  input_height=360, input_width=640)

#Training
print('---------------------Using training data from ', os.path.join(main_path, data_folder,'train/images/'+prefix+'*'))

model.train = MethodType(train,model) #Redefine training function from keras_segmentation

model.train(
    train_images =  train_images_path,
    train_annotations = train_annotations_path,
    validate = True,
    val_images = val_images_path,
    val_annotations = val_annotations_path,
    checkpoints_path = model_path, 
    steps_per_epoch = None, #determined inside function
    epochs=20,
    logging = True,
    loss_name = 'categorical_crossentropy',#'dice',
    ignore_zero_class=True,
)
