#!/usr/bin/env python
import sys
sys.path.insert(1, '../image-segmentation-keras') #add 
import keras_segmentation
import os

setup_name = 'resnet_3class_moreBG'

data_folder = 'Frogn_Dataset'
model_folder = 'models'

model = keras_segmentation.models.segnet.resnet50_segnet(n_classes=3,  input_height=352, input_width=640  )

model.summary()

model.train(
    train_images =  os.path.join(data_folder,'images_prepped_train'),
    train_annotations = os.path.join(data_folder,'annotations_prepped_train'),
    checkpoints_path = os.path.join(model_folder,setup_name),
    epochs=100,
    auto_resume_checkpoint = False,
    logging = True
)

