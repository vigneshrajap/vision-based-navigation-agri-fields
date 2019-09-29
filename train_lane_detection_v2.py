#!/usr/bin/env python
import keras_segmentation
import os

data_folder = 'Frogn_Dataset'
model_folder = 'models'

model = keras_segmentation.models.segnet.segnet(n_classes=4,  input_height=360, input_width=640  )

model.summary()

model.train(
    train_images =  os.path.join(data_folder,'images_prepped_train'),
    train_annotations = os.path.join(data_folder,'annotations_prepped_train'),
    checkpoints_path = os.path.join(model_folder,'segnet_from_scratch_v2'), epochs=50
)

