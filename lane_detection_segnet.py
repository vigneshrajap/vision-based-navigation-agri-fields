#!/usr/bin/env python
import sys
sys.path.append('../image-segmentation-keras')
import keras_segmentation 
import os

main_path = os.path.join('.')
data_folder = 'Frogn_Dataset'
model_folder = 'models'


model = keras_segmentation.models.segnet.segnet(n_classes=3,  input_height=360, input_width=640)

model.train(
    train_images =  os.path.join(main_path, data_folder,'images_prepped_train'),
    train_annotations = os.path.join(main_path, data_folder,'annotations_prepped_train'),
    checkpoints_path = os.path.join(main_path,model_folder,'resnet50_segnet'), epochs=5
)

#model.load_weights('/content/drive/My Drive/Colab Notebooks/segnet_weights/segnet.4') # load json and create model

out = model.predict_segmentation(
    inp="~/Frogn_Dataset/images_prepped_test/frogn_10008.png",
    #checkpoints_path = "~/segnet_weights/segnet",
    out_fname="out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

#out = model.predict_multiple(
    #inp_dir="~/Frogn_Dataset/images_prepped_test/",
    #out_dir="~/predicted_outputs/"
#)
