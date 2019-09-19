#!/usr/bin/env python
import keras_segmentation
import os

main_path = os.path.join('/home/marianne/Code/vision-based-navigation-agri-fields')
data_folder = 'Frogn_Dataset'
model_folder = 'models'

model = keras_segmentation.models.segnet.segnet(n_classes=51,  input_height=360, input_width=640  )

model.train(
    train_images =  os.path.join(main_path, data_folder,'images_prepped_train'),
    train_annotations = os.path.join(main_path, data_folder,'annotations_prepped_train'),
    checkpoints_path = os.path.join(main_path,model_folder,'segnet_from_scatch'), epochs=5
)

out = model.predict_segmentation(
    inp="~/Frogn_Dataset/images_prepped_test/frogn_10008.png", 
    out_fname="out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)
