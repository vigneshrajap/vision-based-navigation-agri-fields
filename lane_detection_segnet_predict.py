#!/usr/bin/env python
import keras_segmentation
import os

main_path = os.path.join('/home/marianne/Code/vision-based-navigation-agri-fields')
data_folder = 'Frogn_Dataset'
model_folder = 'models'

model = keras_segmentation.models.segnet.segnet(n_classes=51, input_height=360, input_width=640  )
model.load_weights(os.path.join(main_path,model_folder,'segnet_from_scatch.0')) # load json and create model

#print(os.path.join(main_path,data_folder,'images_prepped_test/frogn_10008.png'))
#print(model.summary())
out = model.predict_segmentation(
    inp=os.path.join(main_path,data_folder,'images_prepped_test/frogn_10008.png'), 
    out_fname="out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)


'''
#Run from command line:
python -m keras_segmentation predict \
 --checkpoints_path="/home/marianne/Code/vision-based-navigation-agri-fields/models/segnet_from_scatch.0" \
 --input_path="/home/marianne/Code/vision-based-navigation-agri-fields/Frogn_Dataset/images_prepped_test" \
 --output_path="."
 '''