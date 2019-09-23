#!/usr/bin/env python
import keras_segmentation
import os
import matplotlib.pyplot as plt
from keras_segmentation.data_utils.data_loader import get_image_arr , get_segmentation_arr
from keras_segmentation.predict import predict
from keras_segmentation import metrics
import numpy as np
import tqdm

def evaluate( model=None , inp_images=None , annotations=None , checkpoints_path=None ):
    ious = []
    for inp , ann   in zip( inp_images , annotations ):
        pr = predict(model , inp )
        gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height, no_reshape=True)
        gt = gt.argmax(-1)
        iou = metrics.get_iou( gt , pr , model.n_classes )
        ious.append( iou )
    ious = np.array( ious )
    print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
    print("Total  IoU "  ,  np.mean(ious ))


data_folder = 'Frogn_Dataset'
model_folder = 'models'
pred_folder = 'predictions'

model_path = os.path.join(model_folder,'segnet_from_scratch_v1.4')

model = keras_segmentation.models.segnet.segnet(n_classes=3, input_height=360, input_width=640  )
model.load_weights(model_path) # load json and create model

im_file = os.path.join(data_folder,'images_prepped_test/frogn_10008.png')
im = plt.imread(im_file)

gt_file =os.path.join(data_folder,'annotations_prepped_test/frogn_10008.png')
gt = plt.imread(gt_file)

#try built-in function
evaluate(model=model , inp_images=[im_file] , annotations=[gt_file] , checkpoints_path=None )

'''
output_width = model.output_width
output_height  = model.output_height

gt_file =os.path.join(data_folder,'annotations_prepped_test/frogn_10008.png')
gt = plt.imread(gt_file)

im_file = os.path.join(data_folder,'images_prepped_test/frogn_10008.png')
im = plt.imread(im_file)
pr = keras_segmentation.predict.predict(model = model,inp=im_file,out_fname = 'tmp_out.png')
gt = cv2.resize(cv2.transpose(gt)  , (output_width , output_height))

n_classes  = 3
iou = keras_segmentation.metrics.get_iou( gt , pr , n_classes)
print(iou)
'''

'''
#Run from command line:
python -m keras_segmentation predict \
 --checkpoints_path="/home/marianne/Code/vision-based-navigation-agri-fields/models/segnet_from_scatch.0" \
 --input_path="/home/marianne/Code/vision-based-navigation-agri-fields/Frogn_Dataset/images_prepped_test" \
 --output_path="."
 '''
