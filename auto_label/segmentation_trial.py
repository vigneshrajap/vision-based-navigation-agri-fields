#!/usr/bin/env python
import sys
sys.path.append('../../image-segmentation-keras')
import keras_segmentation 
sys.path.append('..')
from lane_detection_segnet_evaluate import evaluate
import os
import glob

main_path = os.path.join('.')
data_folder = 'output/prepped_data'
model_folder = 'models'

model = keras_segmentation.models.segnet.segnet(n_classes=3,  input_height=360, input_width=640)

#Training
model.train(
    train_images =  os.path.join(main_path, data_folder,'train/images'),
    train_annotations = os.path.join(main_path, data_folder,'train/annotations'),
    checkpoints_path = os.path.join(main_path,model_folder,'autolabel_L1_N_L3_S_tmp1'), epochs=1
)

#Quick testing
test_images = glob.glob(os.path.join(data_folder,'test/images/*'))
test_annotations = glob.glob(os.path.join(data_folder,'test/annotations/*'))
evaluate(model=model , inp_images=test_images, annotations=test_annotations , checkpoints_path=None, epoch = None, visualize = True, output_folder = 'output/segmentation')