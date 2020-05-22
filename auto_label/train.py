import json
import sys
from keras_segmentation.models.all_models import model_from_name
from keras_segmentation.train import find_latest_checkpoint,masked_categorical_crossentropy,CheckpointsCallback
from tensorflow.keras.metrics import MeanIoU
import glob
import os
import six
from keras.callbacks import Callback
from keras import callbacks as k_callbacks

from tqdm import tqdm
import cv2 
import numpy as np

from metrics import dice, masked_IoU_3class
from data_loader import verify_segmentation_dataset, get_pairs_from_paths, image_segmentation_generator

def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adadelta',
          loss_name = 'categorical_crossentropy',
          logging = False,
          do_augment=False,
          augmentation_name="aug_all"):

    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None


    if loss_name == 'categorical_crossentropy':
        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = masked_categorical_crossentropy
    if loss_name == 'dice':
        loss_k = dice

    model.compile(loss=loss_k,
                optimizer=optimizer_name,
                metrics=['accuracy', dice])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert (verified, "Verification of training set failed")
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert (verified, "Verification of validation set failed")

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name)
    if steps_per_epoch is None: steps_per_epoch = len(get_pairs_from_paths(train_images, train_annotations))

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)
        if val_steps_per_epoch is None: val_steps_per_epoch = len(get_pairs_from_paths(val_images, val_annotations))

    callbacks = [CheckpointsCallback(checkpoints_path)]
    if logging:
       tbCallBack = k_callbacks.TensorBoard(histogram_freq=0,log_dir=checkpoints_path)
       callbacks.append(tbCallBack)

    model.fit_generator(train_gen, steps_per_epoch, validation_data = val_gen, validation_steps = val_steps_per_epoch, epochs=epochs ,callbacks=callbacks ) #temporary fix, breaks the start_epoch functionality

