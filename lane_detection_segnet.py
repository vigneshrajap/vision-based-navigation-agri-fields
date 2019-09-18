#!/usr/bin/env python
import keras_segmentation

model = keras_segmentation.models.segnet.segnet(n_classes=51,  input_height=360, input_width=640  )

model.train(
    train_images =  "~/Frogn_Dataset/images_prepped_train/",
    train_annotations = "~/Frogn_Dataset/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_segnet" , epochs=5
)

out = model.predict_segmentation(
    inp="~/Frogn_Dataset/images_prepped_test/frogn_10008.png", 
    out_fname="out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)
