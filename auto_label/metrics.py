import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


EPS = 1e-12

#Adapted from keras_segmentation library
'''
def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise

def get_iou_ignore_zero_class(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(1,n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise
'''
#fixme: implement F1 score

def get_iou(gt, pr, n_classes, ignore_zero_class=False):
    #From keras_segmentation.predict.evaluate
    tp = np.ones(n_classes)*np.nan
    fp = np.ones(n_classes)*np.nan
    fn = np.ones(n_classes)*np.nan
    n_pixels = np.zeros(n_classes)
    start_class_ind = 0

    if ignore_zero_class:
        zeros_ind = gt > 0
        gt = gt[zeros_ind]
        pr = pr[zeros_ind]
        start_class_ind = 1

    for cl_i in range(start_class_ind,n_classes): #skip 0 if ignore zero class
        tp[cl_i] = np.sum((pr == cl_i) * (gt == cl_i))
        fp[cl_i] = np.sum((pr == cl_i) * ((gt != cl_i)))
        fn[cl_i] = np.sum((pr != cl_i) * ((gt == cl_i)))
        n_pixels[cl_i] = np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.nansum(n_pixels)
    frequency_weighted_IoU = np.nansum(cl_wise_score*n_pixels_norm)
    mean_IoU = np.nanmean(cl_wise_score)

    return cl_wise_score, mean_IoU, frequency_weighted_IoU

#Compatible with Keras
def masked_IoU_3class(gt,pr):
    #fixme implement keras tensor functions
    return get_iou(gt,pr,3,ignore_zero_class = True)

## Loss functions
def dice(y_true, y_pred, smooth=1):
    print("**********************Using dice loss")
    #From https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice