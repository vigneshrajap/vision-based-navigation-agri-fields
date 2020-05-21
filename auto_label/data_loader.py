from tqdm import tqdm
import glob
import os
import cv2 
import numpy as np
import itertools
import random

from keras_segmentation.data_utils.data_loader import get_image_array, get_segmentation_array
from keras_segmentation.models.config import IMAGE_ORDERING
from keras_segmentation.data_utils.augmentation import augment_seg

def get_pairs_from_paths( images_path , segs_path ):
    images = glob.glob( os.path.join(images_path+"*.jpg")  ) + glob.glob( os.path.join(images_path+"*.png")  ) +  glob.glob( os.path.join(images_path+"*.jpeg")  )
    segmentations  =  glob.glob( os.path.join(segs_path+"*.png")  ) 

    segmentations_d = dict( zip(segmentations,segmentations ))

    ret = []

    for im,seg in zip(images,segmentations):
        #seg_bnme = os.path.basename(im).replace(".jpg" , ".png").replace(".jpeg" , ".png")
        #seg = os.path.join( segs_path , seg_bnme  )
        #assert ( seg in segmentations_d ),  (im + " is present in "+images_path +" but "+seg_bnme+" is not found in "+segs_path + " . Make sure annotation image are in .png"  )
        ret.append((im , seg) )

    return ret

def verify_segmentation_dataset( images_path , segs_path , n_classes ):
    
    img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

    assert len(img_seg_pairs)>0 , "Dataset looks empty or path is wrong "
    
    for im_fn , seg_fn in tqdm(img_seg_pairs) :
        img = cv2.imread( im_fn )
        seg = cv2.imread( seg_fn )
        assert ( img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1] ) , "The size of image and the annotation does not match or they are corrupt "+ im_fn + " " + seg_fn
        assert ( np.max(seg[:,:,0]) < n_classes) , "The pixel values of seg image should be from 0 to "+str(n_classes-1) + " . Found pixel value "+str(np.max(seg[:,:,0]))

    print("Dataset verified! ")


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False,
                                 augmentation_name="aug_all"):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                               augmentation_name)

            X.append(get_image_array(im, input_width,
                                     input_height, ordering=IMAGE_ORDERING))
            Y.append(get_segmentation_array(
                seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)

