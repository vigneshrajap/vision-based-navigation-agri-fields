import matplotlib.pyplot as plt 
import os
import sys
import glob
import cv2
import numpy as np
sys.path.insert(1, '../') #use our own version of keras image segmentation
from visualization_utilities import vis_pred_vs_gt_separate,vis_pred_vs_gt_overlay

def get_segmentation_arr( path , nClasses ,  width , height , no_reshape=False ):

	seg_labels = np.zeros((  height , width  , nClasses ))
		
	if type( path ) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	img = cv2.resize(img, ( width , height ) , interpolation=cv2.INTER_NEAREST )
	img = img[:, : , 0]

	for c in range(nClasses):
		seg_labels[: , : , c ] = (img == c ).astype(int)


	
	if no_reshape:
		return seg_labels

	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels

#Compare automatic field mask to manual labels
if __name__ == "__main__":
    #Make image mask for a folder of images and their robot position data
    #Setup
    visualize = False
    dataset_dir = os.path.join('../Frogn_Dataset')
    label_dir = os.path.join(dataset_dir,'annotations_prepped_train')
    image_dir = os.path.join(dataset_dir,'calibration_selection')
    auto_label_dir = os.path.join('output/calibration/arrays')
    output_dir = os.path.join('output/calibration/compare')

    for auto_ann_file in glob.iglob(auto_label_dir+'/*'):
        #Read image , automatic label and manual label
        filename = os.path.splitext(os.path.basename(auto_ann_file))[0]
        auto_ann = np.load(os.path.join(auto_label_dir,filename + '.npy')).astype(int)
        #auto_ann = get_segmentation_arr(os.path.join(auto_label_dir,filename + '.png'), 3 ,  auto_ann_im.shape[1] , auto_ann_im.shape[0] , no_reshape=True )
        #input_image = plt.imread(os.path.join(image_dir,filename + '.png'))
        inp_im_file = os.path.join(image_dir,filename + '.png')
        manual_ann = plt.imread(os.path.join(label_dir,filename + '.png'))

        m_ann = get_segmentation_arr(os.path.join(label_dir,filename + '.png'), 3 ,  auto_ann.shape[1] , auto_ann.shape[0] , no_reshape=True )
        m_ann = m_ann.argmax(-1)
        #Visualize
        fig1 = vis_pred_vs_gt_separate(os.path.join(image_dir,filename + '.png'),auto_ann, m_ann)
        fig2 = vis_pred_vs_gt_overlay(os.path.join(image_dir,filename + '.png'), auto_ann, m_ann)
        if visualize:
            plt.show()
        else:
            fig2.savefig(os.path.join(output_dir,filename)+'.png')