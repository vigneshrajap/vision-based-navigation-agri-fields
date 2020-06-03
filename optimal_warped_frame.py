#!/usr/bin/env python
import cv2
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import imutils
import sliding_window_approach
import scipy.signal as signal
from sklearn.cluster import KMeans

#from folder import file

import pixel_to_vector

k = pixel_to_vector.conv_pixel_to_vector()

print k.crop_ratio

### Calcuate the Variance in the image
# def winVar(img, wlen):
#   wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
#     borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
#   return wsqrmean - wmean*wmean

img_file = expanduser("~/dummy_folder/20191010_L1_N_predictions/20191010_L1_N_0055_crop_pred.png")
# dummy_folder/lane_predictions/frogn_10000_lane_pred.png")
img = cv2.imread(img_file, 0)
#rows, cols = img.shape
#img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# print img.shape[:2]

crop_ratio = 0.8
rheight, rwidth = img.shape[:2]
roi_img = img[int(crop_ratio*rheight):rheight,0:rwidth]
#dst_size = roi_img.shape[:2]
dst_size = roi_img.shape[1],roi_img.shape[1]
#dst_size = img.shape[:2]

dst=np.float32([(0,0), (1,0), (0,1), (1,1)])
src=np.float32([(-0.1,0), (1.1,0), (-0.1,1), (1.1,1)])
#src=np.float32([(0,0.3), (1,0.3), (-0.4,0.8), (1.4,0.8)])

roi_img, M  = sliding_window_approach.perspective_warp(roi_img, dst_size, src, dst)

coldata = np.sum(roi_img, axis=0) # Sum the columns of warped image to determine peaks

modifiedCenters = signal.find_peaks(coldata, height=60000,  distance=100) #, #, np.arange(1,100), noise_perc=0.1

# # Find white pixels
# whitePixels = np.argwhere(roi_img == 255)

# kmeans = KMeans(n_clusters=2, random_state=0, n_init=3, max_iter=150).fit(whitePixels)

# print modifiedCenters #,  kmeans.cluster_centers_[0][1]
# roi_img = cv2.cvtColor(roi_img,cv2.COLOR_GRAY2RGB)
# print roi_img.shape[:2]
# for mc_in in range(len(modifiedCenters[0])):
# 	cv2.circle(roi_img, (modifiedCenters[0][mc_in],roi_img.shape[0]-25),
# 	                                          0, (0,0,255), thickness=25, lineType=8, shift=0)

	# cv2.circle(roi_img, (int(kmeans.cluster_centers_[mc_in][1]),int(kmeans.cluster_centers_[mc_in][0])),
	#                                           0, (255,0,255), thickness=25, lineType=8, shift=0)
#cv2.imwrite('/home/vignesh/warped_img.png', roi_img)

# edges = cv2.Canny(warped_img,50,150,apertureSize = 3)
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(warped_img,1,np.pi/180,100,minLineLength,maxLineGap)
# warped_img_c = cv2.cvtColor(warped_img,cv2.COLOR_GRAY2RGB)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(warped_img_c,(x1,y1),(x2,y2),(0,255,0),3)
#print M
#cv2.imwrite('frogn_10000_lane_sw_warp_img.png', warped_img)
# cv2.imshow(title_window, warped_img)
# # Wait until user press some key
# cv2.waitKey()

########### Skew Angle Estimation ##############

#   img_file = expanduser("~/dummy_folder/warped_images/frogn_10000_lane_sw_warp_img.png")
#   img = cv2.imread(img_file, 0)

# load the image from disk

# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
# var_arr = []
# ang_arr = []
# min_angle = -10
# max_angle = 10
# increment = 1
# for angle in np.arange(min_angle, max_angle, increment):
# 	rotated = imutils.rotate_bound(warped_img, angle)
# 	img_col_sum = rotated.sum(axis=0)
# 	#var = winVar(rotated, 3)
# 	#var_arr.append([angle,np.var(img_col_sum)])
# 	var_arr.append(np.var(img_col_sum))
# 	ang_arr.append(angle)
#
#
# angle_index = var_arr.index(max(var_arr))
# final_skew_angle = ang_arr[int(angle_index)]
#
# (h, w) = warped_img.shape[:2]
# (cX, cY) = (w // 2, h // 2)
# scale = 1.0
# M = cv2.getRotationMatrix2D((cX, cY), final_skew_angle, scale)
# warped_img = cv2.warpAffine(warped_img, M, (h, w))
#
# 	#angle, max(img_col_sum)
# print max(var_arr), ang_arr[int(angle_index)]
#     #cv2.imwrite(str(angle)+'.png', rotated)
#
# cv2.imshow("Rotated (Correct)", warped_img)
# cv2.waitKey(0)

# #print var_arr.index(max(var_arr))
# angle_r = ang_arr[var_arr.index(max(var_arr))]
# rotated_r = imutils.rotate(warped_img, angle_r)
#cv2.imwrite(str(angle_r)+'.png', warped_img)

#
# # plt.plot(img_col_sum)
# # plt.show()
#
# #print  max(img_col_sum) #len(img_col_sum), img_col_sum
