#!/usr/bin/env python
import cv2
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import imutils
import sliding_window_approach

### Calcuate the Variance in the image
# def winVar(img, wlen):
#   wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
#     borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
#   return wsqrmean - wmean*wmean

img_file = expanduser("~/Third_Paper/Datasets/Frogn_Fields/Frogn_004_crop_predictions/frogn_40270_crop_pred_img.png")
# dummy_folder/lane_predictions/frogn_10000_lane_pred.png")
img = cv2.imread(img_file, 0)

#dst_size = img.shape[:2]
dst_size = (img.shape[1],img.shape[1])

dst=np.float32([(0,0), (1,0), (0,1), (1,1)])
#src=np.float32([(0.2,0.2), (0.8,0.2), (0.2,1.2), (0.8,1.2)])
src=np.float32([(0,0.5), (1,0.5), (0,1), (1,1)])

warped_img, M  = sliding_window_approach.perspective_warp(img, dst_size, src, dst)

edges = cv2.Canny(warped_img,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(warped_img,1,np.pi/180,100,minLineLength,maxLineGap)
warped_img_c = cv2.cvtColor(warped_img,cv2.COLOR_GRAY2RGB)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(warped_img_c,(x1,y1),(x2,y2),(0,255,0),3)
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
# for angle in np.arange(-5, 5, 0.25):
# 	rotated = imutils.rotate(warped_img, angle)
# 	img_col_sum = rotated.sum(axis=0)
# 	#var = winVar(rotated, 3)
# 	#var_arr.append([angle,np.var(img_col_sum)])
# 	var_arr.append(np.var(img_col_sum))
# 	ang_arr.append(angle)
#
# 	#angle, max(img_col_sum)
# 	# print angle, max(img_col_sum)
# 	# cv2.imwrite(str(angle)+'.png', rotated)
# 	# cv2.imshow("Rotated (Correct)", rotated)
# 	# cv2.waitKey(0)
#
# #print var_arr.index(max(var_arr))
# angle_r = ang_arr[var_arr.index(max(var_arr))]
# rotated_r = imutils.rotate(warped_img, angle_r)
#cv2.imwrite(str(angle_r)+'.png', warped_img)
cv2.imwrite('warped_img.png', warped_img_c)

#
# # plt.plot(img_col_sum)
# # plt.show()
#
# #print  max(img_col_sum) #len(img_col_sum), img_col_sum
