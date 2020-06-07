#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import glob
import os.path as osp
import math

total_time = 0
# input_dir = ("/home/vignesh/dummy_folder/test_cases/inclined_terrains/") # HOUGH
matching_score = []
test_dir = str('/home/vignesh/dummy_folder/test_cases/tiny_plants/')
input_dir = (test_dir+"lanes_test/")  # HOUGH
src = np.float32([(0,0), (1,0), (-0.25, 0.9), (1.25, 0.9)])
dst = np.float32([(0,0), (1,0), (0,1), (1,1)])

def perspective_warp(img, dst_size, src, dst): # Choose the four vertices
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size) # For destination points

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)

    return warped, M

def hough_function(gray):
    minLinLength = 265
    maxLineGap = 20
    thres = 250

    lines = cv2.HoughLinesP(gray,1, np.pi/180, thres, minLinLength, maxLineGap)
    midLane = []
    total_x1 = []
    total_y1 = []
    total_x2 = []
    total_y2 = []
    gray_rgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

    for i in range(1,len(lines)):
       for x1, y1, x2, y2 in lines[i]:
          # l_a = abs(np.arctan2(y2-y1,x2-x1)-1.57)
          #if(abs(l_a)<0.1):
          if(abs(y1 - y2)<30):
               continue # Reject the vertical lines

          cv2.line(gray_rgb,(x1,y1),(x2,y2),(255,0,0),2)

          if (y2>y1):
	          total_x1.append(x1)
	          total_y1.append(y1)

	          total_x2.append(x2)
	          total_y2.append(y2)
          else:
	          total_x1.append(x2)
	          total_y1.append(y2)

	          total_x2.append(x1)
	          total_y2.append(y1)

          # np.sum(total_y1)/len(total_y1)
    cv2.line(gray_rgb,(int(np.mean(total_x1)),int(np.mean(total_y1))),(int(np.mean(total_x2)),int(np.mean(total_y2))),(255,0,255),4)

    lenAB = math.sqrt(math.pow(np.mean(total_x1) - np.mean(total_x2), 2.0) + pow(np.mean(total_y1) - np.mean(total_y2), 2.0))
    length = gray_rgb.shape[1]-int(np.mean(total_y1))

    C_x = np.mean(total_x2) + (np.mean(total_x2) - np.mean(total_x1)) / lenAB * length
    C_y = np.mean(total_y2) + (np.mean(total_y2) - np.mean(total_y1)) / lenAB * length

    D_x = np.mean(total_x1) + (np.mean(total_x1) - np.mean(total_x2)) / lenAB * length
    D_y = np.mean(total_y1) + (np.mean(total_y1) - np.mean(total_y2)) / lenAB * length

    cv2.line(gray_rgb,(int(np.mean(total_x1)),int(np.mean(total_y1))),(int(C_x),int(C_y)),(0,0,255),2)
    cv2.line(gray_rgb,(int(np.mean(total_x1)),int(np.mean(total_y1))),(int(D_x),int(D_y)),(0,0,255),2)
    # print C_x, C_y, D_x, D_y
    return gray_rgb, C_x, C_y, D_x, D_y

############################ HOUGH transform ############################
for label_file in glob.glob(osp.join(input_dir, '*.png')):
        print(label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            img = cv2.imread(label_file)
            #img = cv2.resize(img,None,fx=0.75,fy=0.75)
            img = cv2.medianBlur(img, 15)

    	    # Read Images
    	    #img = mpimg.imread("/home/vignesh/dummy_folder/test_cases/20191010_L1_N_0000_crop_pred.png", cv2.CV_8UC1)

            crop_ratio = 0.3

            # Load image, convert to grayscale, threshold and find contours
            # img = cv2.imread("/home/vignesh/dummy_folder/20191010_L1_N_1498_crop_pred.png")
            rheight, rwidth = img.shape[:2]

            roi_img = img[int(crop_ratio*rheight):rheight,0:rwidth]

    	    # img = cv2.resize(img,None,fx=0.75,fy=0.75)
            # gray = cv2.Canny(img, 50, 150)

    	    gray = cv2.cvtColor(roi_img,cv2.COLOR_BGR2GRAY)
            dst_size = gray.shape[:2]

            warp_img, M_t  = perspective_warp(gray, (dst_size[0], dst_size[1]), src, dst) # Perspective warp

            gray_rgb, C_x, C_y, D_x, D_y = hough_function(warp_img)

            invwarp_img, M_tinv  = perspective_warp(gray_rgb, (dst_size[1], dst_size[0]), dst, src) # Perspective warp

            point_wp = np.array([C_x, C_y, 1])
            peakidx_i = np.matmul(M_tinv, point_wp)
            peakidx_in = np.array([peakidx_i[0]/peakidx_i[2],peakidx_i[1]/peakidx_i[2]]) # divide by Z point
            peakidx_in = peakidx_in.astype(int)

            point_wp1 = np.array([D_x, D_y, 1])
            peakidx_i1 = np.matmul(M_tinv, point_wp1)
            peakidx_in1 = np.array([peakidx_i1[0]/peakidx_i1[2],peakidx_i1[1]/peakidx_i1[2]]) # divide by Z point
            peakidx_in1 = peakidx_in1.astype(int)
            cv2.line(invwarp_img,(int(peakidx_in[0]),int(gray.shape[0])),(int(peakidx_in1[0]),0),(0,0,255),2)

            hough_pts = []
            for ind in range(gray.shape[0]):

                pts = peakidx_in[0]*(float(ind)/float(gray.shape[1])) + (1-(float(ind)/float(gray.shape[1])))*peakidx_in1[0]
                hough_pts.append(int(pts))

            #print peakidx_in, peakidx_in1, gray.shape[0]

            # invwarp_img = cv2.inRange(invwarp_img, (0, 0, 0), (0, 0, 255))
            ### Ground Truth ###
            import csv
            gt_row_x = []
            gt_row_y = []
            # rheight1, rwidth1 = gray.shape[:2]

            base_name = osp.splitext(base)[0][0:18]

            # with open('/home/vignesh/dummy_folder/test_cases/inclined_terrains/ground_truth/'+os.path.splitext(self.base)[0][0:11]+'.csv') as csvfile:
            if (osp.splitext(base)[0][11] != str('_')):
               base_name = osp.splitext(base)[0][0:19]

            with open(test_dir+'ground_truth/'+base_name+'.csv') as csvfile: #inclined_terrains #larger_plants
            # with open('/home/vignesh/dummy_folder/test_cases/results/'+base_name+'.csv') as csvfile:
                 spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
                 for row in spamreader:
                     gt_row_x.append(row[1])
                     gt_row_y.append(row[2])

            gt_x = []
            gt_y = []
            for gt_in in range(len(gt_row_x)):
                 gt_x.append(int(gt_row_x[gt_in]))
                 gt_y.append(float(gt_row_y[gt_in])-rheight*crop_ratio)
                 cv2.circle(gray_rgb, (int(gt_row_x[gt_in]),int(float(gt_row_y[gt_in])-rheight*crop_ratio)), 0, (255,0,255), thickness=15, lineType=8, shift=0)

            # Fit a first order straight line / second order polynomial
            fit_l = np.polyfit(gt_y, gt_x, 1, full=True)
            fit_p = np.polyfit(gt_y, gt_x, 2, full=True)
            ploty_gt = np.linspace(0, gray.shape[0]-1, gray.shape[0])

            # Generate x and y values for plotting
            if (np.argmin([fit_l[1], fit_p[1]])==0):
               fit_gt = fit_l[0][0]*ploty_gt + fit_l[0][1]
            else:
               fit_gt = fit_p[0][0]*ploty_gt**2 + fit_p[0][1]*ploty_gt + fit_p[0][2]

            pts_left_gt = np.array([np.transpose(np.vstack([fit_gt, ploty_gt]))])
            pts_left_gt = pts_left_gt[0].astype(int)
            cv2.polylines(gray_rgb, [pts_left_gt], 0, (255,255,0), thickness=5, lineType=8, shift=0)
            ### Ground Truth ###

            ### Evaluation ###
            scale = 1
            ms_temp = 0
            nwindows = 10
            crop_row_spacing = 140
            strip_height = np.int(gray.shape[0]/nwindows)

            for mid_in in range(len(pts_left_gt)):
               # print pts_left[mid_in][0]-pts_left1[mid_in][0]
               ms = abs(pts_left_gt[mid_in][0]-hough_pts[mid_in])
               #print pts_left_gt[mid_in][0], hough_pts[mid_in]

               ms_lite = abs(np.float(ms)/(crop_row_spacing*scale)) #/np.float(self.image.shape[1]-100))

               ms_temp = ms_temp + ms_lite
               if ((mid_in%strip_height)==0):
                   scale = scale - 0.075

            ms_norm = ms_temp/len(pts_left_gt)
            curr_score = 1 - math.pow(ms_norm,2)
            if (curr_score < 0) or (curr_score > 1):
                curr_score = 0

            matching_score.append(curr_score)   #1 - (math.pow(ms_norm,2)/crop_row_spacing)) #len(pts_left1)

            print curr_score

            cv2.imwrite(test_dir+'/Hough/'+str(base)+'.png', invwarp_img)

            # warp_ratio = 0.2
    	    # t_start = time.time()
            # # rheight, rwidth = gray.shape[:2]
            # # crop_img = gray[int(warp_ratio*rheight):rheight,0:rwidth]
            #
            # warp_img, M_t  = perspective_warp(gray, (dst_size[1], dst_size[1]), src, dst) # Perspective warp
            #
            # warp_img, C_x, C_y, D_x, D_y = hough_function(warp_img)
            #

            # invwarp_img, M_tinv  = perspective_warp(gray, (dst_size[1], dst_size[1]), dst, src) # Perspective warp
            # point_wp = np.array([C_x, 0, 1])
            # peakidx_i = np.matmul(M_tinv, point_wp)
            # peakidx_in = np.array([peakidx_i[0]/peakidx_i[2],peakidx_i[1]/peakidx_i[2]]) # divide by Z point
            # peakidx_in = peakidx_in.astype(int)
            #
            # point_wp1 = np.array([D_x, gray.shape[0], 1])
            # peakidx_i1 = np.matmul(M_tinv, point_wp1)
            # peakidx_in1 = np.array([peakidx_i1[0]/peakidx_i1[2],peakidx_i1[1]/peakidx_i1[2]]) # divide by Z point
            # peakidx_in1 = peakidx_in1.astype(int)
            #
            # # self.modifiedCenters[mc_in] = peakidx_in
            #
            # print hough_pts
            # print C_x, D_x, peakidx_in, peakidx_in1
            # cv2.line(invwarp_img,(int(peakidx_in[0]),int(0)),(int(peakidx_in[1]),gray.shape[0]),(0,0,255),2)

            # #edges = cv2.Canny(gray,50,150,apertureSize = 3)
            #
            # # gray_rgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            # #
            # # lines = cv2.HoughLines(gray,1,np.pi/180,200)
            # # print len(lines[0])
            # # for rho,theta in lines[0]:
            # #     a = np.cos(theta)
            # #     b = np.sin(theta)
            # #     x0 = a*rho
            # #     y0 = b*rho
            # #     x1 = int(x0 + 1000*(-b))
            # #     y1 = int(y0 + 1000*(a))
            # #     x2 = int(x0 - 1000*(-b))
            # #     y2 = int(y0 - 1000*(a))
            # #
            # #     cv2.line(gray_rgb,(x1,y1),(x2,y2),(0,0,255),2)
            #

print np.mean(matching_score), np.std(matching_score)

            # # # print np.mean(total_x1), np.mean(total_x2), np.mean(total_y1), np.mean(total_y2)
            # # lineLengthDirection = (np.pi/2)+math.atan((np.mean(total_y2)-np.mean(total_y1))/(np.mean(total_x2)-np.mean(total_x1)))
            # #
            # scaleFactor = 3.0
            # cv2.line(input, start- scaleFactor*lineLengthDirection, start+ scaleFactor*lineLengthDirection, cv::Scalar(255,0,255), 0.1)

            # img[int(rheight*crop_ratio):rheight,0:rwidth] = cv2.addWeighted( roi_img, 0.6, img[int(rheight*crop_ratio):int(rheight),0:rwidth],0.8, 0)

#             t_end = time.time()
#             total_time = total_time + t_end-t_start
#
# print('Prediction time: ', total_time)
############################ HOUGH transform ############################

############################ Linear Regression transform ############################

# for label_file in glob.glob(osp.join(input_dir, '*.png')):
#         print(label_file)
#         with open(label_file) as f:
#             base = osp.splitext(osp.basename(label_file))[0]
#             img = cv2.imread(label_file)
#
#             crop_ratio = 0.3
#
#             # Load image, convert to grayscale, threshold and find contours
#             # img = cv2.imread("/home/vignesh/dummy_folder/20191010_L1_N_1498_crop_pred.png")
#             rheight, rwidth = img.shape[:2]
#
#             roi_img = img[int(crop_ratio*rheight):rheight,0:rwidth]
#
#             gray = cv2.cvtColor(roi_img,cv2.COLOR_BGR2GRAY)
#
#             ret, thresh = cv2.threshold(gray, 127, 255, 0)
#
#             t_start = time.time()
#
#             _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#             lt_method = cv2.DIST_L2
#             if len(contours) >= 2:
#                 sort = sorted(contours, key=cv2.contourArea, reverse=True)
#
#                 cv2.drawContours(roi_img, sort[0], -1, (0,255,0), 3)
#
#                 x1 = []
#                 y1 = []
#                 for i in range(len(sort[0])):
#                     x,y =  sort[0][i][0]
#                     x1.append(x)
#                     y1.append(y)
#                     # print x, y
#
#                 fit_c = np.polyfit(y1, x1, 1, full=True)
#                 plotyc = np.linspace(0, roi_img.shape[0]-1, roi_img.shape[0])
#                 fitxc_ = fit_c[0][0]*plotyc + fit_c[0][1]
#                 pts_left = np.array([np.transpose(np.vstack([fitxc_, plotyc]))])
#                 pts_left = pts_left[0].astype(int)
#                 # print len(pts_left)
#                 cv2.polylines(roi_img, [pts_left], 0, (255,255,0), thickness=5, lineType=8, shift=0)
#
#                 cv2.drawContours(roi_img, sort[1], -1, (0,255,0), 3)
#                 x2 = []
#                 y2 = []
#                 for i in range(len(sort[1])):
#                     x,y =  sort[1][i][0]
#                     x2.append(x)
#                     y2.append(y)
#                     # print x, y
#
#                 fit_c = np.polyfit(y2, x2, 1, full=True)
#                 plotyc = np.linspace(0, roi_img.shape[0]-1, roi_img.shape[0])
#                 fitxc_ = fit_c[0][0]*plotyc + fit_c[0][1]
#                 pts_left1 = np.array([np.transpose(np.vstack([fitxc_, plotyc]))])
#                 pts_left1 = pts_left1[0].astype(int)
#                 # print len(pts_left)
#                 cv2.polylines(roi_img, [pts_left1], 0, (255,255,0), thickness=5, lineType=8, shift=0)
#
#
#             min_ind = min(len(sort[0]), len(sort[1]))
#             average_contour = (sort[0][0:min_ind]+sort[1][0:min_ind])/2
#
#             # x1 = []
#             # y1 = []
#             # for i in range(len(average_contour)):
#             #     x,y =  average_contour[i][0]
#             #     x1.append(x)
#             #     y1.append(y)
#             #     # print x, y
#             #
#
#             # x_f = (x1+x2)/2
#             # y_f = (y1+y2)/2
#             # fit_c = np.polyfit(y_f, x_f, 1, full=True)
#             # plotyc = np.linspace(0, roi_img.shape[0]-1, roi_img.shape[0])
#             # fitxc_ = fit_c[0][0]*plotyc + fit_c[0][1]
#             # pts_left = np.array([np.transpose(np.vstack([fitxc_, plotyc]))])
#             # pts_left = pts_left[0].astype(int)
#             # print len(pts_left)
#             midlane = (pts_left+pts_left1)/2
#             cv2.polylines(roi_img, [midlane], 0, (255,0,0), thickness=5, lineType=8, shift=0)
#
#             # then apply fitline() function
#             # [vx,vy,x,y] = cv2.fitLine(average_contour, lt_method, 0, 0.01, 0.01)
#             #
#             # # Now find two extreme points on the line to draw line
#             # lefty = int((-x*vy/vx) + y)
#             # righty = int(((gray.shape[1]-x)*vy/vx)+y)
#             #
#             # #Finally draw the line
#             # cv2.line(roi_img,(gray.shape[1]-1,righty),(0,lefty),(0,0,255), 2)
#             # print gray.shape[1]-1, righty, lefty, vx,vy,x,y
#
#             img[int(rheight*crop_ratio):rheight,0:rwidth] = cv2.addWeighted( roi_img, 0.6, img[int(rheight*crop_ratio):int(rheight),0:rwidth],0.8, 0)
#
#             cv2.imwrite('/home/vignesh/dummy_folder/test_cases/uneven_plants/LTT/'+str(base)+'.png',img)
#
#             ### CRDA
#             import csv
#             gt_row_x = []
#             gt_row_y = []
#             base_name = osp.splitext(base)[0][0:18]
#
#             # with open('/home/vignesh/dummy_folder/test_cases/inclined_terrains/ground_truth/'+os.path.splitext(self.base)[0][0:11]+'.csv') as csvfile:
#             if (osp.splitext(base)[0][11] != str('_')):
#                base_name = osp.splitext(base)[0][0:19]
#
#             rheight1, rwidth1 = img.shape[:2]
#
#             with open('/home/vignesh/dummy_folder/test_cases/uneven_plants/ground_truth/'+base_name+'.csv') as csvfile:
#                  spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
#                  for row in spamreader:
#                      gt_row_x.append(row[1])
#                      gt_row_y.append(row[2])
#
#             gt_x = []
#             gt_y = []
#             for gt_in in range(len(gt_row_x)):
#                  gt_x.append(int(gt_row_x[gt_in]))
#                  gt_y.append(float(gt_row_y[gt_in])-rheight1*crop_ratio)
#                  cv2.circle(roi_img, (int(gt_row_x[gt_in]),int(float(gt_row_y[gt_in])-rheight1*crop_ratio)), 0, (255,0,255), thickness=15, lineType=8, shift=0)
#
#             fit_c1 = np.polyfit(gt_y, gt_x, 2, full=True)
#             plotyc1 = np.linspace(0, roi_img.shape[0]-1, roi_img.shape[0])
#             fitxc_1 = fit_c1[0][0]*plotyc1**2 + fit_c1[0][1]*plotyc1 + fit_c1[0][2]
#             pts_left1 = np.array([np.transpose(np.vstack([fitxc_1, plotyc1]))])
#             pts_left1 = pts_left1[0].astype(int)
#
#             ms_temp = 0
#             scale = 1
#
#             for mid_in in range(len(midlane)):
#                #print pts_left[mid_in][0]-pts_left1[mid_in][0]
#                ms = abs(midlane[mid_in][0]-pts_left1[mid_in][0])
#                ms_lite = abs(np.float(ms)/np.float(img.shape[1])) #-(20*scale) #-(5*mid_in) #0.25*self.image.shape[1]
#                ms_temp = ms_temp + ms_lite
#                if (mid_in%35==0):
#                    scale = scale + 1
#
#             ms_f = 1 - (math.pow(ms_temp,2)/len(midlane))
#             if ((ms_f<0) or (ms_f >1)):
#                 ms_f = 0
#
#             matching_score.append(ms_f)
#
#
#             # t_end = time.time()
#             # total_time = total_time + t_end-t_start
#
# print np.sum(matching_score)/len(matching_score)
# print np.mean(matching_score), np.std(matching_score)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print('Prediction time: ', total_time)
