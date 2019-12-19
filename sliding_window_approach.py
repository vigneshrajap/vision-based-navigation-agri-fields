#!/usr/bin/env python
# license removed for brevity
import numpy as np
import cv2
from sklearn.cluster import KMeans
from itertools import imap
import matplotlib.pyplot as plt
prev_modifiedCenters = []
base_size = 0.2
clusters = 2

def perspective_warp(img, dst_size, src, dst): # Choose the four vertices

    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped, M

def inv_perspective_warp(img, dst_size, src, dst):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped, M

def initialPoints(warped_img, margin_l, margin_r):

     global prev_modifiedCenters, base_size, clusters

     # Crop the search space
     bottom = (warped_img.shape[0] - int(base_size * warped_img.shape[0]))
     base = warped_img[bottom:warped_img.shape[0], 0:warped_img.shape[1]]

     # Find white pixels
     whitePixels = np.argwhere(base == 255)

     #cluster_1 = []
     #cluster_1 = np.zeros([base.shape[0],base.shape[1],1],dtype=np.uint8)
     #cluster_2 = np.zeros([base.shape[0],base.shape[1],1],dtype=np.uint8)
     #print base_size, margin_l, margin_r

     # Attempt to run kmeans (the kmeans parameters were not chosen with any sort of hard/soft optimization)
     try:
         kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=3, max_iter=150).fit(whitePixels)
         # for k_i in range(0, len(whitePixels)):
         #     if kmeans.labels_[k_i]==0:
         #        cluster_1[whitePixels[k_i][0]][whitePixels[k_i][1]] = 255
         #     else:
         #        cluster_2[whitePixels[k_i][0]][whitePixels[k_i][1]] = 255
         #
         # im_1, contours_1, hierarchy_1 = cv2.findContours(cluster_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         # im_2, contours_2, hierarchy_2 = cv2.findContours(cluster_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

         # print contours, kmeans.cluster_centers_[0] #len(whitePixels), len(kmeans.labels_), len(cluster_1), len(cluster_2), len(cluster_1)+len(cluster_2) #kmeans.cluster_centers_,

         # [margin_r, margin_c] = np.std(whitePixels, axis=0)
         # margin = margin_r.astype(int)

         #print margin_c, margin_r
         # plt.scatter(whitePixels[:, 1], whitePixels[:, 0], c=kmeans.labels_)
         # plt.gca().invert_yaxis()
         # centers1 = [list(imap(int, center)) for center in kmeans.cluster_centers_]
         #
         # plt.scatter(kmeans.cluster_centers_[1][:], kmeans.cluster_centers_[0][:], marker='x', s=169, linewidths=4,
         #    color='w', zorder=10)
         # plt.title("K-Means Clustering")
         #
         # plt.show()

         # for i in len(cluster_index_):
             #whitePixels[0][i] =
         # print base.shape, centroids #, cluster_index_[0], whitePixels[0][1]
         # define criteria and apply kmeans()

     except:
          # If kmeans fails increase the search space unless it is the whole image, then it fails
          if base_size  > 1:
              return None
          else:
              base_size  = base_size  * 1.5
              return initialPoints(warped_img, margin_l, margin_r)

     # conver centers to integer values so can be used as pixel coords
     centers = [list(imap(int, center)) for center in kmeans.cluster_centers_]

     # d_0 = []
     # d_1 = []
     # for c_i in range(0, len(contours_1[0])):
     #   d_0.append(contours_1[0][c_i][0][0]-centers[0][1])
     # for c_j in range(0, len(contours_2[0])):
     #   d_1.append(contours_2[0][c_j][0][0]-centers[1][1])
     #margin = abs(max(d_0))
     #margin_1 = abs(max(d_1))
     #margin_r.astype(int)


     #print  contours_2[0][c_j][0][0], centers[1][1]

     # img = np.zeros([base.shape[0],base.shape[1],3],dtype=np.uint8)
     # cv2.drawContours(img, contours_1, -1, (0,255,0), 3)
     # cv2.putText(img, 'X',(centers[0][1], centers[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
     # cv2.drawContours(img, contours_2, -1, (0,255,0), 3)
     # cv2.putText(img, 'X',(centers[1][1], centers[1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
     # cv2.imwrite("/home/vignesh/result.png", img)

     # Lamda function to remap the y coordiates of the clusters into the image space
     increaseY = lambda points: [points[0] + int((1 - base_size) * warped_img.shape[0]), points[1]]

     # map the centers in terms of the image space
     modifiedCenters = [increaseY(center) for center in centers]

     # if abs(modifiedCenters[0][1]-modifiedCenters[1][1])<50:
     #     #print modifiedCenters, prev_modifiedCenters
     #     modifiedCenters = prev_modifiedCenters
     #     return margin, margin_1, modifiedCenters

     # prev_modifiedCenters = modifiedCenters

     # return a list of tuples for centers
     return margin_l, margin_r, modifiedCenters

def sliding_window(img, modifiedCenters, nwindows=12, margin=35, margin_1=35, minpix=1, draw_windows=True):
    global prev_modifiedCenters, base_size, clusters

    left_a = []
    left_b = []
    left_c = []
    right_a = []
    right_b = []
    right_c = []

    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    #modifiedCenters = initialPoints(img)
    leftx_base = modifiedCenters[0][1]
    rightx_base = modifiedCenters[1][1]

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    #print margin
    for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = img.shape[0] - (window+1)*window_height
      win_y_high = img.shape[0] - window*window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin_1
      win_xright_high = rightx_current + margin_1

      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
      (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)

      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

      # Draw the windows on the visualization image
      if draw_windows == True:
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (255,0,0), 3)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (255,0,0), 3)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 100, 255] #[255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def visualization_polyfit(out_img, curves, lanes, ploty, modifiedCenters):

   cv2.circle(out_img, (modifiedCenters[0][1], modifiedCenters[0][0]), 8, (0, 255, 0), -1)
   cv2.circle(out_img, (modifiedCenters[1][1], modifiedCenters[1][0]), 8, (0, 255, 0), -1)

   # Fitted curves as points
   leftLane = np.array([np.transpose(np.vstack([curves[0], ploty]))])
   rightLane = np.array([np.flipud(np.transpose(np.vstack([curves[1], ploty])))])
   points = np.hstack((leftLane, rightLane))
   curves_m = (curves[0]+curves[1])/2
   midLane = np.array([np.transpose(np.vstack([curves_m, ploty]))])

   leftLane_i = leftLane[0].astype(int)
   rightLane_i = rightLane[0].astype(int)
   midLane_i = midLane[0].astype(int)

   cv2.polylines(out_img, [leftLane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)
   cv2.polylines(out_img, [rightLane_i], 0, (0,255,255), thickness=5, lineType=8, shift=0)
   cv2.polylines(out_img, [midLane_i], 0, (255,0,255), thickness=5, lineType=8, shift=0)

   return out_img, midLane_i
