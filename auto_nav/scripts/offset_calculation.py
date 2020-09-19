#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge
import rospy
import numpy as np
import os.path as osp
import glob
import rosbag
from csv import reader, writer

import shapely.geometry as geom

import geo2UTM # For lat/long to UTM conversion

class automated_labelling():

    def __init__(self):

        self.gps_topic_name = str('gps/filtered')
        self.ekf_topic_name = str('/odometry/global')
        self.imu_topic_name = str('/imu/data')
        self.image_topic_name = str('/camera/color/image_raw')

        self.oneshot_gps = 0
        self.dt_gps = 0
        self.oneshot_ekf = 0
        self.dt_ekf = 0

        self.increment = 1 # Fit Line segments over increment values

        self.gt_xy = []
        with open('/home/vignesh/matlab/ekf_test/apple_fields.csv', 'r') as gnss_read_obj:
            # Skip GNSS header
            gnss_data = reader(gnss_read_obj)
            gnss_line = next(gnss_data)

            for gnss_row in gnss_data:
                self.gt_xy.append([float(gnss_row[1]),float(gnss_row[2])])

        # print self.gt_xy #[0][1]

    def estimate_LO(self):

        # compute the linear segments around each ground truth point (use neighbouring points)
       ngt = len(self.gt_xy)
       ngps = len(self.utm_fix_)
       self.lateral_offsets = np.zeros(int(ngps))

       # segs = np.zeros((ngt, 4))
       # bounds = np.zeros((ngt, 4))
       # for  i = 2:(ngt-1):
       #      # ax + by + c = 0
       #      [segs(i,1), segs(i,2), segs(i,3), segs(i,4)] = orthfit2d(gt_xy(i-1:i+1,1), gt_xy(i-1:i+1,2));
       #
       #       bounds(i,1:4) = [gt_xy(i-1,1), gt_xy(i-1,2), gt_xy(i+1,1), gt_xy(i+1,2)];

       for i in range(ngps):
         # matching ground truth segment (closest ground truth point)
         dist_0 = np.zeros(ngt-2*self.increment)
         for j in range(self.increment,ngt-self.increment):
             dist_0[j] = np.sqrt(pow(self.utm_fix_[i][0]-self.gt_xy[j][0],2)+pow(self.utm_fix_[i][1]-self.gt_xy[j][1],2))

         #print(np.min(dist_0))
         gt_ind = np.where(dist_0 == np.min(dist_0))+self.increment
         print gt_ind
         self.line = geom.LineString(self.gt_xy[int(gt_ind[0][0])-self.increment:int(gt_ind[0][0])+self.increment])
         point = geom.Point(self.utm_fix_[i][0], self.utm_fix_[i][1]) # x, y
         self.lateral_offsets[i] = self.line.distance(point)

         aX, aY, bX, bY = self.line.bounds
         cX, cY = (self.utm_fix_[i][0], self.utm_fix_[i][1])
         if ((bX - aX)*(cY - aY) - (bY - aY)*(cX - aX)) > 0:
                self.lateral_offset = -self.lateral_offset

         print "lateral_offset:", self.lateral_offset

         # print(self.line.distance(point))

         # [~, ind] = min(np.linalg.norm(self.ekf_fix_(i,:) - self.gt_xy, 2, 2));

        # % find line perpendicular to matching ground truth segment
        # x = gps_xy(i,1); y = gps_xy(i,2);
        # abc1 = segs(ind,1:3);
        # a1 = abc1(1); b1 = abc1(2); c1 = abc1(3);
        # a2 = b1; b2 = -a1; c2 = -a2 * x - b2 * y;
        # abc2 = [a2, b2, c2];
        # % find intersection point between ground truth segment and perpendicular line
        # % (closest point on ground truth segment to robot)
        # int = cross(abc1,abc2);
        #
        # % orientation of the ground truth segment at this robot point
        # row_orientation(i,1) = rad2deg(atan2(b2, a2));
        #
        # % no intersection
        # if (int(3) == 0)
        #     continue;
        # end
        #
        # intxy = int(1:2) ./ int(3);
        #
        # % lateral offset is length of vector from robot to intersection
        # lateral_offsets(i,1) = norm(gps_xy(i,:) - intxy);
        #
        # aX = bounds(ind,1); aY = bounds(ind,2); bX = bounds(ind,3); bY = bounds(ind,4);
        # if (((bX - aX)*(gps_xy(i,2) - aY) - (bY - aY)*(gps_xy(i,1) - aX)) > 0)
        #      lateral_offsets(i,1) = -lateral_offsets(i,1);
        # end



       # dist_0 = np.empty([np.int(len(self.ekf_fix_)/self.increment),1])
       # lines = geom.MultiLineString()
       # multilines = []
       # lines = []
       #
       # # Increment by parameter for multiple line segments along GT points
       # for ind in range(self.increment+1,len(self.gt_xy),self.increment):
       #
       #      self.line = geom.LineString(self.gt_xy[ind-self.increment:ind,:])
       #      point = geom.Point(self.pose_map_r.pose.position.x, self.pose_map_r.pose.position.y) # x, y
       #      dist_0[np.int(ind/self.increment)-1] = self.line.distance(point)
       #      lines.append(self.line)
       #
       # multilines.append(geom.MultiLineString(lines))
       #
       # # Min Lateral Offset and its line segement index
       # self.lateral_offset = np.min(dist_0)
       # segment_index = np.where(dist_0 == np.min(dist_0))
       # aX, aY, bX, bY = multilines[0][segment_index[0][0]].bounds
       # cX, cY = (self.pose_map_r.pose.position.x, self.pose_map_r.pose.position.y)
       # if ((bX - aX)*(cY - aY) - (bY - aY)*(cX - aX)) > 0:
       #     self.lateral_offset = -self.lateral_offset
       #
       # print "lateral_offset:", self.lateral_offset

if __name__ == '__main__':
    try:
        #Initialize node
        rospy.init_node('lateral_heading_offset')

        auto_label = automated_labelling()
        input_dir = osp.expanduser("~/matlab/ekf_test/")

        for bag_file in sorted(glob.glob(osp.join(input_dir, '*.bag'))):
            print(bag_file)

            bag = rosbag.Bag(bag_file)

            ##################### Extract GNSS Data #####################
            auto_label.gps_fix_ = []
            auto_label.utm_fix_ = []
            auto_label.dt_gps_fix_ = []

            for gps_topic, gps_msg, t_gps in bag.read_messages(topics=[auto_label.gps_topic_name]):
                 if(auto_label.oneshot_gps==0):
                     t0 = t_gps.to_sec()
                     auto_label.oneshot_gps = 1

                 auto_label.dt_gps = auto_label.dt_gps + (t_gps.to_sec()-t0)
                 auto_label.dt_gps_fix_.append(auto_label.dt_gps)
                 auto_label.gps_fix_.append([gps_msg.latitude, gps_msg.longitude])
                 t0 = t_gps.to_sec()

                 # Custom Library to convert geo to UTM co-ordinates
                 gps_fix_utm = geo2UTM.geo2UTM(gps_msg.latitude, gps_msg.longitude)
                 auto_label.utm_fix_.append([gps_fix_utm[0], gps_fix_utm[1]])

            ##################### Extract EKF Fused Data #####################
            auto_label.ekf_fix_ = []
            auto_label.dt_ekf_fix_ = []

            for ekf_topic, ekf_msg, t_ekf in bag.read_messages(topics=[auto_label.ekf_topic_name]):
                 if(auto_label.oneshot_ekf==0):
                     t0 = t_ekf.to_sec()
                     auto_label.oneshot_ekf = 1

                 auto_label.dt_ekf = auto_label.dt_ekf + (t_ekf.to_sec()-t0)
                 auto_label.dt_ekf_fix_.append(auto_label.dt_ekf)
                 auto_label.ekf_fix_.append([ekf_msg.pose.pose.position.x, ekf_msg.pose.pose.position.y])

                 t0 = t_ekf.to_sec()

            auto_label.estimate_LO()

            # print auto_label.dt_ekf_fix_, auto_label.ekf_fix_

        bag.close()

    except rospy.ROSInterruptException:
         cv2.destroyAllWindows() # Closes all the frames
         pass
