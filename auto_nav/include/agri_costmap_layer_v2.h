#ifndef AGRI_COSTMAP_LAYER_H_
#define AGRI_COSTMAP_LAYER_H_
#include <ros/ros.h>

#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <pluginlib/class_list_macros.h>
#include <dynamic_reconfigure/server.h>

#include <auto_nav/custom_costmap_paramsConfig.h>

#include "opencv2/opencv.hpp" // OPENCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// ROS message includes
// #include <auto_nav/sub_goal.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>  // To get the line end points

using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::FREE_SPACE;
std::string str_name;

namespace custom_layer
 {

  class AgriCostmapLayer_v2 : public costmap_2d::Layer
  {
  public:
    AgriCostmapLayer_v2() {
      dsrv_ = NULL; // this is the unsigned char* member of parent class Costmap2D.
    } // Constructor

    virtual ~AgriCostmapLayer_v2();
    virtual void onInitialize();

    virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y, double* max_x, double* max_y);
    virtual void updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);
    virtual void vecposeCallback (const geometry_msgs::PoseArray::ConstPtr& vec_msg); // Line points
    // virtual bool change_row(auto_nav::sub_goal::Request &req, auto_nav::sub_goal::Response &res);

    bool initParameters();

    void currnodeCallback (const std_msgs::String::ConstPtr& curr_node_msg);

    cv::Point2f rotate_vector(cv::Point2f vec_pt, cv::Point2f l_pt, float vec_yaw);

  private:
    void reconfigureCB(auto_nav::custom_costmap_paramsConfig &config, uint32_t level);

    ros::Subscriber vec_sub, curr_node_sub;
    ros::ServiceServer costmapService;

    // Paramters Initialization
    int pts, row_follow = 0;
    std::vector<cv::Point2f> vec;
    std::vector<cv::Point2f> P;
    std::vector<float> Layer;
    std::vector<float> yaw_a;
    std::vector<std::vector<double> > x_, y_;

    double Total_Layers, costmap_height, costmap_width, costmap_radius;
    bool costmap_status = 0, vector_receive = 0, update_bounds = false;
    int on_curved_lane = 0;

    dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig> *dsrv_;

   };
 }
 #endif
