#include <ros/ros.h>
#include <Eigen/Dense>
#include "Eigen/Core"
#include <Eigen/Geometry>
#include <geometry_msgs/Pose.h>
#include <tf/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h> /*doTransform*/
#include "tf_conversions/tf_eigen.h" // Conversion from eigen to TF
#include <std_msgs/Float64.h>

class Lane_control{

public:
  double position_error = 0, q_x = 0 , q_y = 0, yaw = 0, angular_velocity = 0;
  geometry_msgs::Pose thorvald_pose;
  tf::StampedTransform cam_t;
  tf::TransformListener cam_listener;
  geometry_msgs::Point mini_goal_pts;
  std_msgs::Float64 angular_error;
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(3,3); // K matrix for camera intrinsic

  Lane_control();

  void move();
  void initialize();
  double normalizeangle(double bearing);
  Eigen::Vector3d camera2world(Eigen::Vector3d& x_c, Eigen::Vector3d& t_c, Eigen::Quaternionf& R_c);
  void controller();

};
