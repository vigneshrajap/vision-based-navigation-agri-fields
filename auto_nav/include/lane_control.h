nclude <ros/ros.h>
#include <Eigen/Dense>
#include "Eigen/Core"
#include <Eigen/Geometry>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h> /*doTransform*/
#include "tf_conversions/tf_eigen.h" // Conversion from eigen to TF
#include <std_msgs/Float64.h>
#include <sensor_msgs/CameraInfo.h>

class Lane_control{

public:
  double position_error = 0, q_x = 0 , q_y = 0, yaw = 0, angular_velocity = 0;
  geometry_msgs::Pose thorvald_pose;
  int image_width, image_height;
  tf::StampedTransform robot_t, cam_t;
  geometry_msgs::Twist est_twist_msgs;
  tf::TransformListener robot_pose_listener, cam_listener;
  geometry_msgs::Point mini_goal_pts;
  std_msgs::Float64 angular_error;
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(3,3); // K matrix for camera intrinsic
  geometry_msgs::PoseArray poses_cam, poses_world;
  bool row_follow_mode = true;
  std::string filename, world_frame_, robot_frame_, camera_frame_;

  ros::NodeHandle nh_;

  // Subscribers
  ros::Subscriber posearray_local_sub,CameraInfo_sub;

  // Publishers
  ros::Publisher cmd_velocities, posearray_world, a_err;

  Lane_control();
  void posesCallback (const geometry_msgs::PoseArray::ConstPtr& poses_msg);
  void imagecaminfoCallback(const sensor_msgs::CameraInfoConstPtr& cam_info_msg);

  void move();
  void initialize();
  double normalizeangle(double bearing);
  Eigen::Vector3d camera2world(Eigen::Vector3d& x_c, Eigen::Vector3d& t_c, Eigen::Quaternionf& R_c);
  double controller(geometry_msgs::PoseArray goal_pts);

};

