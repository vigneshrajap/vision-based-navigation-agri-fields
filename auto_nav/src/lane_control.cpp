#include "lane_control.h"

Lane_control::Lane_control(){

    posearray_local_sub = nh_.subscribe("centerline_local", 100, &Lane_control::posesCallback, this);
    CameraInfo_sub = nh_.subscribe("/camera/color/camera_info", 100, &Lane_control::imagecaminfoCallback, this);

    cmd_velocities = nh_.advertise<geometry_msgs::Twist>("nav_vel", 100);  // control
    posearray_world = nh_.advertise<geometry_msgs::PoseArray>("centerline_global", 100);  // control
    a_err = nh_.advertise<std_msgs::Float64>("ang_err", 100);
}

void Lane_control::initialize(){
  if(!nh_.getParam("/lane_control/world_frame", world_frame_)) ROS_ERROR("Could not read parameters.");
  if(!nh_.getParam("/lane_control/robot_frame", robot_frame_)) ROS_ERROR("Could not read parameters.");
  if(!nh_.getParam("/lane_control/camera_frame", camera_frame_)) ROS_ERROR("Could not read parameters.");
  robot_pose_listener.waitForTransform(world_frame_, robot_frame_, ros::Time(), ros::Duration(1.0)); // create the listener
  cam_listener.waitForTransform(world_frame_, camera_frame_, ros::Time(), ros::Duration(1.0)); // create the listener
}

// PoseArray data from camera
void Lane_control::posesCallback (const geometry_msgs::PoseArray::ConstPtr& poses_msg){
    poses_cam.header = poses_msg->header;
    if(poses_msg->poses.size() > 0)
       poses_cam.poses = poses_msg->poses;
}

void Lane_control::imagecaminfoCallback(const sensor_msgs::CameraInfoConstPtr& cam_msg){ // RGB Image
  K << cam_msg->K[0],cam_msg->K[1],cam_msg->K[2],
       cam_msg->K[3],cam_msg->K[4],cam_msg->K[5],
       cam_msg->K[6],cam_msg->K[7],cam_msg->K[8];
  image_width = cam_msg->width;
  image_height = cam_msg->height;
}

// Normalize the bearing
double Lane_control::normalizeangle(double bearing){
  if (bearing < -M_PI)
    bearing += 2 * M_PI;
  else if (bearing > M_PI)
    bearing -= 2 * M_PI;
}

Eigen::Vector3d Lane_control::camera2world(Eigen::Vector3d& x_c, Eigen::Vector3d& t_c, Eigen::Quaternionf& R_c){

// ray in world coordinates
Eigen::Quaternionf x_c_q(0,x_c.x(),x_c.y(),x_c.z());
Eigen::Quaternionf x_wq(R_c*x_c_q*R_c.conjugate());
Eigen::Vector3d x_w(x_wq.x(),x_wq.y(),x_wq.z());

// distance to the plane
// d = dot((t_p - t_c),n_p)/dot(x_w,n_p)
// simplified expression assuming plane t_p = [0 0 0]; n_p = [0 0 1];
double d = -t_c.z()/x_w.z();

//intersection point
Eigen::Vector3d x_p = d*x_w+t_c;
return x_p;
}

double Lane_control::controller(geometry_msgs::PoseArray goal_pts){
  mini_goal_pts = goal_pts.poses[40].position;

  robot_pose_listener.lookupTransform(world_frame_, robot_frame_, ros::Time(0), robot_t);  // Converts to World (Map) Frame
  tf::Quaternion quat(robot_t.getRotation().x(),robot_t.getRotation().y(),robot_t.getRotation().z(),robot_t.getRotation().w());
  quat = quat.normalize();
  yaw = tf::getYaw(quat);

  // calculation of error
  q_x = mini_goal_pts.x - robot_t.getOrigin().x(); //thorvald_pose.position.x;
  q_y = mini_goal_pts.y - robot_t.getOrigin().y(); //thorvald_pose.position.y;

  // range, bearing
  position_error = sqrt(pow(q_x, 2) + pow(q_y, 2));
  angular_error.data = normalizeangle(atan2(q_y, q_x) - yaw);
  a_err.publish(angular_error); // publish the angular error

  angular_velocity = normalizeangle(atan2(2*1.05*sin(angular_error.data),position_error)); // Pure Pursuit Controller
  return angular_velocity;
}

void Lane_control::move()
{
  initialize();
  ros::Rate r(10);

  while(ros::ok()){
  ros::spinOnce();

  if(row_follow_mode == true){ // row follow mode

    cam_listener.lookupTransform(world_frame_, camera_frame_, ros::Time(0), cam_t);  // Converts to World (Map) Frame
    Eigen::Vector3d t_c(cam_t.getOrigin().x(),cam_t.getOrigin().y(),cam_t.getOrigin().z());
    Eigen::Quaternionf R_c(cam_t.getRotation().w(),cam_t.getRotation().x(),cam_t.getRotation().y(),cam_t.getRotation().z());

    //Eigen::Vector3d p_c; //added roi_y for fitting original image
    if(poses_cam.poses.size() > 0){
      for(int i = 0; i<poses_cam.poses.size(); i++){
        Eigen::Vector3d p_c(poses_cam.poses[i].position.x,poses_cam.poses[i].position.y,1); //added roi_y for fitting original image
        Eigen::Vector3d x_c(K.inverse()*p_c); // Intrinsic Calibration
        x_c = x_c.normalized();

        Eigen::Vector3d x_p = camera2world(x_c, t_c, R_c); // Intersection Point Function
        geometry_msgs::Pose pf;
        pf.position.x = x_p[0];
        pf.position.y = x_p[1];
        poses_world.poses.push_back(pf);
      }
      poses_world.header.stamp = ros::Time::now();
      poses_world.header.frame_id = world_frame_;
      posearray_world.publish(poses_world);

      angular_velocity = controller(poses_world); // low-level controller
      est_twist_msgs.linear.x = 0.1; // Sets at constant speed
      est_twist_msgs.angular.z = angular_velocity;
      cmd_velocities.publish(est_twist_msgs);
    }

    else{
      est_twist_msgs.linear.x = 0; // Stop when no centerline received
      est_twist_msgs.angular.z = 0;
      cmd_velocities.publish(est_twist_msgs);
    }

   } // row follow mode

  r.sleep();
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "lane_control");

  Lane_control lane_control;
  lane_control.move();

  return 0;
};
