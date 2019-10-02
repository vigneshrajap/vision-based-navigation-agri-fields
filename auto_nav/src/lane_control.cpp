#include "lane_control.h"

Lane_control::Lane_control(){}

void Lane_control::initialize(){

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

void Lane_control::controller(){
  // calculation of error
  q_x = mini_goal_pts.x - thorvald_pose.position.x;
  q_y = mini_goal_pts.y - thorvald_pose.position.y;

  // range, bearing
  position_error = sqrt(pow(q_x, 2) + pow(q_y, 2));
  angular_error.data = normalizeangle(atan2(q_y, q_x) - yaw);

  angular_velocity = normalizeangle(atan2(2*1.05*sin(angular_error.data),position_error)); // Pure Pursuit Controller
}

void Lane_control::move()
{
  initialize();
  ros::Rate r(10);

  while(ros::ok()){
  ros::spinOnce();

  // if(row_follow_mode == true){ // row follow mode

    cam_listener.lookupTransform("map", "camera_color_optical_frame", ros::Time(0), cam_t);  // Converts to World (Map) Frame

    Eigen::Vector3d p_c; //added roi_y for fitting original image

    // Eigen::Vector3d p_c(l[2*e_pts]+roi_x,l[(2*e_pts)+1]+roi_y,1); //added roi_y for fitting original image

    Eigen::Vector3d x_c(K.inverse()*p_c); // Intrinsic Calibration
    Eigen::Vector3d t_c(cam_t.getOrigin().x(),cam_t.getOrigin().y(),cam_t.getOrigin().z());
    Eigen::Quaternionf R_c(cam_t.getRotation().w(),cam_t.getRotation().x(),cam_t.getRotation().y(),cam_t.getRotation().z());
    x_c = x_c.normalized();

    Eigen::Vector3d x_p = camera2world(x_c,t_c,R_c); // Intersection Point Function
    controller(); // low-level controller

  // } // row follow mode

  r.sleep();
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "lane_control");

  Lane_control lane_control;
  lane_control.move();

  return 0;
};
