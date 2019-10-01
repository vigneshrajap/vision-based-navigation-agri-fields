#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <std_srvs/Empty.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Twist.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <yaml-cpp/yaml.h>
#include <nav_msgs/Odometry.h>

#include <vector>
#include <fstream>
#include <string>
#include <exception>
#include <math.h>
#include <limits>

// ROS message includes
#include <auto_nav/sub_goal.h>

int row_transit_mode = 0, turn_side = 1, row_no = 0, no = 0;
geometry_msgs::PoseArray waypoints_;
geometry_msgs::Twist nav_velocities;
geometry_msgs::Pose current_waypoint_, last_waypoint_, curr_pose_;
std::string robot_frame_, world_frame_, filename;
double yaw, lin_vel_max = 0.2, ang_vel_max = 1.57;
auto_nav::sub_goal end_row_transit;

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

bool readFile(const std::string &filename){
     waypoints_.poses.clear();
     try{
         std::ifstream ifs(filename.c_str(), std::ifstream::in);
         if(ifs.good() == false){
             return false;
         }

         YAML::Node node;
         node = YAML::Load(ifs);
         const YAML::Node &wp_node_tmp = node["waypoints"];
         const YAML::Node *wp_node = wp_node_tmp ? &wp_node_tmp : NULL;

         geometry_msgs::Pose pose;
         if(wp_node != NULL){
             for(int i=0; i < wp_node->size(); i++){

                  std::string point_x = (*wp_node)[i]["pose"]["x"].as<std::string>();
                  std::string point_y = (*wp_node)[i]["pose"]["y"].as<std::string>();
                  std::string point_theta = (*wp_node)[i]["pose"]["theta"].as<std::string>();

                  pose.position.x = std::stof(point_x);
                  pose.position.y = std::stof(point_y);
                  pose.orientation = tf::createQuaternionMsgFromYaw(std::stof(point_theta));
                  waypoints_.poses.push_back(pose);
             }

         }
         else{
             return false;
          }
    }

    catch(YAML::ParserException &e){
    ROS_ERROR("waypoint file cannot be opened!");
    }
return true;
}

void roboposcallback(const geometry_msgs::Pose &rob_pose_msg){
   curr_pose_.position = rob_pose_msg.position;
   curr_pose_.orientation = rob_pose_msg.orientation;

   tf::Quaternion quat(curr_pose_.orientation.x,curr_pose_.orientation.y, curr_pose_.orientation.z, curr_pose_.orientation.w);
   quat = quat.normalize();
   yaw = tf::getYaw(quat);
}

bool change_row(auto_nav::sub_goal::Request &req, rasberry_agricultural_costmaps::sub_goal::Response &res)
{
     row_transit_mode = row_transit_mode + req.counter;
     ROS_INFO("transition service on time");

     /*
     if(row_transit_mode == 2){
       no = 2;
       // present_waypoints = 2;
       current_waypoint_ = waypoints_.poses[no];
     }

     if((row_transit_mode%2) == 0)
       turn_side = 2;
     else
       turn_side = 1;
    */

     return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "waypoints_nav");
  ros::NodeHandle n;
  ros::Rate r(1); // 1 hz

  // Publishers
  ros::Publisher cmd_vel_pub_ = n.advertise<geometry_msgs::Twist>("/nav_vel",1);
  ros::Publisher waypoints_loc_ = n.advertise<geometry_msgs::Pose>("/current_waypoint",1);

  // Subscribers
  ros::Subscriber robo_pos_sub_ = n.subscribe("/robot_pose",1, &roboposcallback);

  // Service Servers
  ros::ServiceServer service = n.advertiseService("/row_transition_mode", change_row);

  // Service Client
  ros::ServiceClient client = n.serviceClient<auto_nav::sub_goal>("/row_transition_end_1");

  // Params
  if(!n.getParam("/waypoints_nav/robot_frame", robot_frame_)) ROS_ERROR("Could not read parameters.");
  if(!n.getParam("/waypoints_nav/world_frame", world_frame_)) ROS_ERROR("Could not read parameters.");
  if(!n.getParam("/waypoints_nav/filename", filename)) ROS_ERROR("Could not read parameters.");

  ROS_INFO_STREAM("Read waypoints data from " << filename);

    if(!readFile(filename)) ROS_ERROR("Failed loading waypoints file");

    if(waypoints_.poses.size())
    {

      current_waypoint_ = waypoints_.poses[0];
      last_waypoint_ = waypoints_.poses[waypoints_.poses.size()-1];
    }

  while(n.ok()){

  ros::spinOnce();

   // if(row_transit_mode > row_no){ // sub check

      Next_pt:
      if(no < waypoints_.poses.size()){
           // tell the action client that we want to spin a thread by default
           MoveBaseClient ac("move_base", true);

           // wait for the action server to come up
           while(!ac.waitForServer(ros::Duration(5.0))){
           ROS_INFO("Waiting for the move_base action server to come up");
           }

           // Declaring move base goal
           move_base_msgs::MoveBaseGoal goal;

           // we'll send a goal to the robot to move 2 meters forward
           goal.target_pose.header.frame_id = "map";
           goal.target_pose.header.stamp = ros::Time::now();
           goal.target_pose.pose.position =  current_waypoint_.position;
           goal.target_pose.pose.orientation = current_waypoint_.orientation;
           ROS_INFO("SENDING GOAL!!!");
           ac.sendGoal(goal);
           ac.waitForResult();

         if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
         {
            no = no + 1;

            if(no<waypoints_.poses.size())
            {
                ROS_INFO("Go to next waypoint");
                current_waypoint_ = waypoints_.poses[no];
                waypoints_loc_.publish(current_waypoint_);
                goto Next_pt;
            }
            else  ROS_INFO("Goal Reached");

          }

          else
            ROS_INFO("Failed to reach the goal");

         }

        else
        {

          end_row_transit.request.counter = 1;
          if (client.call(end_row_transit)) ROS_INFO("End of Row Transition_1");

          nav_velocities.linear.x  = 0;
          nav_velocities.angular.z = 0;
          row_no = row_transit_mode;
        }

  //  } // sub check

  cmd_vel_pub_.publish(nav_velocities);
  r.sleep();
  }

return 0;
}
