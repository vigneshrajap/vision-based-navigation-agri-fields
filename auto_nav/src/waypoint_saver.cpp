#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <fstream>
#include <string>

int save_joy_button_;
sensor_msgs::Joy joy_msgs;
std::string filename, world_frame_, robot_frame_;
std::vector<geometry_msgs::Pose> waypoints;

void waypointsJoyCallback(const sensor_msgs::Joy &joy_msg)
{
 joy_msgs.buttons.resize(15);

 if(joy_msg.buttons.size()>0){

   for(int bns = 0; bns<(joy_msg.buttons.size()); bns++){
    joy_msgs.buttons[bns] = joy_msg.buttons[bns];
   }

 }

}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "current_waypoint_.saver");
    ros::NodeHandle nh_;
    ros::Rate r(1);

    ros::Subscriber waypoints_joy_sub_ = nh_.subscribe("/joy", 10, &waypointsJoyCallback);

    if(!nh_.getParam("/waypoint_saver/filename", filename)) ROS_ERROR("Could not read parameters.");
    if(!nh_.getParam("/waypoint_saver/save_joy_button", save_joy_button_)) ROS_ERROR("Could not read parameters.");
    if(!nh_.getParam("/waypoint_saver/robot_frame", robot_frame_)) ROS_ERROR("Could not read parameters.");
    if(!nh_.getParam("/waypoint_saver/world_frame", world_frame_)) ROS_ERROR("Could not read parameters.");

    tf::TransformListener tf_listener_;
    ros::Duration duration_min(2.0);
    ros::Time current_time, last_time;
    current_time = ros::Time::now();
    last_time = ros::Time::now();

    while(nh_.ok()){

    ros::spinOnce();

    current_time = ros::Time::now();

    if(joy_msgs.buttons.size()>0){

     if(joy_msgs.buttons[save_joy_button_] == 1 && ((current_time-last_time) > duration_min)){

       try{
       tf::StampedTransform waypoint;
       tf_listener_.lookupTransform(world_frame_, robot_frame_, ros::Time(0.0), waypoint);

       geometry_msgs::Pose current_waypoint_;
       current_waypoint_.position.x = waypoint.getOrigin().x();
       current_waypoint_.position.y = waypoint.getOrigin().y();
       current_waypoint_.position.z = waypoint.getOrigin().z();

       current_waypoint_.orientation.x = waypoint.getRotation().x();
       current_waypoint_.orientation.y = waypoint.getRotation().y();
       current_waypoint_.orientation.z = waypoint.getRotation().z();
       current_waypoint_.orientation.w = waypoint.getRotation().w();
       waypoints.push_back(current_waypoint_);

       last_time = current_time;
       ROS_INFO("New waypoint saved");

       }
       catch(tf::TransformException &e)
       {
       ROS_WARN_STREAM("tf::TransformException: " << e.what());
       }


        std::ofstream ofs(filename.c_str(), std::ios::out);
        if(ofs.is_open())
        {

         ofs << "waypoints:" << std::endl;
         for(int i = 0; i < waypoints.size(); i++)
         {
          ofs << "   " << "- pose:" << std::endl;
          ofs << "        x: " << waypoints[i].position.x << std::endl;
          ofs << "        y: " << waypoints[i].position.y << std::endl;
          ofs << "        theta: " << tf::getYaw(waypoints[i].orientation) << std::endl;
         }

        }

        // ROS_INFO_STREAM("Writing current_waypoint_.data to " << filename);
        ofs.close();

    }

   }

   r.sleep();
  }

  return 0;
}
