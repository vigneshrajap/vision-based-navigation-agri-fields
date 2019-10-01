#include "agri_costmap_layer_v2.h"

PLUGINLIB_EXPORT_CLASS(custom_layer::AgriCostmapLayer_v2, costmap_2d::Layer)

namespace custom_layer{

  void AgriCostmapLayer_v2::onInitialize() {

    ros::NodeHandle g_nh;
    current_ = true;

    vec_sub = g_nh.subscribe("vector_poses", 100, &AgriCostmapLayer_v2::vecposeCallback, this);
    curr_node_sub = g_nh.subscribe("current_node", 100, &AgriCostmapLayer_v2::currnodeCallback, this);

    ros::Rate r(10);
    while (g_nh.ok()&&(vector_receive==0)){
       ros::spinOnce();
       if(costmap_status==1){
         dsrv_ = new dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>(g_nh);
         dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>::CallbackType f = boost::bind(
            &AgriCostmapLayer_v2::reconfigureCB, this, _1, _2);
         dsrv_->setCallback(f);
         costmap_status= 0;
       }

       // std::cout << costmap_status << " " << enabled_ << " " << vector_receive << std::endl;
       r.sleep();
    }

  }

  AgriCostmapLayer_v2::~AgriCostmapLayer_v2(){
    if(dsrv_) delete dsrv_;
  }

  void AgriCostmapLayer_v2::reconfigureCB(auto_nav::custom_costmap_paramsConfig &config, uint32_t level){

    Total_Layers = config.Total_Layers;
    Layer.resize(Total_Layers);
    x_.resize(Total_Layers);  y_.resize(Total_Layers);

    if(0<Layer.size()) Layer[0] = config.FirstLayer;
    if(1<Layer.size()) Layer[1] = config.SecondLayer;
    if(2<Layer.size()) Layer[2] = config.ThirdLayer;
    if(3<Layer.size()) Layer[3] = config.FourthLayer;
    if(4<Layer.size()) Layer[4] = config.FifthLayer;

    costmap_height = config.Costmap_Height;
    costmap_width = config.Cost_Propagation_Area;
    costmap_radius = config.Costmap_Radius;
    pts = config.TotalSegmentPts;
    P.resize(pts*2);

    if(costmap_status==1) //(costmap_status==1)&&
        enabled_ = config.enabled;
    else
        enabled_ = false;

    //if(on_curved_lane == 0) costmap_width = config.Cost_Propagation_Area;
    //if(on_curved_lane == 1) costmap_width = 0.9;

    // ROS_INFO("Agri-Costmaps is up!!");
  }

  void AgriCostmapLayer_v2::vecposeCallback (const geometry_msgs::PoseArray::ConstPtr& vec_msg){

    if(vec_msg->poses.size()>0){
      vec.resize(vec_msg->poses.size());
      yaw_a.resize(vec_msg->poses.size());

      for(int v=0; v < vec_msg->poses.size(); v++){
       cv::Point2f vec_c(vec_msg->poses[v].position.x, vec_msg->poses[v].position.y);
       vec.push_back(vec_c);
       yaw_a.push_back(tf::getYaw(vec_msg->poses[0].orientation));
      }

      if(vector_receive==0){
       costmap_status = 1; //temp
       vector_receive = 1;
       }

      }
    } // Vec callback

  void AgriCostmapLayer_v2::currnodeCallback(const std_msgs::String::ConstPtr& curr_node_msg){ // Current Topological Node
     str_name = curr_node_msg->data;

     // if((str_name=="WayPoint59")){ // Entering curves Frogn_Fields: WayPoint1 Polytunnels:WayPoint134
     //   costmap_status = 1;
     //
     //   dsrv_ = new dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>(nh);
     //   dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>::CallbackType f = boost::bind(
     //      &AgriCostmapLayer_v2::reconfigureCB, this, _1, _2);
     //   dsrv_->setCallback(f);
     // }
     //
     // if((str_name=="WayPoint34")){ // Entering curves Frogn_Fields: WayPoint2 Polytunnels:WayPoint99
     //   costmap_status = 0;
     //   dsrv_ = new dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>(nh);
     //   dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>::CallbackType f = boost::bind(
     //      &AgriCostmapLayer_v2::reconfigureCB, this, _1, _2);
     //   dsrv_->setCallback(f);
     // }
     //
     // if((str_name=="WayPoint6")){ // Entering curves
     //   on_curved_lane = 0;
     //
     //   dsrv_ = new dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>(nh);
     //   dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>::CallbackType f = boost::bind(
     //      &AgriCostmapLayer_v2::reconfigureCB, this, _1, _2);
     //   dsrv_->setCallback(f);
     // }
     //
     // if((str_name=="WayPoint4")){ // Entering curves
     //  on_curved_lane = 1;
     //
     //  dsrv_ = new dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>(nh);
     //  dynamic_reconfigure::Server<auto_nav::custom_costmap_paramsConfig>::CallbackType f = boost::bind(
     //     &AgriCostmapLayer_v2::reconfigureCB, this, _1, _2);
     //  dsrv_->setCallback(f);
     // }

    }

  cv::Point2f AgriCostmapLayer_v2::rotate_vector(cv::Point2f vec_pt, cv::Point2f l_pt, float vec_yaw){ // Rotate the points around the vector
    cv::Point2f l_pt_rot;
    double c = cos(vec_yaw), s = sin(vec_yaw);
    l_pt_rot.x = vec_pt.x+(c*(l_pt.x-vec_pt.x)-s*(l_pt.y-vec_pt.y));
    l_pt_rot.y = vec_pt.y+(s*(l_pt.x-vec_pt.x)+c*(l_pt.y-vec_pt.y));
    return l_pt_rot;
  }

  void AgriCostmapLayer_v2::updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y, double* max_x, double* max_y) {

   if (!enabled_) return;

    if(Total_Layers>0){

      //update the area around the costmap layer
      *min_x = std::min(robot_x+10*Total_Layers*costmap_height, robot_x-10*Total_Layers*costmap_height);
      *min_y = std::min(robot_y+10*Total_Layers*costmap_width, robot_y-10*Total_Layers*costmap_width);
      *max_x = std::max(robot_x+10*Total_Layers*costmap_height, robot_x-10*Total_Layers*costmap_height);
      *max_y = std::max(robot_y+10*Total_Layers*costmap_width, robot_y-10*Total_Layers*costmap_width);

    }

  } // updateBounds

  void AgriCostmapLayer_v2::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j){

   if (!enabled_) return;
   unsigned int mx, my;
   double cell_size = master_grid.getResolution();

   for(int v = 0; v < vec.size(); v++){ // Each Vector Pose Loop

   // Rotate the lines around the pose vector
   std::vector<cv::Point2f> P_i;
   P_i.resize(4);

   P_i[0] = cv::Point2f(vec[v].x-costmap_height, vec[v].y-costmap_width);
   P_i[1] = cv::Point2f(vec[v].x+costmap_height, vec[v].y-costmap_width);
   P_i[2] = cv::Point2f(vec[v].x-costmap_height, vec[v].y+costmap_width);
   P_i[3] = cv::Point2f(vec[v].x+costmap_height, vec[v].y+costmap_width);

   for(int p=0;p<pts; p++){ // Line Segments for L1 and L2

     P[2*p] = (P_i[0]*(1-(float(p)/pts)))+(P_i[1]*(float(p)/pts)); //L1_e1
     P[2*p] = rotate_vector(vec[v], P[(2*p)], yaw_a[v]); // rotate the line points by vector

     P[(2*p)+1] = (P_i[2]*(1-(float(p)/pts)))+(P_i[3]*(float(p)/pts)); //L2_e1
     P[(2*p)+1] = rotate_vector(vec[v], P[(2*p)+1], yaw_a[v]); // rotate the line points by vector
   }

   for (int k = 0; k < P.size(); k++) // Update the costs of each grid cell
   {

     if(master_grid.worldToMap(P[k].x, P[k].y, mx, my)) // Update the line points as grid cells
     {
       if(master_grid.getCost(mx, my)==FREE_SPACE)
       {
        master_grid.setCost(mx, my, Layer[0]);
       }
     }

   } // Update the costs of each grid cell

    // for(int w = 1; w < x_.size(); w++) // revert the size of the circle co-ordinates array for next iteration
    // {
    //  x_[w].resize(0); y_[w].resize(0);
    // }
    //
    // std::vector<cv::Point2f> topLeft, topRight, bottomLeft, bottomRight;
    // topLeft.resize(0); topRight.resize(0); bottomLeft.resize(0); bottomRight.resize(0);
    // std::vector<cv::Point2f> l1, l2;
    // std::vector<double> x_t, y_t;
    //
    // for(int q = 0; q < P.size(); q++) // Line Segments for L1 and L2
    // {
    //
    //   for(int n = 1; n < Total_Layers; n++) // Each Costmap Layer
    //   {
    //
    //   // Create the boundaries for the designated radius
    //   topLeft.push_back(cv::Point2f(P[q].x+n*costmap_radius, P[q].y+n*costmap_radius));
    //   topRight.push_back(cv::Point2f(P[q].x+n*costmap_radius, P[q].y-n*costmap_radius));
    //   bottomLeft.push_back(cv::Point2f(P[q].x-n*costmap_radius, P[q].y+n*costmap_radius));
    //   bottomRight.push_back(cv::Point2f(P[q].x-n*costmap_radius, P[q].y-n*costmap_radius));
    //
    //   l1.resize(0); l2.resize(0); x_t.resize(0); y_t.resize(0);
    //
    //   // Find the circle co-ordinates according to costmap radius
    //   for(int r = 0; r < pts; r++) // rect co-ordinates
    //   {
    //       l1.push_back((topLeft.back()*(1-(float(r)/pts)))+(bottomLeft.back()*(float(r)/pts)));
    //       l2.push_back((topRight.back()*(1-(float(r)/pts)))+(bottomRight.back()*(float(r)/pts)));
    //
    //       for(int u = 0; u < pts; u++)
    //         {
    //           x_t.push_back((l1.back().x*(1-(float(u)/pts)))+(l2.back().x*(float(u)/pts)));
    //           y_t.push_back((l1.back().y*(1-(float(u)/pts)))+(l2.back().y*(float(u)/pts)));
    //
    //           if(sqrt(pow(x_t.back()-P[q].x,2)+pow(y_t.back()-P[q].y,2))<(n*costmap_radius)) // circle co-ordinates
    //           {
    //              x_[n].push_back(x_t.back());
    //              y_[n].push_back(y_t.back());
    //           }
    //
    //          }
    //
    //   } // rect co-ordinates
    //
    //  } // Each Costmap Layer
    //
    // } // Line Segments for L1 and L2
    //
    //  for(int n = 1; n < Total_Layers; n++) // Each Costmap Layer
    //  {
    //
    //   if((x_[n].size()>0)&&(y_[n].size()>0)) // UpdateBounds check
    //    {
    //
    //      // Assign the cost from circle co-ordinates
    //        for(int i = 0; i < x_[n].size(); i++)
    //          {
    //
    //            //for(int j = 0; j < y_[n].size(); j++)
    //              //{
    //
    //                if(master_grid.worldToMap(x_[n][i], y_[n][i], mx, my)) // Update the line points as grid cells
    //                {
    //                  if(master_grid.getCost(mx, my)==FREE_SPACE)
    //                   {
    //                    master_grid.setCost(mx, my, Layer[n]);
    //                   }
    //                }
    //
    //              //}
    //
    //          }
    //
    //    } // UpdateBounds check
    //
    //  } // Each Costmap Layer

   } // For loop for each vector poses

  } // updateCosts

} // end namespace
