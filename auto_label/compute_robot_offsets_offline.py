import os
import matplotlib.pyplot as plt
import numpy as np
from namedtuples_csv import read_namedtuples_from_csv, write_namedtuples_to_csv
from scipy.interpolate import interp1d
from scipy.signal import resample
from geometric_utilities import direction_sign, angle_between_vectors, signed_distance_point_to_line, closest_point, line_to_next_point, line_fit_from_points, angle_between_lines
import homogeneous_transformation as ht
import argparse

from collections import namedtuple

def align_gt_direction(gt_x, gt_y, robot_x, robot_y):
    gt_dir_vec = [gt_x[-1]-gt_x[0],gt_y[-1]-gt_y[0]]
    robot_dir_vec = [robot_x[-1]-robot_x[0],robot_y[-1]-robot_y[0]]
    #Align ground truth with robot direction
    gt_direction = direction_sign(gt_dir_vec/np.linalg.norm(gt_dir_vec),robot_dir_vec/np.linalg.norm(robot_dir_vec))
    if gt_direction < 0:
        gt_x = list(reversed(gt_x))
        gt_y = list(reversed(gt_y))
    return gt_x, gt_y

def interpolate_position_to_new_time(pos_time, pos_x, pos_y, new_time, interpolation_mode = 'linear'):  
    f_interp_x = interp1d(x=pos_time, y = pos_x, fill_value = 'extrapolate', kind = interpolation_mode)
    f_interp_y = interp1d(x=pos_time, y = pos_y, fill_value = 'extrapolate', kind = interpolation_mode)

    pos_interp_x = f_interp_x(new_time)
    pos_interp_y = f_interp_y(new_time)
    return pos_interp_x, pos_interp_y

def main():
    parser = argparse.ArgumentParser(description="Compute robot offsets from position data in csv files.")
    parser.add_argument("--base_dir", default = './output/position_data', help = "Input and output directory for position data")
    parser.add_argument("--rec_prefix", help = "Mandatory. Row ID")
    parser.add_argument("--smoothing_windows", default = [50,50], help = "Window size for smoothing position data")
    parser.add_argument("--debug",default = False, help = "Debug flag")
    parser.add_argument("--visualize", default = True, help = "Visualization flag")
    args = parser.parse_args()

    compute_and_save_robot_offsets(input_dir = args.base_dir, 
    output_dir = args.base_dir, 
    rec_prefix = args.rec_prefix, 
    debug = args.debug, 
    visualize = args.visualize,
    smoothing_windows = args.smoothing_windows)

def compute_and_save_robot_offsets(input_dir = os.path.join('.','output'), 
    output_dir = os.path.join('.','output'), 
    rec_prefix = '20191010_L3_S_morning_slaloam', 
    smoothing_windows= [5,5],
    debug = False, 
    visualize = False):

    # Read converted positions and timestamps
    gt_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_gt_pos.csv'),'GTPos')
    #robot_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_robot_pos_and_timestamps.csv'), 'RobotPos')
    gps_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_gps_pos_and_timestamps.csv'), 'GPSPos')
    img_meta = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_image_timestamps.csv'), 'ImageMeta')

    #NB: 2 more frames when reading time stamps than what was saved with rqt image viewer. Should implement image saving in the conversion script (or make a separate script.)

    #Fixed transform between robot/camera and GPS antenna (should be read from somewhere)
    #robot_to_gps_translation = [-0.425, 0.62, 1.05] # Fixed Static Transform
    robot_to_gps_translation = [0, 0.62, 1.05] #approx. camera y position (no x component)

    #Define data structures
    RobotOffset = namedtuple('Offsets',['time', 'frame', 'LO','AO'])
    robot_offsets = []

    #-- Swap coordinates from N,E to E,N
    pos_x = [r.y for r in gps_positions]
    pos_y = [r.x for r in gps_positions]
    gt_x = [g.y for g in gt_positions]
    gt_y = [g.x for g in gt_positions]

    #-- Sync and interpolate gps position to image frames
    pos_time = [r.time for r in gps_positions]
    img_time = [im_m.time for im_m in img_meta]
    pos_upsampled_x, pos_upsampled_y = interpolate_position_to_new_time(pos_time, pos_x, pos_y, img_time, interpolation_mode = 'cubic')

    #-- Preprocess ground truth data
    gt_x,gt_y = align_gt_direction(gt_x, gt_y,robot_x = pos_x, robot_y = pos_y)
    gt_upsampled_x,gt_upsampled_y = interpolate_position_to_new_time(pos_time = np.arange(0,len(gt_x)),pos_x = gt_x,pos_y = gt_y,new_time = np.arange(0,len(gt_x),np.float(len(gt_x))/np.float(len(img_time))))

    print("Processing ", len(img_time), "frames from ", rec_prefix)
    rx_list,ry_list, r_yaw_list = [],[],[]

    #-- Compute GT-robot offsets per image frame

    for ind,im_m in enumerate(img_meta):
        try:
            gpsx = pos_upsampled_x[ind]
            gpsy = pos_upsampled_y[ind]

            #Compute movement direction from GPS points
            gps_point, gps_vec = line_fit_from_points(ind,pos_upsampled_x,pos_upsampled_y,forward_window = smoothing_windows[0], backward_window = smoothing_windows[1])

            #Additional correction between map and robot coordinates
            correction_len = 0.1
            correction_dir = np.abs(np.cross(gps_vec,[0,0,1]))*np.array([1,-1,1]) #correction should be in southeast direction
            correction_EN = correction_len*correction_dir[0:2]
            gpsx, gpsy = [gpsx,gpsy] - correction_EN
            
            #Compute angle
            r_vec = gps_vec
            yaw_angle = angle_between_vectors([1,0],r_vec)

            #Transform position from GPS antenna to robot 
            T_gps_to_map = ht.create_transformation_matrix(r = [0,0,yaw_angle], t = [gpsx,gpsy,0])
            robot_origo_in_GPS_frame = robot_to_gps_translation
            r_pos = ht.transform_xyz_point(T = T_gps_to_map,point = robot_origo_in_GPS_frame)
            rx,ry = r_pos[0],r_pos[1]
            
            #find closest GT point and compute GT direction
            closest_point_ind = closest_point(rx,ry,gt_upsampled_x,gt_upsampled_y)
            gt_line_point, gt_line_vec = line_fit_from_points(closest_point_ind, gt_upsampled_x,gt_upsampled_y,forward_window = 50, backward_window = 50) #naive way - point to point
            
            #Compute lateral and angular offset
            angular_offset = angle_between_lines(gt_line_vec,r_vec)
            lateral_offset = signed_distance_point_to_line([rx,ry],gt_line_point,gt_line_vec)
            
            #Append to list for plotting and saving
            rx_list.append(rx)
            ry_list.append(ry)

            r_yaw = angle_between_lines(-np.array([1,0]),r_vec)
            r_yaw_list.append(r_yaw)
            
            if debug:
                if angular_offset < 100:#all
                    if(ind%50)== 0:
                        plt.figure(1)
                        plt.plot(pos_x,pos_y,'b+')
                        plt.plot(pos_upsampled_x[ind - smoothing_windows[1] : ind + smoothing_windows[0]+1], pos_upsampled_y[ind - smoothing_windows[1] : ind + smoothing_windows[0]+1], 'b.')
                        plt.plot(gpsx,gpsy,'m+')
                        plt.plot([gpsx,gpsx+gps_vec[0]],[gpsy,gpsy+gps_vec[1]],'m-')
                        #Plot gt and gps/robot vectors step by step
                        plt.plot(rx_list,ry_list,'r.')
                        plt.plot([rx,rx+r_vec[0]],[ry,ry+r_vec[1]],'r-')
                        plt.plot(gt_x,gt_y,'go')
                        plt.plot(gt_upsampled_x,gt_upsampled_y,'g.')
                        #plt.plot(gt_line_point[0],gt_line_point[1],'go')
                        plt.plot([gt_line_point[0],gt_line_point[0]+gt_line_vec[0]],[gt_line_point[1],gt_line_point[1]+gt_line_vec[1]],'g-')
                        plt.xlim([rx-5,rx+5])
                        plt.ylim([ry-5,ry+5])
                        plt.title('angular_offset '+  '%.3f' % angular_offset + ', lateral_offset ' + '%.3f' % lateral_offset)
                        plt.show()
        except IndexError:
            print("Index error, frame number ", ind)
            break
        ro = RobotOffset(time = im_m.time, frame = int(im_m.frame_num), LO = lateral_offset, AO = angular_offset)
        robot_offsets.append(ro)
    
    # Write resulting offsets to csv file
    offset_file = os.path.join(output_dir, rec_prefix + '_offsets.csv')
    print('Writing ground truth positions to '+ offset_file)
    write_namedtuples_to_csv(offset_file,robot_offsets)

    if visualize:
        lateral_offsets = np.array([ro.LO for ro in robot_offsets])
        angular_offsets = np.array([ro.AO for ro in robot_offsets])
        lateral_offset_mean = np.mean(lateral_offsets[np.abs(lateral_offsets) < 0.25])
        angular_offset_mean = np.mean(angular_offsets[np.abs(angular_offsets) < 0.25])
        plt.figure(1)
        plt.plot(gt_x,gt_y,'g-+')
        plt.plot(gt_upsampled_x,gt_upsampled_y,'g.')
        plt.plot(pos_upsampled_x,pos_upsampled_y,'b.')
        plt.plot(rx_list, ry_list,'r.')
        #plt.show()
        plt.savefig(os.path.join(output_dir, rec_prefix + '_gps_data'))

        plt.figure(2)
        plt.plot(angular_offsets)
        plt.plot(lateral_offsets)
        #plt.plot(r_yaw_list)
        plt.plot(np.repeat(angular_offset_mean,len(angular_offsets)))
        plt.plot(np.repeat(lateral_offset_mean,len(lateral_offsets)))

        #plt.legend(['Angular offset', 'Lateral Offset', 'Movement direction','Mean angular offset: ' + str(angular_offset_mean),'Mean lateral offset: ' + str(lateral_offset_mean)])
        plt.legend(['Angular offset', 'Lateral Offset', 'Mean angular offset: ' + str(angular_offset_mean),'Mean lateral offset: ' + str(lateral_offset_mean)])
        plt.ylim([-1,1])
        #plt.show()
        plt.savefig(os.path.join(output_dir, rec_prefix + '_offsets'))

if __name__ == '__main__':
    '''
    input_dir = os.path.join('/media/marianne/Seagate Expansion Drive/data/Frogn_Dataset/position_data')
    output_dir = os.path.join('/media/marianne/Seagate Expansion Drive/data/Frogn_Dataset/position_data')
    rec_prefix = '20191010_L2_N'
    smoothing_windows = [50,50]#[50,50]
    gt_position_window = [1,1]
    debug = False
    visualize = True
    
    compute_and_save_robot_offsets(input_dir = input_dir, 
    output_dir = output_dir, 
    rec_prefix = rec_prefix, 
    debug = debug, 
    visualize = visualize,
    smoothing_windows = smoothing_windows)
    '''
    main()





