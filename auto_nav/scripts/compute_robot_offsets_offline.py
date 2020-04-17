import os
import matplotlib.pyplot as plt
import numpy as np
from namedtuples_csv import read_namedtuples_from_csv, write_namedtuples_to_csv
from scipy.interpolate import interp1d
from geometric_utilities import direction_sign, angle_between_vectors, signed_distance_point_to_line, closest_point, line_to_next_point, line_fit_from_points

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

def interpolate_position_to_new_time(pos_time, pos_x, pos_y, new_time):  
    f_interp_x = interp1d(x=pos_time, y = pos_x, fill_value = 'extrapolate', kind = 'linear')
    f_interp_y = interp1d(x=pos_time, y = pos_y, fill_value = 'extrapolate', kind = 'linear')
    
    pos_interp_x = f_interp_x(new_time)
    pos_interp_y = f_interp_y(new_time)
    return pos_interp_x, pos_interp_y

def compute_offset(pos_ind, r_pos_x, r_pos_y, gt_x, gt_y, debug = False):
    rx = r_pos_x[pos_ind]
    ry = r_pos_y[pos_ind]

    #find closest GT point
    closest_point_ind = closest_point(rx,ry,gt_x,gt_y)
   
    gt_line_point, gt_line_vec = line_to_next_point(closest_point_ind, gt_x,gt_y) #naive way - point to point
    r_point, r_vec = line_fit_from_points(pos_ind,r_pos_x,r_pos_y,forward_window = 25, backward_window = 25)

    angular_offset = angle_between_vectors(gt_line_vec,r_vec)
    lateral_offset = signed_distance_point_to_line([rx,ry],gt_line_point,gt_line_vec)

    if debug:
        if(pos_ind%50)== 0:
            #plot upsampled vs original
            plt.figure(1)
            plt.plot(r_pos_x,r_pos_y,'b.-')
            #Plot gt and robot vectors step by step
            plt.plot(rx,ry,'r.')
            plt.plot([rx,rx+r_vec[0]],[ry,ry+r_vec[1]],'r-')
            plt.plot(gt_line_point[0],gt_line_point[1],'go')
            plt.plot([gt_line_point[0],gt_line_point[0]+gt_line_vec[0]],[gt_line_point[1],gt_line_point[1]+gt_line_vec[1]],'g-')
            plt.xlim([rx-2,rx+2])
            plt.ylim([ry-2,ry+2])
            plt.title('angular_offset '+  '%.3f' % angular_offset + ', lateral_offset ' + '%.3f' % lateral_offset)
            plt.show()

    return lateral_offset, angular_offset

def compute_and_save_robot_offsets(input_dir = os.path.join('.','output'), 
    output_dir = os.path.join('.','output'), 
    rec_prefix = '20191010_L3_S_morning_slaloam', 
    debug = False, 
    visualize = False):
    # Read converted positions and timestamps
    gt_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_gt_pos.csv'),'GTPos')
    robot_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_robot_pos_and_timestamps.csv'), 'RobotPos')
    img_meta = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_image_timestamps.csv'), 'ImageMeta')

    #NB: 2 more frames when reading time stamps than what was saved with rqt image viewer. Should implement image saving in the conversion script (or make a separate script.)

    #Define data structures
    RobotOffset = namedtuple('Offsets',['time', 'frame', 'LO','AO'])
    robot_offsets = []
   
    #-- Sync robot position with image frames
    pos_time = [r.time for r in robot_positions]
    pos_x = [r.x for r in robot_positions]
    pos_y = [r.y for r in robot_positions]
    img_time = [im_m.time for im_m in img_meta]
    r_pos_upsampled_x, r_pos_upsampled_y = interpolate_position_to_new_time(pos_time, pos_x, pos_y, img_time)

    #-- Preprocess ground truth data
    gt_x = [g.x for g in gt_positions]
    gt_y = [g.y for g in gt_positions]
    gt_x,gt_y = align_gt_direction(gt_x, gt_y,robot_x = pos_x, robot_y = pos_y)

    #-- Compute GT-robot offsets
    for ind,im_m in enumerate(img_meta):
        try:
            lateral_offset, angular_offset = compute_offset(pos_ind = ind, r_pos_x = r_pos_upsampled_x, r_pos_y = r_pos_upsampled_y, gt_x = gt_x, gt_y = gt_y, debug = debug)
        except IndexError:
            break
        ro = RobotOffset(time = im_m.time, frame = int(im_m.frame_num), LO = lateral_offset, AO = angular_offset)
        robot_offsets.append(ro)
    
    # Write resulting offsets to csv file
    offset_file = os.path.join(output_dir, rec_prefix + '_offsets.csv')
    print('Writing ground truth positions to '+ offset_file)
    write_namedtuples_to_csv(offset_file,robot_offsets)

    if visualize:
        plt.figure(1)
        plt.plot(gt_x,gt_y,'b-+')
        plt.plot(r_pos_upsampled_x,r_pos_upsampled_y,'r.')
        plt.show()

        plt.figure(2)
        plt.plot([ro.AO for ro in robot_offsets])
        plt.plot([ro.LO for ro in robot_offsets])
        plt.legend(['Angular offset', 'Lateral Offset'])
        plt.ylim([-0.5,0.5])
        plt.show()

if __name__ == '__main__':
    input_dir = os.path.join('.','output')
    output_dir = os.path.join('.','output')
    rec_prefix = '20191010_L3_S_morning_slaloam'
    debug = False
    visualize = True

    compute_and_save_robot_offsets(input_dir = input_dir, output_dir = output_dir, rec_prefix = rec_prefix, debug = debug, visualize = visualize)

#### TODO #########
    # linear fit of GT and robot line





