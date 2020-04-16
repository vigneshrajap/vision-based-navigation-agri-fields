import os
import matplotlib.pyplot as plt
import numpy as np
from namedtuples_csv import read_namedtuples_from_csv
from scipy.interpolate import interp1d
from geometric_utilities import direction_sign, angle_between_vectors, signed_distance_point_to_line, closest_point, line_to_next_point

'''
#extract lists with time and positions
def robot_pos_interpolation_function(robot_pos)
    #input: robot_pos - named tuple
    pos_time = [r.time for r in robot_positions]
    pos_x = [r.x for x in robot_positions]
    pos_y = [r.y for y in robot_positions]

    #f = interp1d(x, y)

return interp_function
'''

if __name__ == '__main__':
    input_dir = os.path.join('.','output')
    rec_prefix = '20191010_L3_S_morning_slaloam'

    # Read converted positions and timestamps
    gt_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_gt_pos.csv'),'GTPos')
    robot_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_robot_pos_and_timestamps.csv'), 'RobotPos')
    img_meta = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_image_timestamps.csv'), 'ImageMeta')

    '''
    #debug:
    for r in robot_positions:
        print(r)
    for gt in gt_positions:
        print(gt)
    for i in img_meta:
        print(i)
    '''
    #NB: 2 more frames when reading time stamps than what was saved with rqt image viewer. Should implement image saving in the conversion script (or make a separate script.)
    
    #-- Interpolate robot position
    pos_t = [r.time for r in robot_positions]
    pos_x = [r.x for r in robot_positions]
    pos_y = [r.y for r in robot_positions]
    f_interp_x = interp1d(x=pos_t, y = pos_x,fill_value = 'extrapolate')
    f_interp_y = interp1d(x=pos_t, y = pos_y,fill_value = 'extrapolate')

    img_t = [im_m.time for im_m in img_meta]
    #robot positions at image timestamps
    r_pos_upsampled_x = f_interp_x(img_t)
    r_pos_upsampled_y = f_interp_y(img_t)

    #-- Get ground truth segments
    gt_x = [g.x for g in gt_positions]
    gt_y = [g.y for g in gt_positions]
    plt.plot(gt_x, gt_y,'mx-')
    #plt.show()

    #Align ground truth with robot direction
    gt_dir_vec = [gt_x[-1]-gt_x[0],gt_y[-1]-gt_y[0]]
    robot_dir_vec = [pos_x[-1]-pos_x[0],pos_y[-1]-pos_y[0]]
    gt_direction = direction_sign(gt_dir_vec/np.linalg.norm(gt_dir_vec),robot_dir_vec/np.linalg.norm(robot_dir_vec))
    if gt_direction < 0:
        gt_x = list(reversed(gt_x))
        gt_y = list(reversed(gt_y))

    for i, (rx, ry) in enumerate(zip(r_pos_upsampled_x, r_pos_upsampled_y)):
        #find closest GT point
        closest_point_ind = closest_point(rx,ry,gt_x,gt_y)
        try:
            #get ground truth segment (line)
            gt_line_point, gt_line_vec = line_to_next_point(closest_point_ind, gt_x,gt_y) #naive way - point to point
            #get robot vector
            r_point,r_vec = line_to_next_point(i,r_pos_upsampled_x, r_pos_upsampled_y,step=5)
        except IndexError:
            break
        #print('r_point','r_vec',r_point, r_vec)
        #print(r_pos_upsampled_x[5],r_pos_upsampled_x[10])
        #compute angle
        angular_offset = angle_between_vectors(gt_line_vec,r_vec)
        #compute lateral offset
        lateral_offset = signed_distance_point_to_line([rx,ry],gt_line_point,gt_line_vec)

        #debug
        if(i%50)== 0:
            #plot upsampled vs original
            plt.figure(1)
            plt.plot(pos_x,pos_y,'r+')
            plt.plot(r_pos_upsampled_x,r_pos_upsampled_y,'b.-')
            #Plot gt and robot vectors step by step
            plt.plot(rx,ry,'r.')
            plt.plot([rx,rx+r_vec[0]],[ry,ry+r_vec[1]],'r-')
            plt.plot(gt_line_point[0],gt_line_point[1],'go')
            plt.plot([gt_line_point[0],gt_line_point[0]+gt_line_vec[0]],[gt_line_point[1],gt_line_point[1]+gt_line_vec[1]],'g-')
            plt.xlim([rx-2,rx+2])
            plt.ylim([ry-2,ry+2])
            plt.title('angular_offset '+  '%.3f' % angular_offset + ', lateral_offset ' + '%.3f' % lateral_offset)
            plt.show()
        #print('ao', angular_offset, 'lo', lateral_offset)
plt.show()

#### TODO after Easter: #########
    #Linear segments from GT

    #Per image frame, compute offsets

        #interpolate robot position

        #-lateral offset
            # - compared to GT line segments. Be aware of sign and north/south travel direction
        #-angular offsets
            # - use a moving window around interpolated robot position

        #Output: save as csv file dt, frame, LO AO





