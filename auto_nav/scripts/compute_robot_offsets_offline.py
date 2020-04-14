import os
import matplotlib.pyplot as plt
from namedtuples_csv import read_namedtuples_from_csv
from scipy.interpolate import interp1d

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

    for r in robot_positions:
        print(r)
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
    
    #--Interpolate robot position
    pos_t = [r.time for r in robot_positions]
    pos_x = [r.x for r in robot_positions]
    pos_y = [r.y for r in robot_positions]
    f_interp_x = interp1d(x=pos_t, y = pos_x,fill_value = 'extrapolate')
    f_interp_y = interp1d(x=pos_t, y = pos_y,fill_value = 'extrapolate')

    img_t = [im_m.time for im_m in img_meta]
    #robot positions at image timestamps
    r_pos_upsampled_x = f_interp_x(img_t)
    r_pos_upsampled_y = f_interp_y(img_t)
    
    #plot upsampled vs original
    print('pos',pos_x[:10])
    print('rpos',r_pos_upsampled_x[:10])
    plt.figure()
    plt.plot(pos_y,pos_x,'r+')
    plt.plot(r_pos_upsampled_y,r_pos_upsampled_x,'b.')
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





