import os
from namedtuples_csv import read_namedtuples_from_csv



if __name__ == '__main__':
    input_dir = os.path.join('.','output')
    rec_prefix = '20191010_L3_S_morning_slaloam'

    # Read converted positions and timestamps
    gt_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_gt_pos.csv'),'GTPos')
    robot_positions = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_robot_pos_and_timestamps.csv'), 'RobotPos')
    img_meta = read_namedtuples_from_csv(os.path.join(input_dir, rec_prefix + '_image_timestamps.csv'), 'ImageMeta')

    #debug:
    for gt in gt_positions:
        print(gt)
    for r in robot_positions:
        print(r)
    for i in img_meta:
        print(i)

    #NB: 2 more frames when reading time stamps than what was saved with rqt image viewer. Should implement image saving in the conversion script (or make a separate script.)


#### TODO after Easter: #########
    #Linear segments from GT

    #Per image frame, compute offsets

        #interpolate robot position
        #-lateral offset
            # - compared to GT line segments. Be aware of sign and north/south travel direction
        #-angular offsets
            # - use a moving window around interpolated robot position

        #Output: save as csv file dt, frame, LO AO





