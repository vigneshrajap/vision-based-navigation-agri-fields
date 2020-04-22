%% load in file data
gps_fn = '..\Data\row_data\raw_gps.csv';
imu_fn = '..\Data\row_data\raw_imu.csv';
gt_fn = '..\Data\row_data\ground_truth_coordinates.xls';

% Time(Sec),Lat(deg),Long(deg),UTM_x(m),UTM_y(m)
gps = readmatrix(gps_fn);

% Time(Sec),Roll(rad),Pitch(rad),Yaw(rad),A_x(m/s2),A_y(m/s2),A_z(m/s2)
imu = readmatrix(imu_fn);

% Name	Lat(North)	Lon(East)	Lat(fix)	Lon(fix)	Ht	Ht(G)	Codes	HRMS	Date	Time	Solution Type	Map_x	Map_y
[~,sheet_name]=xlsfinfo(gt_fn);
gt_sheet = {};
for k=1:numel(sheet_name)
  gt_sheet{k}=xlsread(gt_fn,sheet_name{k});
end
% select the right xls sheet for this dataset
gt_sheet_n = 2;
gt = gt_sheet{gt_sheet_n};

% % convert lat/lon to utm -> load from file instead (lat/lon gps not enough precision)
% [E,N,utmzone] = deg2utm(gps(:,2),gps(:,3));
% gps_xy = [N,E];
gps_xy = [gps(:,4), gps(:,5)];

% convert gt lat/lon to UTM
[E,N,utmzone] = deg2utm(gt(:,4),gt(:,5));
gt_xy = [N,E];

% make our plots nicer by centring on first gt coord
origin = gt_xy(1,:);
gps_xy = gps_xy - origin;
gt_xy = gt_xy - origin;

ngt = length(gt_xy);
ngps = length(gps_xy);


%% compute the linear segments around each ground truth point (use neighbouring points)
segs = zeros(ngt, 4);
for  i = 2:(ngt-1)
    % ax + by + c = 0
    [segs(i,1), segs(i,2), segs(i,3), segs(i,4)] = orthfit2d(gt_xy(i-1:i+1,1), gt_xy(i-1:i+1,2));
end


%% plot the ground truth dataset with fitted linear segments and robot dataset
figure(1);
hold off;
plot(gt_xy(:,2), gt_xy(:,1), 'k.');
hold on;
plot(gps_xy(:,2), gps_xy(:,1), 'b-');
grid on;
axis equal;
title('Robot pos (UTM) vs ground truth (UTM)');

for i = 1:ngt
    % ax + by + c = 0
    abc = segs(i,1:3);
    a = abc(1); b = abc(2); c = abc(3);
    
    if (a == 0) && (b == 0)
        continue;
    end
    
    if (abs(a) < abs(b))
        % y = ax + b
    else
        % x = ay + b
        y = [min(gt_xy(i-1:i+1,2)), max(gt_xy(i-1:i+1,2))];
        x = -(c + b.*y) ./ a;

        figure(1);
        hold on;
        plot(y, x, 'g-');
    end
end


%% find matching ground truth segment for each robot point, then compute lateral offset and row orientation
lateral_offsets = zeros(ngps,1);
row_orientation = zeros(ngps,1);
for i = 1:ngps
    % matching ground truth segment (closest ground truth point)
    [~, ind] = min(vecnorm(gps_xy(i,:) - gt_xy, 2, 2));
    
    % find line perpendicular to matching ground truth segment
    x = gps_xy(i,1); y = gps_xy(i,2);
    abc1 = segs(ind,1:3);
    a1 = abc1(1); b1 = abc1(2); c1 = abc1(3);
    a2 = b1; b2 = -a1; c2 = -a2 * x - b2 * y;
    abc2 = [a2, b2, c2];
    % find intersection point between ground truth segment and perpendicular line
    % (closest point on ground truth segment to robot)
    int = cross(abc1,abc2);
    
    % orientation of the ground truth segment at this robot point
    row_orientation(i,1) = rad2deg(atan2(b2, a2));
    
    % no intersection
    if (int(3) == 0)
        continue;
    end
    
    intxy = int(1:2) ./ int(3);
    
    % lateral offset is length of vector from robot to intersection
    lateral_offsets(i,1) = norm(gps_xy(i,:) - intxy);
    
    % draw the matching ground truth segment and lateral offset geometry for each robot point
    figure(99);
    hold off;
    yy = [y-1, y+1];
    xx1 = -(c1 + b1.*yy) ./ a1;
    plot(yy, xx1, 'g');
    hold on;
    plot(y, x, 'bo');
    xx2 = -(c2 + b2.*yy) ./ a2;
    plot(yy, xx2, 'r');
    plot(intxy(2), intxy(1), 'kx');
    axis equal;
    
    drawnow;
%     pause;
end


%% draw some results
figure(2);
hold off;
plot(lateral_offsets);
grid on;
title('Lateral offset');
ylabel('m');
xlabel('it');

step = [0, 0; diff(gps_xy)];
movement_dir = rad2deg(atan2(step(:,2), step(:,1)));

figure(3);
hold off;
plot(movement_dir);
hold on;
plot(row_orientation);
grid on;
title('Movement dir and row heading');

figure(4);
hold off;
plot(movement_dir - row_orientation);
grid on;
title('Angular offset');
ylabel('deg');
xlabel('it');
ax = axis;
axis([ax(1) ax(2) -8 8]);
