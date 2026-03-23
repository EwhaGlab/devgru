% script  trajectory data collection

clear all; close all; clc;

data_dir = '/media/data/mydata/former_datasets/colldata/colldata-all/bag_2025-11-21-15-20-52'
sync_metadata_dir='/media/results/navdata_collector/colldata/processed/T9_coll/coll_2025-11-21-15-20/bag_2025-11-21-15-20-52/synced'
topomap_name = 'T9' ;
topomap_root_dir = sprintf('/home/hankm/python_ws/viznav/depth-nav/deployment/topomaps/%s',topomap_name) ;
test_dir = '/home/hankm/python_ws/viznav/depth-nav/test';
test_inout_dir = '/home/hankm/python_ws/viznav/depth-nav/test/data';

setenv('LD_PRELOAD', '');
% 2) (Optional but helpful) avoid crazy memory arenas
setenv('MALLOC_ARENA_MAX', '1');

rgb_folder = sprintf('%s/rgb*.png', topomap_root_dir)  ;
D      = dir(rgb_folder);
names  = {D.name} ;                 % cell array of ALL file names
S = lower(string(names));                   % string array for endsWith

for idx=1:length(S)
    s= S{idx} ;
    digits_str = regexprep(s, '\D', '') ;   % -> '0000'
    topoimg_idxstr{idx} = digits_str ;
end

% load topomap
topo_odom_file = sprintf('%s/topo_odom.txt', topomap_root_dir) ;
topo_m2b_file = sprintf('%s/topo_tf_m2b.txt', topomap_root_dir) ;
topo_m2o_file = sprintf('%s/topo_tf_m2o.txt', topomap_root_dir) ;
[topo_odom_raw, topo_odom_xy, topo_o1Hb ] = load_pose_data(topo_odom_file) ; 
[topo_m2b_raw, topo_m2b_xy, topo_m1Hb ] = load_pose_data(topo_m2b_file) ; 
[topo_m2o_raw, topo_m2o_xy, topo_m1Ho ] = load_pose_data(topo_m2o_file) ; 

[num_node, ~] = size(topo_m2b_raw ) ;

nav_m2b_file = sprintf('%s/sync_tf_m2b.txt', sync_metadata_dir) ;
nav_m2o_file = sprintf('%s/sync_tf_m2b.txt', sync_metadata_dir) ;
nav_odom_file = sprintf('%s/sync_odom.txt', sync_metadata_dir) ;

[nav_odom_raw, nav_odom_xy, nav_o2Hb ] = load_pose_data(nav_odom_file) ; 
[nav_m2b_raw,  nav_m2b_xy,  nav_m2Hb ] = load_pose_data(nav_m2b_file) ; 
[nav_m2o_raw,  nav_m2o_xy,  nav_m2Ho ] = load_pose_data(nav_m2o_file) ; 

[num_data, c] = size(nav_odom_raw) ;

str_split = split(sync_metadata_dir, '/') ;
navtime_id = str_split{end-1} ;
% colldata_dir = sprintf('%s/%s', out_base_dir, navtime_id) ; 
% if isdir(colldata_dir)
%     %rmdir(colldata_dir, 's') ;
%     error('%s  dir exist!! \n Make sure to remove this dir before processing the data \n', colldata_dir) ;
% end


context_len = 5;
eta = 1.0 ;             % Repulsive potential scaling factor (η)
rho0 = 1.2 / 0.05 ;     % Influence distance (ρ0, # grids )
obs_thr = 99 ;
FPS = 10 ;
v_max = 0.3 ; % 0.3 m/s
w_max = 0.6 ; % 0.3 rad/s
ws = 3 ;      % waypoint spacing / context spacing (spacing btwn odom poses)

resolution = 0.05 ;
map_size_m = 10.0 ;
robot_radius_m = 0.25 ;
inflation_radius_m = 1.0 ;
max_range_m = 25;

cm_params = struct('resolution', resolution, 'map_size_m', map_size_m, 'robot_radius_m', robot_radius_m, ...
    'inflation_radius_m', inflation_radius_m, 'max_range_m', max_range_m ) ;

%o1Ho2 = topo_o1Hb(:,:,1) * inv(o2Hb(:,:,1))  ;

% load waypoints
navdata = load(sprintf('%s/sync_navdata.txt', sync_metadata_dir)) ;

%% === Step 2: Compute ρ(q) = distance to the nearest obstacle ===
% bwdist() returns the Euclidean distance in pixels from each free cell
% to the nearest obstacle cell.

map_size_px = map_size_m / resolution ; 
rx = map_size_px / 2 ;  % robot position (center of the map)
ry = rx ;

% load subgoal
subgoals = load(sprintf('%s/sync_rel_subgoals.txt', sync_metadata_dir)) ;
data_cnt = 0;

coll_attention_idxs_init = find( navdata(:, 5) ) ;

% add one pads @ the beginning of "joy_on"  
v = navdata(:, 5)' ;
joy_start_idx = find(diff([0 v] == 1) ) ;
k = 16; % 1.6 sec ahead of collsion. (10 FPS)
J = joy_start_idx(:) + (-k:-1);          % size: [num_starts x k]
J = J(J >= 1 & J <= numel(v));       % clip to bounds
v2 = v;
v2(J) = 1;

coll_attention_idxs = find(v2) ;

wd = cd ;

for tmp_idx = 1:length(coll_attention_idxs) %num_data-1
    data_idx = coll_attention_idxs(tmp_idx) ;
    %% === 1. Read metadata == %%
    navdata_line = navdata(data_idx,:) ;
    %joy_on = waypt_line(5) ;
    sg_idx = navdata_line(6) + 1 ;  % +1 matlab conv
    old_sg_px = subgoals(data_idx,5:6) /resolution ; % old sg_px

    o1Hb_curr  = topo_o1Hb(:,:,sg_idx) ; 
    m1Ho1_curr = topo_m1Ho(:,:,sg_idx) ; % odom wrt map1 @ curr sgidx
    m1Hsg_curr = topo_m1Hb(:,:,sg_idx) ; % sg pose wrt map1 @ curr sgidx
    
    o2Hb_curr  = nav_o2Hb(:,:,data_idx) ;
    m2Ho2_curr = nav_m2Ho(:,:,data_idx) ;
    m2Hb_curr  = nav_m2Hb(:,:,data_idx) ;

    %bHsg = inv( m2Hb_curr ) * m1Hsg_curr ;
    %sgs_corrected_px = bHsg(1:2,4)' / resolution ;
    
    % draw future sgs
    sgs_xyzq_corrected_px = zeros(num_node, 7) ; % x y z qw qx qy qz
    for ii=1:num_node
        next_sg_idx = min(ii, num_node) ;
        bHsg = inv( m2Hb_curr ) * topo_m1Hb(:,:,next_sg_idx) ; 
        sgs_xyzq_corrected_px(ii,1:2) = bHsg(1:2,4)' / resolution ;
        sgs_xyzq_corrected_px(ii,4:end) = htm_to_quat( bHsg ) ;
    end
    [num_sgs, ~] = size(sgs_xyzq_corrected_px) ;
    assert (num_sgs > 0) ;

    img_idx = data_idx - 1 ;
    % We go ahead do the data collection if the joy msg is on
    depth_img = double( imread(sprintf('%s/depth%05d.png', sync_metadata_dir, img_idx)) ) / 1000 ;
    rgb_img = imread(sprintf('%s/rgb%05d.png', sync_metadata_dir, img_idx )) ;
    waypt_traj_px = reshape( navdata_line([14,15, 18,19, 22,23, 26,27, 31,32]), 2, 5)' / resolution ; 
    tgt_wx_px = waypt_traj_px(2,1) ; % 2nd wpt
    tgt_wy_px = waypt_traj_px(2,2)  ; % 
    attr_ang_rad = atan2(tgt_wy_px, tgt_wx_px) ;

    scandata = load(sprintf('%s/scan%05d.txt', sync_metadata_dir, img_idx) ) ;
    ranges = scandata(:,2) ;
    angles = scandata(:,1) ;


    context_idxs = [data_idx-context_len*ws:ws:data_idx ] ;
    wHr = nav_o2Hb(:,:,context_idxs) ;    
    rcHr = zeros(4,4,context_len) ; 
    rcHr(:,:,end) = wHr(:,:,end) ;

    %% === 0. Save context index ===== %%
    cd(test_inout_dir) ;
    system('rm *.*') ;
    cd(wd) ;
    fid = fopen( sprintf('%s/context_index.txt', test_inout_dir), 'w') ;
    fprintf(fid, '%d ', context_idxs); fprintf(fid, '\n');
    fclose(fid) ;

    %% === 1. Save RGB, Depth pose context ==== %%
    % Pose context
    
    obs_len = context_len + 1;
    wHr = nav_o2Hb(:,:,context_idxs ) ;
    pose_context_m = zeros(obs_len, 7);  % xyz quat
    fid = fopen( sprintf('%s/pose_context_m.txt', test_inout_dir) , 'w') ;
    for ii=1:obs_len
        r0Hr(:,:,ii) = inv( o2Hb_curr ) * wHr(:,:,ii) ;
        q = htm_to_quat(r0Hr(:,:,ii)) ;
        [x y z rol pit yaw] = htm_to_xyzypr( r0Hr(:,:,ii) ) ; 
        pose_context_m(ii,:) = [x; y; z; q(:) ]' ;
        fprintf( fid, '%6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\n', [ x y 0 q(:)']' ) ; % x y z qw qx qy qz
    end
    fclose(fid) ;


    pose_context_px = [pose_context_m(:,1:3) / resolution  pose_context_m(:,4:end) ] ;

    % Depth and RGB context
    for cidx = 1 : length(context_idxs)
        src_rgb = sprintf('%s/rgb%05d.png', sync_metadata_dir, context_idxs(cidx) )  ;
        src_dep = sprintf('%s/depth%05d.png', sync_metadata_dir, context_idxs(cidx) )  ;
        %depth_context(:,:,cidx) = imread(src_depth) ;
        copyfile(src_rgb, sprintf('%s/', test_inout_dir) ) ;
        copyfile(src_dep, sprintf('%s/', test_inout_dir) ) ;
    end
    
    %% === 2. Get/ save Subgoal RGB, Depth, pose  ==== %%
    % SG RGB & depth
    sgimg_idx = topoimg_idxstr{sg_idx} ;
    rgb_sg_file =  sprintf('%s/rgb%s.png',topomap_root_dir, sgimg_idx) ;
    depth_sg_file =  sprintf('%s/depth%s.png',topomap_root_dir, sgimg_idx) ;
    copyfile(rgb_sg_file, sprintf('%s/rgb_sg.png', test_inout_dir)) ;
    copyfile(depth_sg_file, sprintf('%s/depth_sg.png', test_inout_dir));

    % Old SG pose 
    fid = fopen( sprintf('%s/old_subgoal_m.txt', test_inout_dir), 'w') ; % old; 
    fprintf(fid, '%6.4f %6.4f 0 0\n', old_sg_px(1) * resolution, old_sg_px(2) * resolution) ;
    fclose(fid) ;

    % GT new SG pose
    new_sg_idx = sg_idx; %find(dist_to_sg_cands == min(dist_to_sg_cands)) ;
    new_sg_xyzq_px = sgs_xyzq_corrected_px(new_sg_idx, :) ;
    new_sg_xyzq_m  = new_sg_xyzq_px ;
    new_sg_xyzq_m(1:3)  = new_sg_xyzq_px(1:3) * resolution ;

    % dx = gx - rx ;   % wpt1  should heading to wpt2
    % dy = gy - ry ;
    % theta = atan2(dy, dx) ;  % radians
    % half_theta = theta / 2 ;
    % q = [cos(half_theta), zeros(size(theta)), zeros(size(theta)), sin(half_theta)] ;
    % q = q / norm(q) ; % subgoal orient

    q = new_sg_xyzq_px(4:end) ;
    fid = fopen( sprintf('%s/new_subgoal_m.txt', test_inout_dir), 'w') ; % corrected
    fprintf(fid, '%6.4f %6.4f %6.4f %6.4f\n', new_sg_xyzq_px(1)*resolution, new_sg_xyzq_px(2)*resolution, q(1), q(4) ) ;
    fclose(fid) ;
    %%===========================================================================%%
    %% 3. run model step
    cd('/home/hankm/python_ws/viznav/depth-nav/test') ;
    python_bin = '/home/hankm/miniconda3/envs/vint_deployment/bin/python';
    script     = '/home/hankm/python_ws/viznav/depth-nav/test/test_colldata_step.py';
    cmd = sprintf('%s %s', python_bin, script);
    [status, cmdout] = system(cmd);

    cd(wd) ;
    %%===========================================================================%%

    %% 4. load results
    pred_pose_diff_m = load(sprintf('%s/pred_pose_diff.txt', test_inout_dir)) ;
    pred_waypoints_m = load(sprintf('%s/pred_waypoints.txt', test_inout_dir)) ;
    pred_collprob  = load(sprintf('%s/pred_coll.txt', test_inout_dir)) ;

    %% === Step 3: display data (To inspect them)  === %%  
    
    %  Build costmap %
    [costmap_i8, costmap_u8] = build_costmap( ranges, angles, cm_params) ;
    ukn_idx = find(costmap_i8 == -1) ;
    obstacle_mask = costmap_i8 >= obs_thr;
    rho = bwdist(obstacle_mask) ; %resolution;  % ρ(q) in meters

    fig = figure(1); clf;
    fig.Position = [2800, 300, 1600, 780] ;
    tmain = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tmain, sprintf('Bag ID: %s', navtime_id),'FontWeight','bold','FontSize',14, 'Interpreter','none');
    
    % left column: 2x2 img grid
    leftgrid = tiledlayout(tmain, 2,2, 'TileSpacing', 'compact', 'Padding', 'compact');
    leftgrid.Layout.Tile = 1;

    nexttile(leftgrid, 1);
    imshow(rgb_img) ; % obs rgb 
    title('Observed RGB [{\color{green}▶ }]') ;

    nexttile(leftgrid,2);  % subgoal
    sgimg_idx = topoimg_idxstr{sg_idx} ;
    rgb_sg_file =  sprintf('%s/rgb%s.png',topomap_root_dir, sgimg_idx) ;
    depth_sg_file =  sprintf('%s/depth%s.png',topomap_root_dir, sgimg_idx) ;
    rgb_sg = imread(rgb_sg_file) ;
    depth_sg = double( imread(depth_sg_file) ) / 1000 ;
    imshow(rgb_sg)  ;
    title('SG (corrected) RGB [{\color{red}●}]')

    nexttile(leftgrid,3);  % obs depth
    draw_depth_colormap( depth_img, 0.05, 3.0, 0.5 ) ;

    ax4=nexttile(leftgrid,4);  % sg depth
    draw_depth_colormap( depth_sg, 0.05, 3.0, 0.5 ) ;

    nexttile(tmain, 2); 
    imshow( costmap_i8 );  % Show potential field as a heatmap
    hold on;
    
    drawPoseSeq(sgs_xyzq_corrected_px, rx, ry, 2) ; % draw SLAM SGs
    drawPoseSeq(pose_context_px, rx, ry, 1, 'cs', 'c') ; % draw pose context 
    plot( rx + waypt_traj_px(:,1) * 100 , ry + waypt_traj_px(:,2) * 100, 'ys', 'MarkerFaceColor','y') ;
    plot( rx, ry, 'g>', 'MarkerSize', 15, 'MarkerFaceColor', 'g' ) ;
    plot( rx + old_sg_px(1), rx + old_sg_px(2),  'mo', 'MarkerSize', 8 , 'MarkerFaceColor', 'm') ;

    colorbar; 
    title( sprintf('Data idx: %d / %d ,  # Collected Data: %d ',data_idx, num_data, data_cnt)) ;
    hold on;


    % show pred waypoint
    pred_waypoints_px = pred_waypoints_m(:,1:2) / resolution ;
    plot( pred_waypoints_px(:,1) + rx, pred_waypoints_px(:,2) + ry, 'gs', 'MarkerSize', 4, 'MarkerFaceColor', 'g' ) ;

    % show corr sg pose
    pred_pose_diff_px = pred_pose_diff_m(1:2) / resolution ;
    plot( pred_pose_diff_px(1) + rx, pred_pose_diff_px(2) + ry, 'p', 'MarkerSize', 12, 'MarkerFaceColor', [0.3, 0, 0.51] ) ;

    if pred_collprob > 0.5
        status_str = 'COLLISION';
        box_color = [1 0 0];     % red
    else
        status_str = 'SAFE';
        box_color = [0 0.6 0];   % green
    end
annotation('textbox', [0.08 0.90 0.20 0.06], ...
    'String', status_str, ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'Color', 'white', ...           % Text color
    'BackgroundColor', box_color, ... % Box color
    'EdgeColor', 'none', ...
    'Margin', 6);


    %    dist_to_sg_cands = sqrt( (sgs_px_cands(:,1) - gx).^2 + (sgs_px_cands(:,2) - gy).^2 ) ;
    new_sg_idx = sg_idx; %find(dist_to_sg_cands == min(dist_to_sg_cands)) ;
    new_sg_xyzq_px = sgs_xyzq_corrected_px(new_sg_idx, :) ;
    new_sg_xyzq_m  = new_sg_xyzq_px ;
    new_sg_xyzq_m(1:3)  = new_sg_xyzq_m(1:3) * resolution ;
    plot( new_sg_xyzq_px(1) + rx, new_sg_xyzq_px(2) + ry, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'm' ) ;

    lgd = legend('Pred Waypts', 'SG(SLAM)', 'Robot', 'SG(pred)', 'SG(corr)', 'Location', 'NW')
    lgd.Color = [0.8 0.8 0.8];
    set(gca,'YDir','normal');

    % %% == 6. Data collection from expert's choice == %%
    % [gx, gy] = ginput(1) ;
    % if (gx <0 | gx > map_size_px | gy < 0 | gy > map_size_px)
    %     continue;  % if the not qualitifed to be a collision data. Don't bother
    % end

    set(gcf, 'pointer', 'crosshair'); 
    set(gcf, 'pointer', 'arrow');
    P0_m = [rx, ry] * resolution ; 
    P2_m = [new_sg_xyzq_m(1), new_sg_xyzq_m(2)] + P0_m ;

    % Compute the corrected waypoints
    P0m_xyzq = [0 0 0 1 0 0 0] ;
    P2m_xyzq = [new_sg_xyzq_px(1:3) * resolution, new_sg_xyzq_px(4:end) ] ;
    
    P1m_xyzq = makeP1Between(P0m_xyzq, P2m_xyzq, alpha=0.5) ;
    out_waypts_m = planKinodynamicPath(P0m_xyzq, P1m_xyzq, P2m_xyzq, v_max, w_max, FPS) ;
    out_waypts_m = out_waypts_m(1:ws:end,:)  ; % waypoint spacing 3
    out_waypts_m = out_waypts_m(2:6,:) ;    % next 5 steps
    out_waypts_px = out_waypts_m ; out_waypts_px(:,1:2) = out_waypts_px(:,1:2) / resolution + map_size_px/2  ;

    %[out_waypts_px, target_waypoint_px] = sample_corrected_waypoints(P0_xyzq, P2_xyzq, resolution, fps, ws) ;
    %plot( out_waypts(:,1) + rx, out_waypts(:,2) + rx, 'gs', 'MarkerFaceColor', 'g') ;
    drawPoses2D(out_waypts_px) ; 

    new_sg_idx = new_sg_idx-1;
    fn = names{new_sg_idx} ;
    global_sg_idx = str2double(regexp(fn, '\d+', 'match', 'once')) ;
    subgoal_idxs = [new_sg_idx, global_sg_idx] ;  % sg idx of /topomap folder, global sg idx in /synced folder
    context_idxs = [data_idx-context_len*ws:ws:data_idx ] ;
    %nc = length(context_idxs) ;
    % (1) save prev robot pose 
    % robot pose w.r.t curr robot pose

    data_cnt = data_cnt + 1 ;

    outfig = sprintf('/media/results/devgru/fig%05d.png', data_cnt) ;
    saveas(gcf, outfig);
end