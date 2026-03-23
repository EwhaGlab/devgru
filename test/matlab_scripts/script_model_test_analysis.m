
clear all; close all; clc;

base_dir = '/media/results/devgru/devgru_tester' ;
coll_res_dir = dir( sprintf('%s/*-colldata_test', base_dir) ) ;
noncoll_res_dir = dir( sprintf('%s/*-noncolldata_test', base_dir) ) ;

coll_res_mean = zeros(4,3) ;
noncoll_res_mean = zeros(4,3) ;

for idx = 1:length(coll_res_dir)
    coll_res_path = sprintf('%s/%s', coll_res_dir(idx).folder, coll_res_dir(idx).name) ;
    data_error = load(sprintf('%s/data_error.txt', coll_res_path) ) ;
    coll_res_mean(idx,:) = mean(data_error) ;
    coll_res_std(idx,:) = std(data_error) ;
end

coll_thr = coll_res_mean(4,:) + coll_res_std(4,:)  ;


for idx = 1:length(noncoll_res_dir)
    noncoll_res_path = sprintf('%s/%s', noncoll_res_dir(idx).folder, noncoll_res_dir(idx).name) ;
    data_error = load( sprintf('%s/data_error.txt', noncoll_res_path) ) ;
    noncoll_res_mean(idx,:) = mean(data_error) ;
    noncoll_res_std(idx,:) = std(data_error) ;
end

noncoll_thr = noncoll_res_mean(4,:) + noncoll_res_std(4,:)  ;

bad_coll_idxs = [];
for idx = 1:length(coll_res_dir)
    coll_res_path = sprintf('%s/%s', coll_res_dir(idx).folder, coll_res_dir(idx).name) ;
    data_error = load(sprintf('%s/data_error.txt', coll_res_path) ) ;
    sg_pose_error = data_error(:, 1) ;
    action_pose_error = data_error(:, 2) ;
    bad_sg_pose_idx = find(sg_pose_error > coll_thr(1) ) ;
    bad_action_pose_idx = find(action_pose_error > coll_thr(2) ) ;
    bad_idx = union( bad_sg_pose_idx, bad_action_pose_idx ) ;
    bad_coll_idxs{idx} = bad_idx ;
end


bad_noncoll_idxs = [];
for idx = 1:length(noncoll_res_dir)
    noncoll_res_path = sprintf('%s/%s', noncoll_res_dir(idx).folder, noncoll_res_dir(idx).name) ;
    data_error = load(sprintf('%s/data_error.txt', noncoll_res_path) ) ;
    sg_pose_error = data_error(:, 1) ;
    action_pose_error = data_error(:, 2) ;
    bad_sg_pose_idx = find(sg_pose_error > noncoll_thr(1) ) ;
    bad_action_pose_idx = find(action_pose_error > noncoll_thr(2) ) ;
    bad_idx = union( bad_sg_pose_idx, bad_action_pose_idx ) ;
    bad_noncoll_idxs{idx} = bad_idx ;
end


% visualize
res_idx = 1 ; 
bad_coll_idx_list = bad_coll_idxs{res_idx} ;

for ii = 1:length(bad_coll_idx_list)
    data_idx = bad_coll_idx_list(ii) ;
    fig_name = sprintf('%s/%s/data%05d.png', ...
        coll_res_dir(res_idx).folder, coll_res_dir(res_idx).name, data_idx) ;
    imshow(fig_name) ;
    pause ;
end

res_idx = 1 ; 
bad_noncoll_idx_list = bad_noncoll_idxs{res_idx} ;

for ii = 1:length(bad_noncoll_idx_list)
    data_idx = bad_noncoll_idx_list(ii) ;
    fig_name = sprintf('%s/%s/data%05d.png', noncoll_res_dir(res_idx).folder, ...
        noncoll_res_dir(res_idx).name, data_idx) ;
    imshow(fig_name) ;
    pause ;
end


