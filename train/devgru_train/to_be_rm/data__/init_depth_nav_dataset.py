index_to_data_path = os.path.join(
            data_split_folder,
            f"dataset_dist_{min_dist_cat}_to_{max_dist_cat}_context_{context_type}_n{context_size}_slack_{end_slack}.pkl")



with open(index_to_data_path, "rb") as f:
    index_to_data, goals_index = pickle.load(f)
    
    
traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
with open(traj_names_file, "r") as f:
    file_lines = f.read()
    traj_names = file_lines.split("\n")
if "" in traj_names:
    traj_names.remove("")
    
def _get_trajectory(trajectory_name):
    if trajectory_name in trajectory_cache:
        return trajectory_cache[trajectory_name]
    else:
        with open(os.path.join(data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        trajectory_cache[trajectory_name] = traj_data
        return traj_data
    
    
def _sample_goal(trajectory_name, curr_time, max_goal_dist):
    """
    Sample a goal from the future in the same trajectory.
    Returns: (trajectory_name, goal_time, goal_is_negative)
    """
    #goal_offset = np.random.randint(0, max_goal_dist + 1)       # 1 out of 21 chooses a neg sample (0~20)
    goal_offset = np.random.randint(1, max_goal_dist + 1)        # 1 out of 19 chooses w/o neg sample (1~20)
    if goal_offset == 0:
        trajectory_name, goal_time = _sample_negative()  # select a different traj
        return trajectory_name, goal_time, True
    else:
        goal_time = curr_time + int(goal_offset * 1)  #  waypoint_spacing = 1 by default  
        return trajectory_name, goal_time, False


#def _build_caches(self, use_tqdm: bool = True):   build _build_caches

depth_cache_filename = os.path.join(data_split_folder, f"depth_dataset_{dataset_name}.lmdb",)

# Load all the trajectories into memory. These should already be loaded, but just in case.
for traj_name in traj_names:   # saves trajs in dict name: self.trajectory_cache
    _get_trajectory(traj_name)
        
if not os.path.exists(depth_cache_filename):
    tqdm_iterator = tqdm.tqdm(
        self.goals_index,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Building Depth LMDB cache for {self.dataset_name}"
    )
    with lmdb.open(depth_cache_filename, map_size=2**40) as depth_cache:
        with depth_cache.begin(write=True) as txn:
            for traj_name, time in tqdm_iterator:
                rgb_path, depth_path = get_data_path(self.data_folder, traj_name, int(time))
                # print("image_path: %s"%image_path )
                with open(depth_path, "rb") as f:
                    txn.put(depth_path.encode(), f.read())

#def _compute_actions(traj_data, curr_time, goal_time): # goal_time : temporal idx of sg in the given traj
    start_index = curr_time
    end_index = curr_time + len_traj_pred * waypoint_spacing + 1
    yaw = traj_data["yaw"][start_index:end_index:waypoint_spacing]
    positions = traj_data["position"][start_index:end_index:waypoint_spacing]
    goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (self.len_traj_pred + 1,):
        const_len = self.len_traj_pred + 1 - yaw.shape[0]
        yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
        positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

    assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
    assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

    #######################################################################################3
    # critical bug: yaw[0] is DEG while np.cos() and np.sin() takes radian !!!
    # seems RECON's yaw is in radian ... though
    # TODO: Change the orientation representation to Quaternion from yaw

    theta = float(yaw[0])  # hkm
    waypoints = to_local_coords(positions, positions[0], theta)  # yaw[0])
    goal_pos = to_local_coords(goal_pos, positions[0],   theta)  # yaw[0])
    delta_waypoints = waypoints[1:] - waypoints[0]
    delta_time = 0.025   # get_time( traj_data )

    assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

    if self.learn_angle:
        yaw = yaw[1:] - yaw[0]     #TODO:  Check if this simple subtraction is correct/wrong....
        actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
    else:
        actions = waypoints[1:]
    
    if self.normalize:
        actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

    assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

    return actions, goal_pos










# get_item

#def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
    """
    Args:
        i (int): index to ith datapoint
    Returns:
        Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
            obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
            goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
            dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
            action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
            which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
    """
    f_curr, curr_time, max_goal_dist = index_to_data[0] # ( 'canteen/dynamic/part3/Capture_1', 5, 20 )
    f_goal, goal_time, goal_is_negative = _sample_goal(f_curr, curr_time, max_goal_dist)   # returns random goal  # ( 'canteen/dynamic/part3/Capture_1', 17, False)
    # f_goal : traj name, goal_time : temporal idx of future sg, goal_is_negative: True or False
    # Load images
    
    context = []
    if context_type == "temporal":
        # sample the last self.context_size times from interval [0, curr_time)
        context_times = list(
            range(
                curr_time + -context_size * waypoint_spacing,
                curr_time + 1,
                waypoint_spacing,
            )
        )
        context = [(f_curr, t) for t in context_times]
    else:
        raise ValueError(f"Invalid context type {self.context_type}")

    #obs_rgb = torch.cat([self._load_rgb(f, t) for f, t in context]) # hkm
    obs_depth = torch.cat([self._load_depth(f, t) for f, t in context])  # hkm

    # Load goal image
    #goal_rgb = self._load_rgb(f_goal, goal_time)  # hkm
    goal_depth = self._load_depth(f_goal, goal_time)  # hkm

    # Load other trajectory data
    curr_traj_data = self._get_trajectory(f_curr)
    curr_traj_len = len(curr_traj_data["position"])
    assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

    goal_traj_data = self._get_trajectory(f_goal)
    goal_traj_len = len(goal_traj_data["position"])
    assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

    # Compute actions
    actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
    
    # Compute distances (temporal)
    if goal_is_negative:
        distance = self.max_dist_cat
    else:
        distance = (goal_time - curr_time) // self.waypoint_spacing
        assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

    actions_torch = torch.as_tensor(actions, dtype=torch.float32)
    if self.learn_angle:
        actions_torch = calculate_sin_cos(actions_torch)    # x, y, cos(yaw), sin(yaw)
    
    action_mask = ( # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound
        (distance < self.max_action_distance) and       # max_action_distance = config[action][max_dist_cat] 10 in vint
        (distance > self.min_action_distance) and       # min_action_distance = config[action][min_dist_cat] 0  in vint
        (not goal_is_negative)
    )

    return (
        torch.as_tensor(obs_rgb, dtype=torch.float32),
        torch.as_tensor(obs_depth, dtype=torch.float32),
        torch.as_tensor(goal_rgb, dtype=torch.float32),
        torch.as_tensor(goal_depth, dtype=torch.float32),
        actions_torch,
        torch.as_tensor(distance, dtype=torch.int64),
        torch.as_tensor(goal_pos, dtype=torch.float32),
        torch.as_tensor(self.dataset_index, dtype=torch.int64),
        torch.as_tensor(action_mask, dtype=torch.float32),
    )
