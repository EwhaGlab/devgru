import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb
import math
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

import utils.rigid_motion as rm

from data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
    #_get_rel_cam_poses,
    #_get_rel_robot_poses,
    _interp_pose,
    _get_rel_pose_se2,
    _normalize_pose,
    _normalize_subgoal
)
import cv2
MAX_DEPTH = 65535.0
#MAX_TOGOAL_DIST = 2.412 # when max frame diff = 80
#MAX_TOGOAL_DIST =  0.6085 # when max frame diff = 20
from scipy.io import savemat
class DepthNav_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_frame_dist: int,
        max_frame_dist: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        goal_dist_type: str = "spatial",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str =  "Depth",   #"image",
        goal_type: str = "Depth"    #"image",
    ):
        """
        Main DepthNav dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_frame_dist (int): Minimum distance (frame) to use
            max_frame_dist (int): Maximum distance (frame) to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the orientation of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        #self.distance_categories = list( range(min_frame_dist, max_frame_dist + 1, self.waypoint_spacing) )
        self.min_frame_dist = min_frame_dist     #  self.distance_categories[0]
        self.max_frame_dist = max_frame_dist     #  self.distance_categories[-1]
        self.negative_mining = negative_mining
        #if self.negative_mining:
            #self.distance_categories.append(-1)
        self.goal_dist_type = goal_dist_type
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type
        # load data/data_config.yaml
        with open(
            os.path.join(BASE_DIR + "/config/data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()          # sample_index = (traj_name, curr_time, max_goal_distance),  goals_index = (traj_name, 0~max_idx)
        self._build_caches()        # build LMDB cache for depth images        
        
        if self.learn_angle:
            self.num_action_params = 4  # x y qw qz
        else:
            self.num_action_params = 2

    def __getstate__(self):  # hkm
        state = self.__dict__.copy()
        #state["_image_cache"] = None
        state["_rgb_cache"] = None
        state["_depth_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        rgb_cache_filename = os.path.join(
            self.data_split_folder,
            f"rgb_dataset_{self.dataset_name}.lmdb",
        )
        depth_cache_filename = os.path.join(
            self.data_split_folder,
            f"depth_dataset_{self.dataset_name}.lmdb",
        )
        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:   # saves trajs in dict name: self.trajectory_cache
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(rgb_cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building RGB LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(rgb_cache_filename, map_size=2**40) as rgb_cache:
                with rgb_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        #rgb_path = get_data_path(self.data_folder, traj_name, int(time), 'rgb')
                        rgb_path = get_data_path(traj_name, int(time), 'rgb')
                        #print("image_path: %s"%rgb_path )
                        with open(rgb_path, "rb") as f:
                            txn.put(rgb_path.encode(), f.read())

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
                        #depth_path = get_data_path(self.data_folder, traj_name, int(time), 'depth')
                        depth_path = get_data_path(traj_name, int(time), 'depth')
                        # print("image_path: %s"%image_path )
                        dm = cv2.imread(depth_path, -1)  # read as uint16
                        #dm = dm.astype('float32') / 65535.0 * 255.0  # rescale it to 0 ~ 255
                        #dm = np.round(dm).astype(np.uint8)        # convert to uint8
                        dm_bytes = cv2.imencode('.png', dm)[1].tobytes()
                        txn.put(depth_path.encode(), dm_bytes)
                        # with open(depth_path, "rb") as f:
                            #txn.put(depth_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._rgb_cache: lmdb.Environment = lmdb.open(rgb_cache_filename, readonly=True)
        self._depth_cache: lmdb.Environment = lmdb.open(depth_cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []  #( traj_name, temporal index, max_goal_distance )
        goals_index = []    #( traj_name, temporal index, 0 ~ end_idx )

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_frame_dist * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index       # goals_index is needed for "sample_negative().  i.e.) neg random sampling"

    def _sample_goal(self, trajectory_name, curr_time, max_frame_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        # goal_offset = np.random.randint(0, max_goal_dist + 1)       # 1 out of 21 chooses a neg sample (0~20)
        # max_goal_frame_dist = max_goal_dist(24) * waypoint_spacing
        goal_offset = np.random.randint(1, max_frame_goal_dist + 1)  # 1 out of 19 chooses w/o neg sample (1 ~ max_goal_dist * waypoint_spacing)
        if goal_offset == 0:
            assert 0 # something went wrong...
            trajectory_name, goal_time = self._sample_negative()    # select a different traj
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + goal_offset  #* self.waypoint_spacing
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",)
        
        #index_to_data_path = os.path.join(
            #self.data_split_folder,
            #f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",)

        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_rgb(self, trajectory_name, time):
        #rgb_path = get_data_path(self.data_folder, trajectory_name, time, 'rgb')
        rgb_path = get_data_path(trajectory_name, time, 'rgb')
        #print("rgb path: %s \n"%rgb_path )
        #print("time : %d \n"%time)
        try:
            with self._rgb_cache.begin() as txn:
                rgb_buffer = txn.get(rgb_path.encode())
                rgb_bytes = bytes(rgb_buffer)
            rgb_bytes = io.BytesIO(rgb_bytes)
            return img_path_to_data(rgb_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load rgb image {rgb_path} @ {time}")

    def _load_depth(self, trajectory_name, time):
        #depth_path = get_data_path(self.data_folder, trajectory_name, time, 'depth')
        depth_path = get_data_path(trajectory_name, time, 'depth')
        #print("image_path: "%depth_path)
        try:
            with self._depth_cache.begin() as txn:
                depth_buffer = txn.get(depth_path.encode())
                depth_bytes = bytes(depth_buffer)
            depth_bytes = io.BytesIO(depth_bytes)
            #Q = np.asarray( Image.open(depth_bytes) )
            #print("min max %d %d"%(np.min(Q), np.max(Q)) )
            return img_path_to_data(depth_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load depth image {depth_path}")

    def _compute_context_actions(self, traj_data, curr_time,
                              context_times):  # goal_time : temporal idx of sg in the given traj
        #######################################################################################3
        # context_times: 0 ~ 5  if curr_time is 5
        curr_idx = curr_time
        context_idx = context_times
        num_context = len(context_idx)

        thetas = traj_data['orientation'][context_idx].squeeze()  # (N,)
        positions = traj_data['position'][context_idx]  # (N,2)

        xc, yc = traj_data['position'][curr_idx].squeeze()  # robot traj data
        th_c = traj_data['orientation'][curr_idx][0]
        xps = positions[:, 0]  # shape: (len_traj_pred,)
        yps = positions[:, 1]  # shape: (len_traj_pred,)
        th_ps = thetas  # shape: (len_traj_pred,)

        if self.dataset_name == 'former':
            bHc = rm.xyzrpy_to_htm(self.data_config['camera_matrics']['cam_wrt_base'])
        else:
            bHc = np.eye(4)

        rel_xp, rel_yp, rel_theta_p = _get_rel_pose_se2(np.array([xc, yc, th_c]), np.array([xps, yps, th_ps]), self.dataset_name, bHc)

        # TODO: Explore better way to normalize actions !!! ###
        if self.normalize:
            action = _normalize_pose(np.asarray([rel_xp, rel_yp, rel_theta_p]).transpose(),
                                     waypoint_spacing=self.waypoint_spacing, dataset_name=self.dataset_name)

        else:
            action = np.array([rel_xp, rel_yp, rel_theta_p], dtype='float32').transpose()

        if self.learn_angle:
            # conv angle to quat rep
            out_action = np.zeros([num_context, 4], dtype='float32')
            for ii in range(0, num_context):
                q_a = rm.rpy2quat(0, 0, action[ii, 2])
                assert round(math.sqrt(q_a[0] * q_a[0] + q_a[-1] * q_a[-1]), 3) == 1.0, f"Is q_a: {q_a} unit quat ? "
                out_action[ii] = np.concatenate((action[ii, :2], np.array([q_a[0], q_a[-1]])), axis=0)
        else:
            out_action = action[..., :2]

        return out_action

    def _compute_next_actions(self, traj_data, curr_time, goal_time):  # goal_time : temporal idx of sg in the given traj
        #######################################################################################3
        # traj_data must be base_link's motion data. i.e, curr base_link wrt origin (base_link)

        # seems RECON's orientation is in radian ... though
        # TODO: Change the orientation representation to Quaternion from orientation
        # TODO Save time-stamp data for the correspoinding pose data pkl
        # TODO Need to develop a func to read timestamp of depth data

        # #############################################################################################################
        # We now transform the P0 ~ P5 and th0 ~ th5 w.r.t P0 and th0
        # i.e.) We transform the abs coordinate pose to the relative coordinate (w.r.t P0)
        # Then, we normalize the tformed pts by the geometric max_goal distance to enforce it btwn 0 ~ 1
        # i.e.) normalized pts : metric_waypoint_space * len_traj_pred
        # (1) the transformed & normalized Pn1 ~ Pn5 are GT waypoints
        # (2) the transformed & normalized Pn5 corresponds to the GT goal_pos
        ###############################################################################################################
        #num_pos = len(positions)  # curr pos ~ curr pos + 5

        start_idx = curr_time
        end_idx   = curr_time + self.len_traj_pred * self.waypoint_spacing + 1

        thetas = traj_data['orientation'][start_idx:end_idx:self.waypoint_spacing].squeeze()  # (N,)
        positions = traj_data['position'][start_idx:end_idx:self.waypoint_spacing]            # (N,2)

        x0, y0 = positions[0]    # robot traj data
        th0 = thetas[0]
        # curr (xc,yc,th_c) -------- pred (xp,yp,th_p) -------- goal (xf, yf, th_f)
        # pred xy and th are the cam pose at the middle of cam traj between the curr and the goal location
        #assert (goal_idx - curr_time >= self.len_traj_pred), (f"goal_idx - curr_time {goal_idx - curr_time} should be GEQ to len_traj_pred")
        xp = positions[1:, 0]   # shape: (len_traj_pred,)
        yp = positions[1:, 1]   # shape: (len_traj_pred,)
        th_p = thetas[1:]       # shape: (len_traj_pred,)

        goal_idx = min(goal_time, len(traj_data["position"]) - 1)
        xe = traj_data['position'][goal_idx, 0]
        ye = traj_data['position'][goal_idx, 1]
        th_e= traj_data['orientation'][goal_idx][0]

        if self.dataset_name == 'former':
            bHc = rm.xyzrpy_to_htm( self.data_config['camera_matrics']['cam_wrt_base'] )
        else:
            bHc = np.eye(4)

        rel_xp, rel_yp, rel_theta_p = _get_rel_pose_se2(np.array([x0, y0, th0]), np.array([xp, yp, th_p]), self.dataset_name, bHc)
        rel_xg, rel_yg, rel_theta_g = _get_rel_pose_se2(np.array([x0, y0, th0]), np.array([xe, ye, th_e]), self.dataset_name, bHc)

        #TODO: Explore better way to normalize actions !!! ###
        if self.normalize:
            action = _normalize_pose(np.asarray([rel_xp, rel_yp, rel_theta_p]).transpose(), waypoint_spacing=self.waypoint_spacing, dataset_name=self.dataset_name)
            goal = _normalize_subgoal(np.asarray([rel_xg, rel_yg, rel_theta_g]), max_frame_dist=self.max_frame_dist, dataset_name=self.dataset_name)

        else:
            action = np.array([rel_xp, rel_yp, rel_theta_p], dtype='float32').transpose()
            goal = np.array([rel_xg, rel_yg, rel_theta_g], dtype='float32')

        #assert( np.linalg.norm(action[2:]) == 1), (f"action quat norm {np.linalg.norm(action[2:])}  should be 1,  quat ws {quat_p}, and   rol pit orientation were: {rol_p}, {pit_p}, {orientation_p}")

        if self.learn_angle:
            # conv angle to quat rep
            out_action = np.zeros([self.len_traj_pred, 4], dtype='float32')
            for ii in range(0, self.len_traj_pred):
                q_a = rm.rpy2quat(0, 0, action[ii, 2])
                assert round( math.sqrt( q_a[0]*q_a[0] + q_a[-1]*q_a[-1]), 3 )== 1.0, f"Is q_a: {q_a} unit quat ? "
                out_action[ii] = np.concatenate( (action[ii, :2], np.array([q_a[0], q_a[-1]]) ), axis=0 )
            q_g = rm.rpy2quat(0, 0, goal[2])
            assert round( math.sqrt(q_g[0] * q_g[0] + q_g[-1] * q_g[-1]), 3 ) == 1.0, f"Is q_g: {q_g} unit quat ? "
            out_relgoal = np.concatenate( (goal[:2], np.array([q_g[0], q_g[-1]]) ), axis= 0 )
        else:
            out_action  = action [...,:2]
            out_relgoal = goal[:2]
#           raise NotImplementedError

        # assert actions.shape == (self.len_traj_pred,
        #                          self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"
        # return actions, goal_pos
        return out_action, out_relgoal

    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple:
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
        f_curr, curr_time, max_frame_goal_dist = self.index_to_data[i] # ( obs_path, sidx, max_frame_goal_dist (max_goal_dist * waypoint_spacing) )
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_frame_goal_dist) # returns random goal  # ( , traj_len * waypoint_spacing, False)

        # f_goal : traj name, goal_time : temporal idx of future sg, goal_is_negative: True or False
        # Load images
        context = []
        # TODO: eliminate self.context_type parameter in the future
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_rgbs = torch.cat([self._load_rgb(f, t) for f, t in context])        # hkm
        obs_depths = torch.cat([self._load_depth(f, t) for f, t in context])    # hkm
        # print("obs minmax: %f %f"%(torch.min(obs_depths), torch.max(obs_depths)) )
        # Load goal image
        goal_rgb = self._load_rgb(f_goal, goal_time)  # hkm
        goal_depth = self._load_depth(f_goal, goal_time)  # hkm

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        #print("f curr %s time %d  curr_traj len: %d\n"%(f_curr, curr_time, curr_traj_len) )
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        #print("goal %s  time %d \n"%(f_goal, goal_time) )
        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len} curr_time: {curr_time}, curr_traj_len: {curr_traj_len} " \
                                          f"\n f_curr: {f_curr} and f_goal: {f_goal}"

        # Compute actions
        # actions and goal_pos are computed w.r.t the 1st image frame. (goal_time-curr_time : 1 ~ max_frame_dist * waypoint_spacing )
        actions, goal_pos = self._compute_next_actions(curr_traj_data, curr_time, goal_time) # actions.shape:  num_len_traj x (2 or 3)
        context_actions = self._compute_context_actions(curr_traj_data, curr_time,  context_times) # context actions : previous actions

        # Compute distances (temporal or spatial)
        pose_diff = []
        if self.goal_dist_type == "temporal":
            if goal_is_negative:
                distance = self.max_frame_dist
            else:
                distance = (goal_time - curr_time) // self.waypoint_spacing
                assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        elif self.goal_dist_type == "spatial":
            if goal_is_negative:
                raise NotImplementedError
            else:
                dx = goal_pos[0]
                dy = goal_pos[1]

            if self.learn_angle:
                if goal_is_negative:
                    raise NotImplementedError
                else:
                    qw = goal_pos[2]
                    qz = goal_pos[3]
                pose_diff = goal_pos    #np.array([dx, dy, qw, qz]) #angle_diff_rad])
            else:
                pose_diff = goal_pos[:2]
            # if goal_is_negative:
            #     xy_distance = self.max_frame_dist * self.data_config["metric_frame_spacing"]   #
            # else:
            #     xy_distance = ( goal_pos[0]**2 + goal_pos[1]**2 ) ** 0.5  # relative distance
            #     xy_distance = xy_distance * ( self.data_config["metric_frame_spacing"] * self.waypoint_spacing )   # normalized relative distance
            #
            # if self.learn_angle:
            #     if goal_is_negative:
            #         q_dist = 1
            #     else:
            #         q_curr = rm.rpy2quat(0, 0, 0)
            #         q_goal = np.array([ goal_pos[2], 0, 0, goal_pos[-1] ], dtype='float32')
            #         q_dist, angle_diff_rad = rm.quat_ang_dist( q_curr, q_goal )
            #     distance = np.array([xy_distance, q_dist]) #angle_diff_rad])
            # else:
            #     distance = xy_distance
        else:
            print("Unknown goal_dist_type \n")
            raise NotImplementedError

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        context_actions_torch = torch.as_tensor(context_actions, dtype=torch.float32)
        goal_torch = torch.as_tensor(goal_pos, dtype=torch.float32)
        action_mask = (1) # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound

        obs_rgb_norm    = obs_rgbs   # normalized when load_rgb is called
        goal_rgb_norm   = goal_rgb   
        obs_depths_norm = obs_depths
        goal_depth_norm = goal_depth
        data_info = f'{self.dataset_name} {f_curr} {curr_time} {f_goal} {goal_time} {goal_time - curr_time}'
        return (
            torch.as_tensor(obs_rgb_norm, dtype=torch.float32),
            torch.as_tensor(obs_depths_norm, dtype=torch.float32),
            torch.as_tensor(goal_rgb_norm, dtype=torch.float32),
            torch.as_tensor(goal_depth_norm, dtype=torch.float32),
            actions_torch,
            context_actions_torch,
            goal_torch,         # wrt base_link
            torch.as_tensor(pose_diff, dtype=torch.float32), # sg:  normalized geometric pose diff wrt base_link (on xy plane)
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(False, dtype=torch.bool),  # train for ordinary data
            data_info
        )
