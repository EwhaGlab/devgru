import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import math
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(BASE_DIR)
from pathlib import Path

import utils.rigid_motion as rm

from data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_colldata_path, # get_data_path,
    to_local_coords,
    _interp_pose,
    #_get_rel_pose_se2,
    #_normalize_waypoints,
    #_denormalize_waypoints,
    #_normalize_context_poses,
    #_denormalize_context_poses,
    _normalize_subgoal,
    _denormalize_subgoal,
    _normalize_pose,
    _denormalize_pose,
)
import cv2
MAX_DEPTH = 65535.0
#MAX_TOGOAL_DIST = 2.412 # when max frame diff = 80
#MAX_TOGOAL_DIST =  0.6085 # when max frame diff = 20
from scipy.io import savemat
class Collision_Dataset_Simple(Dataset):
    def __init__(
        self,
        base_data_folder: str,
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
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the orientation of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context

            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
        """
        self.base_data_folder = base_data_folder
        self.dataset_name = dataset_name

        path = Path(self.base_data_folder)
        self.traj_names = [str(p) for p in path.iterdir() if p.is_dir() and p.name.startswith("data")]
        self.traj_names.sort()  # by hkm

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.min_frame_dist = min_frame_dist     #  self.distance_categories[0]
        self.max_frame_dist = max_frame_dist     #  self.distance_categories[-1]
        self.negative_mining = negative_mining

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
        self.nav_seq_idx = []
        self.target_sg_idx = []

        if self.learn_angle:
            self.num_action_params = 4  # x y qw qz
        else:
            self.num_action_params = 2

    # def __getstate__(self):  # hkm
    #     state = self.__dict__.copy()
    #     #state["_image_cache"] = None
    #     state["_rgb_cache"] = None
    #     state["_depth_cache"] = None
    #     state["_goal_rgb_cache"] = None
    #     state["_goal_depth_cache"] = None
    #     return state


    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_rgb(self, trajectory_name, time):
        rgb_path = get_colldata_path(trajectory_name, time, 'rgb')
        try:
            return img_path_to_data(rgb_path, self.image_size)
        except TypeError:
            print(f"Failed to load rgb image {rgb_path} @ {time}")

    def _load_depth(self, trajectory_name, time):
        depth_path = get_colldata_path(trajectory_name, time, 'depth')
        try:
            return img_path_to_data(depth_path, self.image_size)
        except TypeError:
            print(f"Failed to load depth image {depth_path} @ {time}")

    def _load_goal_rgb(self, trajectory_name):
        goal_rgb_path = get_colldata_path(trajectory_name, -1, 'rgb')
        try:
            return img_path_to_data(goal_rgb_path, self.image_size)
        except TypeError:
            print(f"Failed to load goal rgb image {goal_rgb_path} ")

    def _load_goal_depth(self, trajectory_name):
        goal_depth_path = get_colldata_path(trajectory_name, -1, 'depth')
        try:
            return img_path_to_data(goal_depth_path, self.image_size)
        except TypeError:
            print(f"Failed to load goal depth {goal_depth_path} ")

    # def _get_trajectory(self, trajectory_name):
    #     if trajectory_name in self.trajectory_cache:
    #         return self.trajectory_cache[trajectory_name]
    #     else:
    #         with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
    #             traj_data = pickle.load(f)
    #         self.trajectory_cache[trajectory_name] = traj_data
    #         return traj_data

    def __len__(self) -> int:
        return len(self.traj_names)
        #return len(self.index_to_data)

    def is_collision(self, old_goal_pose, new_goal_pose):
        dx = new_goal_pose[0] - old_goal_pose[0]
        dy = new_goal_pose[1] - old_goal_pose[1]
        dist = math.hypot(dx, dy)
        return  not math.isclose(dist, 0.0, abs_tol=1e-3)

    def _load_context_actions(self, data_dir):  # goal_time : temporal idx of sg in the given traj
        #######################################################################################3
        # context_times: 0 ~ 5  if curr_time is 5

        traj_file = '%s/pose_context_m.txt' % data_dir
        traj_data = np.loadtxt(traj_file)

        quats = traj_data[:, 3:]  # (N,)
        positions = traj_data[:, :2]
        num_context = len(positions)
        # xc, yc = positions[-1]

        xps = positions[:, 0]  # shape: (len_traj_pred,)
        yps = positions[:, 1]  # shape: (len_traj_pred,)
        qw = quats[:, 0]
        qz = quats[:, 3]
        thetas = 2 * np.arctan2(qz, qw)
        th_ps = thetas  # shape: (len_traj_pred,)

        # TODO: Explore better way to normalize actions !!! ###
        if self.normalize:
            action = _normalize_pose(np.asarray([xps, yps, qw, qz]).transpose(),
                                     waypoint_spacing=self.waypoint_spacing, dataset_name=self.dataset_name)

        else:
            action = np.array([xps, yps, qw, qz], dtype='float32').transpose()

        return action #out_action

    def _load_next_actions(self, data_dir):  # goal_time : temporal idx of sg in the given traj
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
        # num_pos = len(positions)  # curr pos ~ curr pos + 5

        data_sg_idx_file = '%s/data_sg_idx.txt' % data_dir

        if os.path.exists(data_sg_idx_file):
            self.nav_seq_idx, self.target_sg_idx, global_target_sg_idx = np.loadtxt(data_sg_idx_file, dtype=np.int64)
        else:
            self.nav_seq_idx = self.target_sg_idx = -1,

        # load old sg
        old_sg_data_file = '%s/old_subgoal_m.txt' % data_dir
        old_sg_data = np.loadtxt(old_sg_data_file)  # [xg_old, yg_old, qw_old, qz_old; xg_new, yg_new, qw_new, qz_new]
        old_xg = old_sg_data[0].copy()
        old_yg = old_sg_data[1].copy()
        old_qw_g = old_sg_data[2].copy()
        old_qz_g = old_sg_data[3].copy()
        #old_th_g = 2 * np.arctan2(old_qz_g, old_qw_g)

        # load new sg
        new_sg_data_file = '%s/new_subgoal_m.txt' % data_dir
        new_sg_data = np.loadtxt(new_sg_data_file)  # [xg_old, yg_old, qw_old, qz_old; xg_new, yg_new, qw_new, qz_new]
        new_xg = new_sg_data[0].copy()
        new_yg = new_sg_data[1].copy()
        new_qw_g = new_sg_data[2].copy()
        new_qz_g = new_sg_data[3].copy()
        #new_th_g = 2 * np.arctan2(new_qz_g, new_qw_g)

        # load waypoint
        traj_data_file = '%s/corrected_waypoints_m.txt' % (data_dir)
        traj_data = np.loadtxt(traj_data_file)

        xp = traj_data[:, 0].copy()
        yp = traj_data[:, 1].copy()
        qw = traj_data[:, 2].copy()
        qz = traj_data[:, 3].copy()

        # TODO: Explore better way to normalize actions !!! ###
        if self.normalize:
            action = _normalize_pose(traj_data,
                                     waypoint_spacing=self.waypoint_spacing, dataset_name=self.dataset_name)
            old_goal = _normalize_subgoal(old_sg_data, max_frame_dist=self.max_frame_dist,
                                     dataset_name=self.dataset_name)
            new_goal = _normalize_subgoal(new_sg_data, max_frame_dist=self.max_frame_dist,
                                     dataset_name=self.dataset_name)

        else:
            action = traj_data.copy() #np.array([xp, yp, th_p], dtype='float32').transpose()
            old_goal = old_sg_data.copy()  #np.array([old_xg, old_yg, old_th_g], dtype='float32')
            new_goal = new_sg_data.copy()  #np.array([new_xg, new_yg, new_th_g], dtype='float32')

        if self.learn_angle:
            out_action = action
            out_old_relgoal = old_goal
            out_new_relgoal = new_goal
        else:
            out_action = action[..., :2]
            out_old_relgoal = old_goal[:2]
            out_new_relgoal = new_goal[:2]

        # return actions, goal_pos
        return out_action, out_old_relgoal, out_new_relgoal

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
        #f_curr, curr_time, max_frame_goal_dist = self.index_to_data[i] # ( obs_path, sidx, max_frame_goal_dist (max_goal_dist * waypoint_spacing) )
        #f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_frame_goal_dist) # returns random goal  # ( , traj_len * waypoint_spacing, False)

        f_curr = self.traj_names[i]
        context_time = np.loadtxt('%s/context_index.txt' % f_curr, dtype=int)
        curr_time = context_time[-1]

        f_goal = f_curr
        goal_is_negative = False

        # f_goal : traj name, goal_time : temporal idx of future sg, goal_is_negative: True or False
        # Load images
        context = []
        # TODO: eliminate self.context_type parameter in the future

        context = [(f_curr, t) for t in context_time]
        goal_time = -1

        obs_rgbs = torch.cat([self._load_rgb(f, t) for f, t in context])        # hkm
        obs_depths = torch.cat([self._load_depth(f, t) for f, t in context])    # hkm
        # print("obs minmax: %f %f"%(torch.min(obs_depths), torch.max(obs_depths)) )
        # Load goal image
        goal_rgb = self._load_goal_rgb(f_goal)  # hkm
        goal_depth = self._load_goal_depth(f_goal)  # normalized depth ( by MAX_DEPTH)

        # Load other trajectory data
        #curr_traj_data = self._get_trajectory(f_curr)
        #curr_traj_len = len(curr_traj_data["position"])

        # Compute actions
        # actions and goal_pos are computed w.r.t the 1st image frame. (goal_time-curr_time : 1 ~ max_frame_dist * waypoint_spacing )
        actions_n, old_goalpos_n, new_goalpos_n = self._load_next_actions(f_curr) # actions: (x,y) or (x,y,theta), normalized
        context_actions_n = self._load_context_actions(f_curr) # context actions : previous actions
        collision_flag = self.is_collision(old_goalpos_n, new_goalpos_n)

        # Compute distances (temporal or spatial)
        if self.goal_dist_type == "temporal":
            raise NotImplementedError
            # if goal_is_negative:
            #     distance = self.max_frame_dist
            # else:
            #     distance = (goal_time - curr_time) // self.waypoint_spacing
            #     assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        elif self.goal_dist_type == "spatial":
            if goal_is_negative:
                raise NotImplementedError
            else:
                dx = new_goalpos_n[0]
                dy = new_goalpos_n[1]

            if self.learn_angle:
                if goal_is_negative:
                    raise NotImplementedError
                else:
                    qw = new_goalpos_n[2]
                    qz = new_goalpos_n[3]
                pose_diff = new_goalpos_n    #np.array([dx, dy, qw, qz]) #angle_diff_rad])
            else:
                pose_diff = new_goalpos_n[:2]
        else:
            print("Unknown goal_dist_type \n")
            raise NotImplementedError

        actions_torch = torch.as_tensor(actions_n, dtype=torch.float32)
        context_actions_torch = torch.as_tensor(context_actions_n, dtype=torch.float32)
        goal_torch = torch.as_tensor(old_goalpos_n, dtype=torch.float32)
        action_mask = (1) # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound

        obs_rgb_norm    = obs_rgbs   # normalized when load_rgb is called
        goal_rgb_norm   = goal_rgb   
        obs_depths_norm = obs_depths
        goal_depth_norm = goal_depth
        data_info = f'{self.dataset_name} {f_curr} {curr_time} {f_goal} {-1} {-1}'#{goal_time} {goal_time - curr_time}'
        return (
            torch.as_tensor(obs_rgb_norm, dtype=torch.float32),
            torch.as_tensor(obs_depths_norm, dtype=torch.float32),
            torch.as_tensor(goal_rgb_norm, dtype=torch.float32),
            torch.as_tensor(goal_depth_norm, dtype=torch.float32),
            actions_torch,          # normalized action
            context_actions_torch,  # normalized action
            goal_torch,   #  old sg : error accumulated sg pose (wrt base_link)
            torch.as_tensor(pose_diff, dtype=torch.float32),  # new sg:  normalized geometric pose diff (to the new sg) wrt base_link (on xy plane)
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(self.nav_seq_idx, dtype=torch.int64),
            torch.as_tensor(self.target_sg_idx, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(collision_flag, dtype=torch.bool),
            data_info
        )
