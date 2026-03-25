import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import rigid_motion as rm

from depth_nav_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
    #_get_rel_cam_poses,
    #_get_rel_robot_poses,
    _interp_pose,
    _get_rel_pose_se2,
    _normalize_pose,
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
        #min_action_distance: int,
        #max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
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
        self.waypoint_spacing = 1 # waypoint_spacing
        #self.distance_categories = list( range(min_frame_dist, max_frame_dist + 1, self.waypoint_spacing) )
        self.min_frame_dist = min_frame_dist     #  self.distance_categories[0]
        self.max_frame_dist = max_frame_dist     #  self.distance_categories[-1]
        #self.negative_mining = negative_mining
        #if self.negative_mining:
            #self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

#        self.min_action_distance = min_action_distance
#        self.max_action_distance = max_action_distance

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
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
            #os.path.join(os.path.dirname(__file__), "../../config/data_config.yaml"), "r"
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
            self.num_action_params = 3 #4
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
#           #end_time = traj_len - self.end_slack - self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_frame_dist * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index       # goals_index is needed for "sample_negative().  i.e.) neg random sampling"

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        #goal_offset = np.random.randint(0, max_goal_dist + 1)       # 1 out of 21 chooses a neg sample (0~20)
        goal_offset = np.random.randint(1, max_goal_dist + 1)        # 1 out of 19 chooses w/o neg sample (1~20)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()    # select a different traj
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
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
        try:
            with self._rgb_cache.begin() as txn:
                rgb_buffer = txn.get(rgb_path.encode())
                rgb_bytes = bytes(rgb_buffer)
            rgb_bytes = io.BytesIO(rgb_bytes)
            return img_path_to_data(rgb_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load rgb image {rgb_path}")

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

    # def _compute_next_control_actions(self):
    #     # not implemented yet
    #     print("Not implemented yet \n")

#     def _compute_waypoint_actions(self, traj_data, curr_time, goal_time): # goal_time : temporal idx of sg in the given traj
#         start_index = curr_time
#         end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
#         orientation = traj_data["orientation"][start_index:end_index:self.waypoint_spacing]
#         positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]      # P0, P1, P2, P3, P4, P5  pos btwn curr ~ curr + context (5)
#         # given traj_len = 5,  P0 ~ P5 and th0 ~ th5 are taken
#         # where P0 corresponds to curr_pos Pg is at P0 + (1~19)
#         #print("goal_time:", goal_time)
#         goal_idx = min(goal_time, (len(traj_data["position"]) - 1) )
#         goal_pos = traj_data["position"][goal_idx]    # goalidx= 6-1,  goal = P5  abs goal pos
#         goal_theta = traj_data["orientation"][goal_idx]       # abs goal theta
#         if len(orientation.shape) == 2:
#             orientation = orientation.squeeze(1)
#
#         if orientation.shape != (self.len_traj_pred + 1,):
#             const_len = self.len_traj_pred + 1 - orientation.shape[0]
#             orientation = np.concatenate([orientation, np.repeat(orientation[-1], const_len)])
#             positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)
#
#         assert orientation.shape == (self.len_traj_pred + 1,), f"{orientation.shape} and {(self.len_traj_pred + 1,)} should be equal"
#         assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
#
#
# # for debugging
#         # compute orientation
#         #orientation = traj_data["orientation"]
#         #positions = traj_data["position"]
#         #num_wpts = len(positions)
#         #wHc0_left = rm.xyzrpy_to_htm(np.array([positions[0,0],positions[0,1],0,0,0,orientation[0][0]]))
#         #wHc0 = rm.Left2Right( wHc0_left )
#         #c0Hw = np.linalg.inv(wHc0)
#         #c0Hc = np.zeros([4,4, num_wpts])
#         #wHc  = np.zeros([4,4, num_wpts])
#         #for ii in range(0, num_wpts):
#             #x = positions[ii,0]
#             #y = positions[ii,1]
#             #th = orientation[ii][0]
#             #wHc_left = rm.xyzrpy_to_htm(np.array([x,y,0,0,0,th]))
#             #wHc [:,:,ii] = rm.Left2Right(wHc_left)
#             #c0Hc[:,:,ii] = np.matmul(c0Hw, wHc[:,:,ii])
#         #savemat('/home/hankm/matlab_ws/DepthNav/wHc.mat', {'w0Hc':wHc})
#
#         #######################################################################################3
#         # critical bug: orientation[0] is DEG while np.cos() and np.sin() takes radian !!!
#         # seems RECON's orientation is in radian ... though
#         # TODO: Change the orientation representation to Quaternion from orientation
#         # TODO Save time-stamp data for the correspoinding pose data pkl
#         # TODO Need to develop a func to read timestamp of depth data
#
#         #theta = float(orientation[0])  # hkm
#         #waypoints = to_local_coords(positions, positions[0], theta)  # orientation[0])
#         #goal_pos = to_local_coords(goal_pos, positions[0],   theta)  # orientation[0])
#         #delta_waypoints = waypoints[1:] - waypoints[0]
#
#         #flen = float( len(delta_waypoints) ) + 1.0 - 0.0001
#         #delta_time = np.arange(0.025, 0.025 * flen, 0.025 )  # s
#         #vel_x = delta_waypoints[:,0] / delta_time
#         #vel_y = delta_waypoints[:,1] / delta_time
#
#         # #############################################################################################################
#         # We now transform the P0 ~ P5 and th0 ~ th5 w.r.t P0 and th0
#         # i.e.) We transform the abs coordinate pose to the relative coordinate (w.r.t P0)
#         # Then, we normalize the tformed pts by the geometric max_goal distance to enforce it btwn 0 ~ 1
#         # i.e.) normalized pts : metric_waypoint_space * len_traj_pred
#         # (1) the transformed & normalized Pn1 ~ Pn5 are GT waypoints
#         # (2) the transformed & normalized Pn5 corresponds to the GT goal_pos
#         ###############################################################################################################
#         num_pos = len(positions)            # curr pos ~ curr pos + 5
#         wHc0_left = rm.xyzrpy_to_htm(np.array([positions[0, 0], positions[0, 1], 0, 0, 0, orientation[0]]))
#         wHc0 = rm.Left2Right( wHc0_left )
#         c0Hw = np.linalg.inv( wHc0 )
#         c0Hc = np.zeros([4, 4, num_pos])
#         wHc  = np.zeros([4, 4, num_pos])
#         relpose_2d  = np.zeros([num_pos, 3])  # relative cam pose w.r.t the init cam pose
#         waypoints   = np.zeros([num_pos, 2])
#         twist_2d    = np.zeros([num_pos, 3])
#         step_time   = np.ones([num_pos, 1]) * 0.025  # 40 FPS
#         step_time[0] = 0
#         cum_time = np.cumsum(step_time)
#         for ii in range(1, num_pos):
#             x = positions[ii, 0]
#             y = positions[ii, 1]
#             th = orientation[ii]
#             wHc_left = rm.xyzrpy_to_htm(np.array([x, y, 0, 0, 0, th]))
#             wHc [..., ii] = rm.Left2Right(wHc_left)
#             c0Hck = []
#             c0Hck = np.matmul(c0Hw, wHc[:, :, ii])
#             x, y, z, rol, pit, th = rm.htm_to_xyzrpy( c0Hck )
#             c0Hc[..., ii] = c0Hck
#             relpose_2d[ii, ...] = np.array([x, y, th])
#             waypoints[ii, ...] = np.array([x, y])
#             twist_2d[ii, ...] = relpose_2d[ii, ...] / step_time[ii]
#
#         # goal_pos w.r.t c0
#         wHce_left = rm.xyzrpy_to_htm( np.array([goal_pos[0], goal_pos[1], 0, 0, 0, goal_theta[0] ]) )  # abs  goal pose htm
#         wHce = rm.Left2Right(wHce_left)
#         c0Hce = np.matmul(c0Hw, wHce)   # relative goal pose htm
#         x_g, y_g, z_g, rol_g, pit_g, orientation_g = rm.htm_to_xyzrpy( c0Hce )  #
#         relgoal_2d = np.zeros(3)        # relative goal pose w.r.t
#         relgoal_2d[0] = x_g
#         relgoal_2d[1] = y_g
#         relgoal_2d[2] = orientation_g
#
#         # print(goal_pos)
#         # print(relgoal_2d)
#         assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
#
#         #if self.learn_angle:
#             #orientation = orientation[1:] - orientation[0]     #TODO:  Check if this simple subtraction is correct/wrong....
#             #actions = np.concatenate([waypoints[1:], orientation[:, None]], axis=-1)
#         #else:
#             #actions = waypoints[1:]
#
#         actions = waypoints[1:]
#
#         # out_debug_path = '/home/hankm/matlab_ws/DepthNav/compute_action.mat'
#         # mdic = {"positions": positions, "traj_data_pos": traj_data["position"], "curr_time": curr_time,
#         #         "goal_time": goal_time, "goal_idx": goal_idx, "goal_pos": goal_pos,
#         #         "goal_theta": goal_theta, "orientation": orientation, "wHc0": wHc0, "c0Hc": c0Hc,
#         #         "waypoints": waypoints, "wHce": wHce, "c0Hce": c0Hce, "relgoal_2d": relgoal_2d, "actions": actions}
#         # savemat(out_debug_path, mdic)
#
#         if self.normalize:
#             # BN
#             #actions[:, :2] /= 0.001 #MAX_TOGOAL_DIST  #(self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing * self.len_traj_pred)
#             #relgoal_2d[:2] /= 0.001 #MAX_TOGOAL_DIST #(self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing * self.len_traj_pred)
#             raise NotImplementedError
#
#         assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"
#         #return actions, goal_pos
#         return actions, twist_2d, relgoal_2d.squeeze()



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

        thetas = traj_data['orientation'][start_idx:end_idx].squeeze()  # (N,)
        positions = traj_data['position'][start_idx:end_idx]            # (N,2)

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

        # rel_xp, rel_zp, rel_pred_qs, rel_pred_qy, rel_xg, rel_zg, rel_goal_qs, rel_goal_qy =\
        #     _get_rel_cam_poses(self, [x0, y0, th0], [xp, yp, th_p], [xe, ye, th_e], self.dataset_name, rHc )

#        rel_xp, rel_yp, rel_pred_qs, rel_pred_qz, rel_xg, rel_yg, rel_goal_qs, rel_goal_qz = _get_rel_robot_poses([x0, y0, th0], [xp, yp, th_p], [xe, ye, th_e], self.dataset_name, bHc )
        rel_xp, rel_yp, rel_theta_p, rel_xg, rel_yg, rel_theta_g = _get_rel_pose_se2([x0, y0, th0], [xp, yp, th_p], [xe, ye, th_e], self.dataset_name, bHc)

        #TODO: Explore better way to normalize actions !!! ###
        if self.normalize:
            action = _normalize_pose(np.asarray([rel_xp, rel_yp, rel_theta_p]).transpose(), max_frame_dist=self.max_frame_dist, eta=1.0, dataset_name=self.dataset_name)
            goal = _normalize_pose(np.asarray([rel_xg, rel_yg, rel_theta_g]), max_frame_dist=self.max_frame_dist, eta=1.0, dataset_name=self.dataset_name)

        else:
            action = np.array([rel_xp, rel_yp, rel_theta_p], dtype='float32')
            goal = np.array([rel_xg, rel_yg, rel_theta_g], dtype='float32')

        #assert( np.linalg.norm(action[2:]) == 1), (f"action quat norm {np.linalg.norm(action[2:])}  should be 1,  quat ws {quat_p}, and   rol pit orientation were: {rol_p}, {pit_p}, {orientation_p}")

        if self.learn_angle:
            out_action  = action
            out_relgoal = goal
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

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
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
        f_curr, curr_time, max_goal_dist = self.index_to_data[i] # ( 'canteen/dynamic/part3/Capture_1', 5, 20 )
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist) # returns random goal  # ( 'canteen/dynamic/part3/Capture_1', 17, False)
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

        obs_rgbs = torch.cat([self._load_rgb(f, t) for f, t in context]) # hkm
        obs_depths = torch.cat([self._load_depth(f, t) for f, t in context])  # hkm
        # print("obs minmax: %f %f"%(torch.min(obs_depths), torch.max(obs_depths)) )
        # Load goal image
        goal_rgb = self._load_rgb(f_goal, goal_time)  # hkm
        goal_depth = self._load_depth(f_goal, goal_time)  # hkm

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        # actions and goal_pos are computed w.r.t the 1st image frame.
        actions, goal_pos = self._compute_next_actions(curr_traj_data, curr_time, goal_time) # actions.shape:  num_len_traj x (2 or 3)

        # Compute distances (geometric)
        if goal_is_negative:
            xy_distance = 2  # 100 m
        else:
            xy_distance = ( goal_pos[0]**2 + goal_pos[1]**2 ) ** 0.5  # relative & normalized distance

        if self.learn_angle:
            raise NotImplementedError
            if goal_is_negative:
                q_dist = 2
            else:
                q_pred = np.array( [actions[...,2],  0.0, 0.0, actions[3]], dtype='float32')
                q_goal = np.array( [goal_pos[2], 0.0, 0.0, goal_pos[3]], dtype='float32')
                q_dist = rm.quat_dist( q_pred, q_goal )
            distance = np.array([xy_distance, q_dist])
        else:
            distance = xy_distance
            #raise NotImplementedError

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        #assert( torch.norm(actions_torch[2:]) == 1, f"actions torch q norm {torch.norm(actions_torch[2:])} should be 1" )
        #assert (torch.linalg.vector_norm(actions_torch, dim=1) == 1, f"{np.linalg.norm(actions[2:])}  should be 1")
        action_mask = (1) # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound
            
        # Compute distances (temporal)
        #if goal_is_negative:
            #distance = self.max_dist_cat
        #else:
            #distance = (goal_time - curr_time) // self.waypoint_spacing
            #assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

        #actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        #if self.learn_angle:
            #actions_torch = calculate_sin_cos(actions_torch)    # x, y, cos(orientation), sin(orientation)
        
        #action_mask = ( # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound
            #(distance < self.max_action_distance) and       # max_action_distance = config[action][max_dist_cat] 10 in vint
            #(distance > self.min_action_distance) and       # min_action_distance = config[action][min_dist_cat] 0  in vint
            #(not goal_is_negative)
        #)
        obs_rgb_norm    = obs_rgbs   / 255.0
        goal_rgb_norm   = goal_rgb   / 255.0
        obs_depths_norm = obs_depths / MAX_DEPTH
        goal_depth_norm = goal_depth / MAX_DEPTH
        #data_info = {'f_curr': f_curr, 'curr_time': curr_time, 'f_goal': f_goal, 'goal_time': goal_time}
        data_info = f'{self.dataset_name} {f_curr} {curr_time} {f_goal} {goal_time} {goal_time - curr_time}'
        return (
            torch.as_tensor(obs_rgb_norm, dtype=torch.float32),
            torch.as_tensor(obs_depths_norm, dtype=torch.float32),
            torch.as_tensor(goal_rgb_norm, dtype=torch.float32),
            torch.as_tensor(goal_depth_norm, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.float32),  # geometric distance wrt base_link (on xy plane)
            torch.as_tensor(goal_pos, dtype=torch.float32),  # wrt base_link
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            data_info
        )
