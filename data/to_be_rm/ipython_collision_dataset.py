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
BASE_DIR = '/home/hankm/python_ws/viznav/depth-nav' #dirname(dirname(abspath(__file__)))
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
)
import cv2
MAX_DEPTH = 65535.0
#MAX_TOGOAL_DIST = 2.412 # when max frame diff = 80
#MAX_TOGOAL_DIST =  0.6085 # when max frame diff = 20
from scipy.io import savemat


    def _get_trajectory(trajectory_name):
        if trajectory_name in trajectory_cache:
            return trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            trajectory_cache[trajectory_name] = traj_data
            return traj_data


"""
Main DepthNav dataset class  (input)

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

data_split_folder = '/media/mydata/viznav/data_splits/former_collision/ws3/train'
traj_names_file = os.path.join(data_split_folder, "traj_names.txt")

with open(traj_names_file, "r") as f:
    file_lines = f.read()
    traj_names = file_lines.split("\n")
if "" in traj_names:
    traj_names.remove("")

dataset_name = 'former'
nav_config = config["collision_datasets"][dataset_name]
if "negative_mining" not in nav_config:
    nav_config["negative_mining"] = True
if "goals_per_obs" not in nav_config:
    nav_config["goals_per_obs"] = 1
if "end_slack" not in nav_config:
    nav_config["end_slack"] = 0
if "waypoint_spacing" not in nav_config:
    nav_config["waypoint_spacing"] = 1
data_folder = nav_config['data_folder']


image_size = config['image_size']
waypoint_spacing = config['collision_datasets'][dataset_name]['waypoint_spacing']
#self.distance_categories = list( range(min_frame_dist, max_frame_dist + 1, self.waypoint_spacing) )
min_frame_dist = config['distance']['min_frame_dist']     #  self.distance_categories[0]
max_frame_dist = config['distance']['max_frame_dist']     #  self.distance_categories[-1]
negative_mining = config['collision_datasets'][dataset_name]['negative_mining']
#if self.negative_mining:
    #self.distance_categories.append(-1)
goal_dist_type = config['goal_dist_type']
len_traj_pred = config['len_traj_pred']
learn_angle = config['learn_angle']

context_size = config['context_size']

context_type = config['context_type']
end_slack = 3 #config['end_slack']
goals_per_obs = 1 #config['goals_per_obs']
normalize = config['normalize']
obs_type = config['obs_type']
goal_type = config['goal_type']

    # load data/data_config.yaml
with open(
    os.path.join(BASE_DIR + "/config/data_config.yaml"), "r"
) as f:
    all_data_config = yaml.safe_load(f)
assert (
    dataset_name in all_data_config
), f"Dataset {dataset_name} not found in data_config.yaml"
dataset_names = list(all_data_config.keys())
dataset_names.sort()
    
# use this index to retrieve the dataset name from the data_config.yaml
dataset_index = dataset_names.index(dataset_name)
data_config = all_data_config[dataset_name]
trajectory_cache = {}

_rgb_cache = []
_depth_cache = []
_goal_rgb_cache = []
_goal_depth_cache = []

##########################################################################
#self._load_index()          # sample_index = (traj_name, curr_time, max_goal_distance),  goals_index = (traj_name, 0~max_idx)
##########################################################################

"""
Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
"""

index_to_data_path = os.path.join(
    data_split_folder,
    f"dataset_context_{context_type}_n{context_size}_slack_{end_slack}.pkl",)

use_tqdm = True
try:
    # load the index_to_data if it already exists (to save time)
    with open(index_to_data_path, "rb") as f:
        index_to_data, goals_index = pickle.load(f)
except:
    # if the index_to_data file doesn't exist, create it
    # index_to_data, goals_index = self._build_index()
    index_to_data = []  #( traj_name, temporal index, max_goal_distance )
    goals_index = []    #( traj_name, temporal index, 0 ~ end_idx )

    for traj_name in tqdm.tqdm(traj_names, disable=not use_tqdm, dynamic_ncols=True):
        traj_data = _get_trajectory(traj_name) # x, y, theta
        traj_len = len(traj_data["position"])

        for goal_time in range(0, traj_len):
            goals_index.append((traj_name, goal_time))

        begin_time = context_size * waypoint_spacing
        end_time = traj_len - end_slack - len_traj_pred * waypoint_spacing
        for curr_time in range(begin_time, end_time):
            max_goal_distance = min(max_frame_dist * waypoint_spacing, traj_len - curr_time - 1)
            index_to_data.append((traj_name, curr_time, max_goal_distance))

    with open(index_to_data_path, "wb") as f:
        pickle.dump((index_to_data, goals_index), f)


##########################################################################
#self._build_caches()        # build LMDB cache for depth images        
##########################################################################
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

    def _build_caches(use_tqdm: bool = True):

        rgb_cache_filename = os.path.join(
            data_split_folder,
            f"rgb_dataset_{dataset_name}.lmdb",
        )
        depth_cache_filename = os.path.join(
            data_split_folder,
            f"depth_dataset_{dataset_name}.lmdb",
        )
        
        goal_rgb_cache_filename = os.path.join(
            data_split_folder,
            f"goal_rgb_dataset_{dataset_name}.lmdb",
        )
        goal_depth_cache_filename = os.path.join(
            data_split_folder,
            f"goal_depth_dataset_{dataset_name}.lmdb",
        )

        
        for traj_name in traj_names:   # saves trajs in dict name: self.trajectory_cache
            _get_trajectory(traj_name)

        if not os.path.exists(rgb_cache_filename):
            tqdm_iterator = tqdm.tqdm(
                samples_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building RGB LMDB cache for {dataset_name}"
            )
            with lmdb.open(rgb_cache_filename, map_size=2**40) as rgb_cache:
                with rgb_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        rgb_path = get_colldata_path(traj_name, int(time), 'rgb')
                        with open(rgb_path, "rb") as f:
                            txn.put(rgb_path.encode(), f.read())

        if not os.path.exists(goal_rgb_cache_filename):
            tqdm_iterator = tqdm.tqdm(
                goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building RGB LMDB cache for {dataset_name}"
            )
            with lmdb.open(goal_rgb_cache_filename, map_size=2**40) as goal_rgb_cache:
                with goal_rgb_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        goal_rgb_path = get_colldata_path(traj_name, int(time), 'rgb')
                        with open(goal_rgb_path, "rb") as f:
                            txn.put(goal_rgb_path.encode(), f.read())

        if not os.path.exists(depth_cache_filename):
            tqdm_iterator = tqdm.tqdm(
                goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building Depth LMDB cache for {dataset_name}"
            )
            with lmdb.open(depth_cache_filename, map_size=2**40) as depth_cache:
                with depth_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        depth_path = get_data_path(traj_name, int(time), 'depth')
                        dm = cv2.imread(depth_path, -1)  # read as uint16
                        dm_bytes = cv2.imencode('.png', dm)[1].tobytes()
                        txn.put(depth_path.encode(), dm_bytes)

        _rgb_cache: lmdb.Environment = lmdb.open(rgb_cache_filename, readonly=True)
        _depth_cache: lmdb.Environment = lmdb.open(depth_cache_filename, readonly=True)
        _goal_rgb_cache: lmdb.Environment = lmdb.open(goal_rgb_cache_filename, readonly=True)
        _goal_depth_cache: lmdb.Environment = lmdb.open(goal_depth_cache_filename, readonly=True)



    def _build_index(use_tqdm: bool = False):
        samples_index = []  #( traj_name, temporal index, max_goal_distance )
        goals_index = []    #( traj_name, temporal index, 0 ~ end_idx )

        for traj_name in tqdm.tqdm(traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = _get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

 #           for goal_time in range(0, traj_len):
            goals_index.append((traj_name, -1)) # just add dummy number

            #begin_time = context_size * waypoint_spacing
            #end_time = len_traj_pred * waypoint_spacing
            #for curr_time in range(begin_time, end_time): # 0 ~ end_time
                #max_goal_distance = min(self.max_frame_dist * self.waypoint_spacing, traj_len - curr_time - 1)
                #samples_index.append((traj_name, curr_time, max_goal_distance))

            context_indexs = np.loadtxt('%s/context_index.txt'%traj_name, dtype=int)
            curr_time = context_indexs[-1]
            
            for index in context_indexs:
                samples_index.append((traj_name, index))
                
            
        return samples_index, goals_index       # goals_index is needed for "sample_negative().  i.e.) neg random sampling"

    #def _sample_goal(self, trajectory_name, curr_time, max_frame_goal_dist):

        #goal_offset = np.random.randint(1, max_frame_goal_dist + 1)  
        #if goal_offset == 0:
            #assert 0 # something went wrong...
            #trajectory_name, goal_time = self._sample_negative()    # select a different traj
            #return trajectory_name, goal_time, True
        #else:
            #goal_time = curr_time + goal_offset  #* self.waypoint_spacing
            #return trajectory_name, goal_time, False

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
            data_split_folder,
            f"dataset_context_{context_type}_n{context_size}.pkl",)
        
        #index_to_data_path = os.path.join(
            #self.data_split_folder,
            #f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",)

        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                index_to_data, goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            index_to_data, goals_index = _build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((index_to_data, goals_index), f)


    def _load_rgb(trajectory_name, time):
        rgb_path = get_colldata_path(trajectory_name, time, 'rgb')
        try:
            with _rgb_cache.begin() as txn:
                rgb_buffer = txn.get(rgb_path.encode())
                rgb_bytes = bytes(rgb_buffer)
            rgb_bytes = io.BytesIO(rgb_bytes)
            return img_path_to_data(rgb_bytes, image_size)
        except TypeError:
            print(f"Failed to load rgb image {rgb_path} @ {time}")

    def _load_depth(trajectory_name, time):
        depth_path = get_colldata_path(trajectory_name, time, 'depth')
        try:
            with _depth_cache.begin() as txn:
                depth_buffer = txn.get(depth_path.encode())
                depth_bytes = bytes(depth_buffer)
            depth_bytes = io.BytesIO(depth_bytes)
            return img_path_to_data(depth_bytes, image_size)
        except TypeError:
            print(f"Failed to load depth image {depth_path}")



    #def _load_rgb(rgb_path):
        #try:
            #return img_path_to_data(rgb_path, image_size)
        #except TypeError:
            #print(f"Failed to load rgb image {rgb_path} @ {time}")

    #def _load_depth(trajectory_name, time):
        #try:
            #return img_path_to_data(depth_path, image_size)
        #except TypeError:
            #print(f"Failed to load depth image {depth_path}")

    #def _load_rgb(rgb_path):
        #try:
            #return img_path_to_data(rgb_path, image_size)
        #except TypeError:
            #print(f"Failed to load rgb image {rgb_path} @ {time}")

    #def _load_depth(depth_path):
        #try:
            #return img_path_to_data(depth_path, image_size)
        #except TypeError:
            #print(f"Failed to load depth image {depth_path}")



    def _load_context_actions(data_dir):  # goal_time : temporal idx of sg in the given traj
        #######################################################################################3
        # context_times: 0 ~ 5  if curr_time is 5
        
        traj_file = '%s/pose_context_m.txt' % data_dir
        traj_data = np.loadtxt(traj_file) 
        
        quats = traj_data[:,3:]  # (N,)
        positions = traj_data[:,:2]
        num_context = len(positions)
        #xc, yc = positions[-1] 
        
        xps = positions[:, 0]  # shape: (len_traj_pred,)
        yps = positions[:, 1]  # shape: (len_traj_pred,)
        qw = quats[:,0]
        qz = quats[:,3]
        thetas = 2 * np.arctan2(qz, qw)
        
        th_ps = thetas  # shape: (len_traj_pred,)

        #if self.dataset_name == 'former':
            #bHc = rm.xyzrpy_to_htm(self.data_config['camera_matrics']['cam_wrt_base'])
        #else:
            #bHc = np.eye(4)
        #rel_xp, rel_yp, rel_theta_p = _get_rel_pose_se2(np.array([xc, yc, th_c]), np.array([xps, yps, th_ps]), self.dataset_name, bHc)
        rel_xp = xps; rel_yp = yps; rel_theta_p = th_ps

        # TODO: Explore better way to normalize actions !!! ###
        if normalize:
            action = _normalize_pose(np.asarray([rel_xp, rel_yp, rel_theta_p]).transpose(),
                                     waypoint_spacing=waypoint_spacing, eta=1.0, dataset_name=dataset_name)

        else:
            action = np.array([rel_xp, rel_yp, rel_theta_p], dtype='float32').transpose()

        if learn_angle:
            # conv angle to quat rep
            out_action = np.zeros([num_context, 4], dtype='float32')
            for ii in range(0, num_context):
                q_a = rm.rpy2quat(0, 0, action[ii, 2])
                assert round(math.sqrt(q_a[0] * q_a[0] + q_a[-1] * q_a[-1]), 3) == 1.0, f"Is q_a: {q_a} unit quat ? "
                out_action[ii] = np.concatenate((action[ii, :2], np.array([q_a[0], q_a[-1]])), axis=0)
        else:
            out_action = action[..., :2]

        return out_action


    def _load_next_actions(data_dir):  # goal_time : temporal idx of sg in the given traj
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

        # load sg
        sg_data_file = '%s/new_subgoal_m.txt'%data_dir
        sg_data = np.loadtxt(sg_data_file) # [xg_old, yg_old, qw_old, qz_old; xg_new, yg_new, qw_new, qz_new]
        xg = sg_data[0]
        yg = sg_data[1]
        qw_g = sg_data[2]
        qz_g = sg_data[3]
        th_g = 2 * np.arctan2(qz_g, qw_g)
        
        
        # load waypoint
        traj_data_file = '%s/corrected_waypoints_m.txt'%(data_dir)
        traj_data = np.loadtxt(traj_data_file) 

        xp = traj_data[:,0]   
        yp = traj_data[:,1]
        qw = traj_data[:,2]
        qz = traj_data[:,3]
        th_p = 2 * np.arctan2( qw, qz )
    
        #if dataset_name == 'former':
            #bHc = rm.xyzrpy_to_htm( data_config['camera_matrics']['cam_wrt_base'] )
        #else:
            #bHc = np.eye(4)

        rel_xp = xp; rel_yp = yp; rel_theta_p = th_p
        rel_xg = xg; rel_yg = yg; rel_theta_g = th_g
        
        #rel_xp, rel_yp, rel_theta_p = _get_rel_pose_se2(np.array([x0, y0, th0]), np.array([xp, yp, th_p]), self.dataset_name, bHc)
        #rel_xg, rel_yg, rel_theta_g = _get_rel_pose_se2(np.array([x0, y0, th0]), np.array([xe, ye, th_e]), self.dataset_name, bHc)

        #TODO: Explore better way to normalize actions !!! ###
        if normalize:
            action = _normalize_pose(np.asarray([rel_xp, rel_yp, rel_theta_p]).transpose(), waypoint_spacing=waypoint_spacing, eta=1.0, dataset_name=dataset_name)
            goal = _normalize_pose(np.asarray([rel_xg, rel_yg, rel_theta_g]), waypoint_spacing=waypoint_spacing, eta=1.0, dataset_name=dataset_name)

        else:
            action = np.array([rel_xp, rel_yp, rel_theta_p], dtype='float32').transpose()
            goal = np.array([rel_xg, rel_yg, rel_theta_g], dtype='float32')

        #assert( np.linalg.norm(action[2:]) == 1), (f"action quat norm {np.linalg.norm(action[2:])}  should be 1,  quat ws {quat_p}, and   rol pit orientation were: {rol_p}, {pit_p}, {orientation_p}")

        if learn_angle:
            # conv angle to quat rep
            out_action = np.zeros([len_traj_pred, 4], dtype='float32')
            for ii in range(0, len_traj_pred):
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





    def __len__(self) -> int:
        return len(self.index_to_data)

#####################################################################################################
#   I don't need most of functions used to present DepthNav_dataset such as _build_caches()  
#
#    def __getitem__(self, i: int) -> Tuple:
#####################################################################################################

    #f_curr, curr_time, max_frame_goal_dist = self.index_to_data[i]  # ( obs_path, sidx, max_frame_goal_dist (max_goal_dist * waypoint_spacing) )
    #f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_frame_goal_dist) 
    # returns random goal  # ( , traj_len * waypoint_spacing, False)

    # f_goal : traj name, goal_time : temporal idx of future sg, goal_is_negative: True or False
    # Load images

    # TODO: eliminate self.context_type parameter in the future    
    
    #context = []
    #if self.context_type == "temporal":
        #context_times = list(
            #range(
                #curr_time + -self.context_size * self.waypoint_spacing,
                #curr_time + 1,
                #self.waypoint_spacing,
            #)
        #)
        #context = [(f_curr, t) for t in context_times]
    #else:
        #raise ValueError(f"Invalid context type {self.context_type}")

    rgb_paths = sorted(glob.glob(os.path.join(data_dir, 'rgb*.png')))
    dep_paths = sorted(glob.glob(os.path.join(data_dir, 'depth*.png')))
    goal_rgb_path = rgb_paths[-1]
    goal_dep_path = dep_paths[-1]
    obs_rgb_paths = rgb_paths[0:-1]
    obs_dep_paths = dep_paths[0:-1]

    obs_rgbs = torch.cat([_load_rgb(obs_rgb_path) for obs_rgb_path in obs_rgb_paths])        #
    obs_depths = torch.cat([_load_depth(obs_dep_path) for obs_dep_path in obs_dep_paths])    # hkm
    
    # print("obs minmax: %f %f"%(torch.min(obs_depths), torch.max(obs_depths)) )
    # Load goal image
    
    goal_rgb = _load_rgb(goal_rgb_path)  #_load_rgb(goal_rgb_path)  # hkm
    goal_depth = _load_depth(goal_dep_path)  #_load_depth(goal_depth_path)  # hkm
    
    actions, goal_pos = _load_next_actions( data_dir ) # actions.shape:  num_len_traj x (2 or 3)
    context_actions = _load_context_actions( data_dir ) # context actions : previous actions

    # Compute distances (temporal or spatial)

    if goal_dist_type == "spatial":
        if goal_is_negative:
            xy_distance = max_frame_dist * data_config["metric_frame_spacing"]   #
        else:
            xy_distance = ( goal_pos[0]**2 + goal_pos[1]**2 ) ** 0.5  # relative distance
            xy_distance = xy_distance * ( data_config["metric_frame_spacing"] * waypoint_spacing )   # normalized relative distance

        if learn_angle:
            if goal_is_negative:
                q_dist = 1
            else:
                q_curr = rm.rpy2quat(0, 0, 0)
                q_goal = np.array([ goal_pos[2], 0, 0, goal_pos[-1] ], dtype='float32')
                q_dist, angle_diff_rad = rm.quat_ang_dist( q_curr, q_goal )
            distance = np.array([xy_distance, q_dist]) #angle_diff_rad])
        else:
            distance = xy_distance
    else:
        print("Unknown goal_dist_type \n")
        raise NotImplementedError

    actions_torch = torch.as_tensor(actions, dtype=torch.float32)
    context_actions_torch = torch.as_tensor(context_actions, dtype=torch.float32)
    goal_torch = torch.as_tensor(goal_pos, dtype=torch.float32)
    #assert( torch.norm(actions_torch[2:]) == 1, f"actions torch q norm {torch.norm(actions_torch[2:])} should be 1" )
    #assert (torch.linalg.vector_norm(actions_torch, dim=1) == 1, f"{np.linalg.norm(actions[2:])}  should be 1")
    action_mask = (1) # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound

    #actions_torch = torch.as_tensor(actions, dtype=torch.float32)
    #if self.learn_angle:
        #actions_torch = calculate_sin_cos(actions_torch)    # x, y, cos(orientation), sin(orientation)
    
    #action_mask = ( # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound
        #(distance < self.max_action_distance) and       # max_action_distance = config[action][max_dist_cat] 10 in vint
        #(distance > self.min_action_distance) and       # min_action_distance = config[action][min_dist_cat] 0  in vint
        #(not goal_is_negative)
    #)
    obs_rgb_norm    = obs_rgbs   # normalized when load_rgb is called
    goal_rgb_norm   = goal_rgb   
    obs_depths_norm = obs_depths
    goal_depth_norm = goal_depth
    data_info = f'{data_dir}'
    return (
        torch.as_tensor(obs_rgb_norm, dtype=torch.float32),
        torch.as_tensor(obs_depths_norm, dtype=torch.float32),
        torch.as_tensor(goal_rgb_norm, dtype=torch.float32),
        torch.as_tensor(goal_depth_norm, dtype=torch.float32),
        actions_torch,
        context_actions_torch,
        goal_torch,         # wrt base_link
        torch.as_tensor(distance, dtype=torch.float32),  # geometric distance wrt base_link (on xy plane)
        torch.as_tensor(self.dataset_index, dtype=torch.int64),
        torch.as_tensor(action_mask, dtype=torch.float32),
        data_info
    )
