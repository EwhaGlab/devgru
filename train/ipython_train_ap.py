import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
from os.path import dirname, abspath
BASE_DIR = '/home/hankm/python_ws/viznav/depth-nav'  
import sys
sys.path.append(BASE_DIR)
#print("base dir from train_isaac:", BASE_DIR)

from models.image_rnn.image_rnn import ImageRNN
#from depth_nav_train.data.depth_nav_dataset import
from data.collision_dataset import Collision_Dataset
from depth_nav_train.train_eval_loop import (
    train_eval_loop,
    load_model,
)

torch.multiprocessing.set_start_method("spawn")
parser = argparse.ArgumentParser(description="Depth Navigation ANN")

# project setup
parser.add_argument(
    "--config",
    "-c",
    default="config/depth_nav.yaml",
    type=str,
    help="Path to the config file in train_config folder",
)
args = parser.parse_args()

with open("/home/hankm/python_ws/viznav/depth-nav/config/defaults.yaml", "r") as f:
    default_config = yaml.safe_load(f)

config = default_config

#with open(args.config, "r") as f:
with open("/home/hankm/python_ws/viznav/depth-nav/config/depth_nav.yaml", "r") as f:
    user_config = yaml.safe_load(f)


config.update(user_config)

config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
config["project_folder"] = os.path.join(
    "logs", config["project_name"], config["run_name"]
)
os.makedirs(
    config[
        "project_folder"
    ],  # should error if dir already exists to avoid overwriting and old project
)
    

################################    
# main
################################

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "gpu_ids" not in config:
        config["gpu_ids"] = [0]
    elif type(config["gpu_ids"]) == int:
        config["gpu_ids"] = [config["gpu_ids"]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(x) for x in config["gpu_ids"]]
    )
    print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
else:
    print("Using cpu")

first_gpu_id = config["gpu_ids"][0]
device = torch.device(
    f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
)

if "seed" in config:
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cudnn.deterministic = True

cudnn.benchmark = True  # good if input sizes don't vary
transform = ([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#transform = ([transforms.Normalize(mean=[1, 1, 1], std=[0, 0, 0]),])
transform = transforms.Compose(transform) # data preprocessing ( e.g. data augmentation )

# Load the data
train_dataset = []
test_dataloaders = {}

if "context_type" not in config:
    config["context_type"] = "temporal"

if "clip_goals" not in config:
    config["clip_goals"] = False


for dataset_name in config["collision_datasets"]:
    data_config = config["collision_datasets"][dataset_name]
    if "negative_mining" not in data_config:
        data_config["negative_mining"] = True
    if "goals_per_obs" not in data_config:
        data_config["goals_per_obs"] = 1
    if "end_slack" not in data_config:
        data_config["end_slack"] = 0
    if "waypoint_spacing" not in data_config:
        data_config["waypoint_spacing"] = 1

    train_ = []
    test_ = []
    
    for data_split_type in ["train", "test"]:
        if data_split_type in data_config:
                dataset = Collision_Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],   # 1 by default b/c waypoint spacing is not set in the config file
                    min_frame_dist=config["distance"]["min_frame_dist"],
                    max_frame_dist=config["distance"]["max_frame_dist"],
                    #min_action_distance=config["action"]["min_dist_cat"],
                    #max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle= True, #config["learn_angle"],
                    context_size=config["context_size"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"],
                    goal_type=config["goal_type"],
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                    train_ = dataset
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset
                    test_ = dataset

train_dataset = ConcatDataset(train_dataset)
# next(iter(train_loader))  # to access a batch of dataset
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle= False, #True,
    num_workers=config["num_workers"],
    drop_last=False,
    persistent_workers=True,
)

#########################################################################3

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "depth-nav":
        model = DepthRNN(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            #late_fusion=config["late_fusion"],
            #mha_num_attention_heads=config["mha_num_attention_heads"],
            #mha_num_attention_layers=config["mha_num_attention_layers"],
            #mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())





################################################################
#               ViNT_Dataset ( data_name )
################################################################

dataset_name = 'former' #'thud' #'go_stanford'
data_config = config['datasets'][dataset_name]
if "negative_mining" not in data_config:
    data_config["negative_mining"] = True
if "goals_per_obs" not in data_config:
    data_config["goals_per_obs"] = 1
if "end_slack" not in data_config:
    data_config["end_slack"] = 0
if "waypoint_spacing" not in data_config:
    data_config["waypoint_spacing"] = 1

data_split_type = 'train'
data_folder=data_config["data_folder"]
data_split_folder=data_config[data_split_type]
image_size=config["image_size"]
waypoint_spacing=data_config["waypoint_spacing"]   # 1 by default b/c waypoint spacing is not set 
min_frame_dist=config["distance"]["min_frame_dist"]
max_frame_dist=config["distance"]["max_frame_dist"]
min_action_distance=config["action"]["min_frame_dist"]
max_action_distance=config["action"]["max_frame_dist"]
negative_mining=data_config["negative_mining"]
len_traj_pred=config["len_traj_pred"]
learn_angle=config["learn_angle"]
context_size=config["context_size"]
context_type=config["context_type"]
end_slack=data_config["end_slack"]
goals_per_obs=data_config["goals_per_obs"]
normalize=config["normalize"]
goal_type=config["goal_type"]


######################################################################
# Depth_Nav_Dataset init()
######################################################################
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

#from vint_train.data.data_utils import (
from data.data_utils import(
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
    _get_rel_cam_poses,
)


traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
with open(traj_names_file, "r") as f:
    file_lines = f.read()
    traj_names = file_lines.split("\n")
if "" in traj_names:
    traj_names.remove("")

    distance_range = list( range(min_frame_dist, max_frame_dist + 1, waypoint_spacing) )
    min_frame_dist = distance_range[0]
    max_frame_dist = distance_range[-1]

    if negative_mining:
        distance_range.append(-1)

    assert context_type in {
        "temporal",
        "randomized",
        "randomized_temporal",
    }, "context_type must be one of temporal, randomized, randomized_temporal"

    # load data/data_config.yaml
    with open( os.path.join("/home/hankm/python_ws/viznav/depth-nav/config", "data_config.yaml"), "r") as f:
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

#####################################################################3

    def _get_trajectory(trajectory_name):
        if trajectory_name in trajectory_cache:
            return trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            trajectory_cache[trajectory_name] = traj_data
            return traj_data


#########################   build_indexx        ###################################

def _build_index(use_tqdm: bool = False):
    """
    Build an index consisting of tuples (trajectory name, time, max goal distance)
    """
    samples_index = []  #( traj_name, temporal index, max_goal_distance )
    goals_index = []    #( traj_name, temporal index, 0 ~ end_idx )

    for traj_name in tqdm.tqdm(traj_names, disable=not use_tqdm, dynamic_ncols=True):
        traj_data = _get_trajectory(traj_name)
        traj_len = len(traj_data["position"])

        for goal_time in range(0, traj_len):
            goals_index.append((traj_name, goal_time))

        begin_time = context_size * waypoint_spacing
        end_time = traj_len - end_slack - waypoint_spacing
        for curr_time in range(begin_time, end_time):
            max_goal_distance = min(max_frame_dist * waypoint_spacing, traj_len - curr_time - 1)
            samples_index.append((traj_name, curr_time, max_goal_distance))

    return samples_index, goals_index


##########################        _load_index         ##############################

#def _load_index():

    index_to_data_path = os.path.join(
        data_split_folder,
        f"dataset_dist_{min_frame_dist}_to_{max_frame_dist}_context_{context_type}_n{context_size}_slack_{end_slack}.pkl",
    )
    try:
        # load the index_to_data if it already exists (to save time)
        with open(index_to_data_path, "rb") as f:
            index_to_data, goals_index = pickle.load(f)
    except:
        # if the index_to_data file doesn't exist, create it
        index_to_data, goals_index = _build_index()
        with open(index_to_data_path, "wb") as f:
            pickle.dump((index_to_data, goals_index), f)
    

###########################  Begining of _build_caches ###############################
#def _build_caches( use_tqdm: bool = True):  # hkm
    """
    Build a cache of images for faster loading using LMDB
    """
    use_tqdm = True
    rgb_cache_filename = os.path.join(
        data_split_folder,
        f"rgb_dataset_{dataset_name}.lmdb",
    )

    depth_cache_filename = os.path.join(
        data_split_folder,
        f"depth_dataset_{dataset_name}.lmdb",
    )
    # Load all the trajectories into memory. These should already be loaded, but just in case.
    for traj_name in traj_names:   # saves trajs in dict name: trajectory_cache
        _get_trajectory(traj_name)
   
    """
    If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
    """
    if not os.path.exists(rgb_cache_filename):
        tqdm_iterator = tqdm.tqdm( goals_index, disable=not use_tqdm, dynamic_ncols=True, desc=f"Building LMDB rgb cache for{dataset_name}")
        with lmdb.open(rgb_cache_filename, map_size=2**40) as image_cache:
            with image_cache.begin(write=True) as txn:
                for traj_name, time in tqdm_iterator:
                    rgb_path = get_data_path(traj_name, int(time), data_type="rgb")
                    # print("image_path: %s"%image_path )
                    with open(rgb_path, "rb") as f:
                        txn.put(rgb_path.encode(), f.read())
                        
    if not os.path.exists(depth_cache_filename):
        tqdm_iterator = tqdm.tqdm( goals_index, disable=not use_tqdm, dynamic_ncols=True, desc=f"Building LMDB depth cache for{dataset_name}")
        with lmdb.open(depth_cache_filename, map_size=2**40) as image_cache:
            with image_cache.begin(write=True) as txn:
                for traj_name, time in tqdm_iterator:
                    depth_path = get_data_path(traj_name, int(time), 'depth')
                    # print("image_path: %s"%image_path )
                    with open(depth_path, "rb") as f:
                        txn.put(depth_path.encode(), f.read())

    # Reopen the cache file in read-only mode
    _rgb_cache: lmdb.Environment = lmdb.open(rgb_cache_filename, readonly=True)
    _depth_cache: lmdb.Environment = lmdb.open(depth_cache_filename, readonly=True)

###########################  End of _build_caches ###############################

#def _build_index(use_tqdm: bool = False):
    """
    Build an index consisting of tuples (trajectory name, time, max goal distance)
    """
    samples_index = []  #( traj_name, temporal index, max_goal_distance )
    goals_index = []    #( traj_name, temporal index, 0 ~ end_idx )

    for traj_name in tqdm.tqdm(traj_names, disable=not use_tqdm, dynamic_ncols=True):
        traj_data = _get_trajectory(traj_name)
        traj_len = len(traj_data["position"])

        for goal_time in range(0, traj_len):
            goals_index.append((traj_name, goal_time))

        begin_time = context_size * waypoint_spacing
        end_time = traj_len - end_slack - waypoint_spacing
        for curr_time in range(begin_time, end_time):
            max_goal_distance = min(max_frame_dist * waypoint_spacing, traj_len - curr_time - 1)
            samples_index.append((traj_name, curr_time, max_goal_distance))
    #return samples_index, goals_index


#############################################################################

# run load index() as script...
#_load_index()  # <----- do not run this func ... we need index_to_data
#_build_caches()

if learn_angle:
    num_action_params = 4
else:
    num_action_params = 2


##################################################################################
#           get item
##################################################################################

def _sample_negative():
    """
    Sample a goal from a (likely) different trajectory.
    """
    return goals_index[np.random.randint(0, len(goals_index))]


def _sample_goal(trajectory_name, curr_time, max_goal_dist):
    """
    Sample a goal from the future in the same trajectory.
    Returns: (trajectory_name, goal_time, goal_is_negative)
    """
    goal_offset = np.random.randint(0, max_goal_dist + 1)       # 1 out of 6 chooses a neg sample
    if goal_offset == 0:
        trajectory_name, goal_time = _sample_negative()    # select a different traj
        return trajectory_name, goal_time, True
    else:
        goal_time = curr_time + int(goal_offset * waypoint_spacing)
        return trajectory_name, goal_time, False


def _load_rgb(trajectory_name, time):   # hkm
    rgb_path = get_colldata_path(trajectory_name, time, 'rgb')

    try:
        with _rgb_cache.begin() as txn:
            rgb_buffer = txn.get(rgb_path.encode())
            rgb_bytes = bytes(rgb_buffer)
        rgb_bytes = io.BytesIO(rgb_bytes)
        
        return img_path_to_data(rgb_bytes, image_size)
    except TypeError:
        print(f"Failed to load image {rgb_path}")


def _load_depth(trajectory_name, time):  # hkm 
    depth_path = get_data_path(trajectory_name, time, 'depth')

    try:
        with _depth_cache.begin() as txn:
            depth_buffer = txn.get(depth_path.encode())
            depth_bytes = bytes(depth_buffer)
        depth_bytes = io.BytesIO(depth_bytes)
        
        return img_path_to_data(depth_bytes, image_size)
    except TypeError:
        print(f"Failed to load depth {depth_path}")


def _compute_prev_actions(self, traj_data, curr_time, context_times):  # goal_time : temporal idx of sg in the given traj

    #######################################################################################3
    # traj_data must be base_link's motion data. i.e, curr base_link wrt origin (base_link)

    curr_idx = curr_time
    prev_idx  = context_times[:-1]
    num_prev  = len(prev_idx)

    thetas = traj_data['orientation'][prev_idx].squeeze()  # (N,)
    positions = traj_data['position'][prev_idx]            # (N,2)

    xc, yc  = traj_data['position'][curr_idx].squeeze()    # robot traj data
    th_c    = traj_data['orientation'][curr_idx][0]
    # curr (xc,yc,th_c) -------- pred (xp,yp,th_p) -------- goal (xf, yf, th_f)
    # pred xy and th are the cam pose at the middle of cam traj between the curr and the goal location
    #assert (goal_idx - curr_time >= self.len_traj_pred), (f"goal_idx - curr_time {goal_idx - curr_time} should be GEQ to len_traj_pred")
    xps = positions[:-1, 0]   # shape: (len_traj_pred,)
    yps = positions[:-1, 1]   # shape: (len_traj_pred,)
    th_ps = thetas[:-1]       # shape: (len_traj_pred,)

    if self.dataset_name == 'former':
        bHc = rm.xyzrpy_to_htm( self.data_config['camera_matrics']['cam_wrt_base'] )
    else:
        bHc = np.eye(4)

    # rel_xp, rel_zp, rel_pred_qs, rel_pred_qy, rel_xg, rel_zg, rel_goal_qs, rel_goal_qy =\
    #     _get_rel_cam_poses(self, [x0, y0, th0], [xp, yp, th_p], [xe, ye, th_e], self.dataset_name, rHc )

#        rel_xp, rel_yp, rel_pred_qs, rel_pred_qz, rel_xg, rel_yg, rel_goal_qs, rel_goal_qz = _get_rel_robot_poses([x0, y0, th0], [xp, yp, th_p], [xe, ye, th_e], self.dataset_name, bHc )
    rel_xp, rel_yp, rel_theta_p = _get_rel_pose_se2([x0, y0, th0], [xps, yps, th_ps], self.dataset_name, bHc)

    #TODO: Explore better way to normalize actions !!! ###
    if self.normalize:
        action = _normalize_pose(np.asarray([rel_xp, rel_yp, rel_theta_p]).transpose(), max_frame_dist=self.max_frame_dist, eta=1.0, dataset_name=self.dataset_name)
        
    else:
        action = np.array([rel_xp, rel_yp, rel_theta_p], dtype='float32').transpose()
        
    #assert( np.linalg.norm(action[2:]) == 1), (f"action quat norm {np.linalg.norm(action[2:])}  should be 1,  quat ws {quat_p}, and   rol pit orientation were: {rol_p}, {pit_p}, {orientation_p}")

    if self.learn_angle:
        # conv angle to quat rep
        out_action = np.zeros([num_prev, 4], dtype='float32')
        for ii in range(0, num_prev):
            q_a = rm.rpy2quat(0, 0, action[ii, 2])
            assert round( math.sqrt( q_a[0]*q_a[0] + q_a[-1]*q_a[-1]), 3 )== 1.0, f"Is q_a: {q_a} unit quat ? "
            out_action[ii] = np.concatenate( (action[ii, :2], np.array([q_a[0], q_a[-1]]) ), axis=0 )
    else:
        out_action  = action [...,:2]

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
        action = _normalize_pose(np.asarray([rel_xp, rel_yp, rel_theta_p]).transpose(), waypoint_spacing=self.waypoint_spacing, eta=1.0, dataset_name=self.dataset_name)
        goal = _normalize_pose(np.asarray([rel_xg, rel_yg, rel_theta_g]), waypoint_spacing=self.waypoint_spacing, eta=1.0, dataset_name=self.dataset_name)

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





def _load_data_test(traj_data, curr_time, goal_time): # goal_time : temporal idx of sg in the given traj
    start_index = curr_time
    end_index = curr_time + len_traj_pred * waypoint_spacing + 1
    yaw = traj_data["yaw"][start_index:end_index:waypoint_spacing]
    positions = traj_data["position"][start_index:end_index:waypoint_spacing]
    goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (len_traj_pred + 1,):
        const_len = len_traj_pred + 1 - yaw.shape[0]
        yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
        positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

    assert yaw.shape == (len_traj_pred + 1,), f"{yaw.shape} and {(len_traj_pred + 1,)} should be equal"
    assert positions.shape == (len_traj_pred + 1, 2), f"{positions.shape} and {(len_traj_pred + 1, 2)} should be equal"
    if len(yaw.shape) ==1:
        theta = yaw[0]
    else:
         theta = yaw[0][0]   
    waypoints_local = to_local_coords(positions, positions[0], theta)
    goal_pos_local = to_local_coords(goal_pos, positions[0],   theta)
    return positions, yaw, waypoints_local, goal_pos_local


########################## ___getitem___ ########################################

f_curr, curr_time, max_goal_dist = index_to_data[0]
f_goal, goal_time, goal_is_negative = _sample_goal(f_curr, curr_time, max_goal_dist)
    

if context_type == "temporal":
    # sample the last context_size times from interval [0, curr_time)
    context_times = list(
        range(
            curr_time + -context_size * waypoint_spacing,
            curr_time + 1,
            waypoint_spacing,
        )
    )
    context = [(f_curr, t) for t in context_times]
else:
    raise ValueError(f"Invalid context type {context_type}")


obs_rgb = torch.cat([_load_rgb(f, t) for f, t in context])      # hkm
obs_depth = torch.cat([_load_depth(f, t) for f, t in context])  # hkm

goal_rgb = _load_rgb(f_goal, goal_time)         # hkm
goal_depth = _load_depth(f_goal, goal_time)     # hkm

# Load other trajectory data
curr_traj_data = _get_trajectory(f_curr)
curr_traj_len = len(curr_traj_data["position"])
assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

goal_traj_data = _get_trajectory(f_goal)
goal_traj_len = len(goal_traj_data["position"])
assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

# Compute actions
actions, goal_pos = _compute_actions(curr_traj_data, curr_time, goal_time)

pos, yaw, wp_l, goalpos_l = _load_data_test(curr_traj_data, curr_time, goal_time)

# Compute distances (temporal)
if goal_is_negative:
    distance = max_dist_cat
else:
    distance = (goal_time - curr_time) // waypoint_spacing
    assert (goal_time - curr_time) % waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {waypoint_spacing}"

actions_torch = torch.as_tensor(actions, dtype=torch.float32)
if learn_angle:
    actions_torch = calculate_sin_cos(actions_torch)    # x, y, cos(yaw), sin(yaw)

action_mask = ( # checks if the computed distance to subgoal is bounded btwn the action (temporal waypoint) bound
    (distance < max_action_distance) and       # max_action_distance = config[action][max_dist_cat] 10 in vint
    (distance > min_action_distance) and       # min_action_distance = config[action][min_dist_cat] 0  in vint
    (not goal_is_negative)
)

























    folder_names = [
        f
        for f in input_dirs
        if os.path.isdir(os.path.join(data_dir, f))
        and "traj_data.pkl" in os.listdir(os.path.join(data_dir, f, 'RGB'))
    ]


    split_index = int(split * len(folder_names))
    train_folder_names = folder_names[:split_index]
    test_folder_names = folder_names[split_index:]



    # Create directories for the train and test sets
    train_dir = os.path.join(data_splits_dir, dataset_name, "train")
    test_dir = os.path.join(data_splits_dir, dataset_name, "test")
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")
            remove_files_in_dir(dir_path)
        else:
            print(f"Creating {dir_path}")
            os.makedirs(dir_path)

    # Write the names of the train and test folders to files
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for folder_name in train_folder_names:
            f.write(folder_name + "\n")

    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for folder_name in test_folder_names:
            f.write(folder_name + "\n")
