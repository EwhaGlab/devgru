import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

import math
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

from os.path import dirname, abspath
#BASE_DIR = '/home/hankm/python_ws/viznav/depth-nav' #
BASE_DIR = dirname(dirname(abspath(__file__)))
print(BASE_DIR)

import sys
sys.path.append(BASE_DIR)
import utils.rigid_motion as rm
from deployment.src.nav_utils import msg_to_pil, to_numpy, transform_images, load_model, msg_to_caminfo, set_tform
import re


from visualize.visualizer import Visualizer, CostmapParams
from data.collision_dataset_simple import Collision_Dataset_Simple

from data.data_utils import(
    MAX_DEPTH,
    #_normalize_waypoints,
    #_denormalize_waypoints,
    #_normalize_context_poses,
    #_denormalize_context_poses,
    _normalize_subgoal,
    _denormalize_subgoal,
    _normalize_pose,
    _denormalize_pose,
    _get_rel_pose_se2,
    resize_and_aspect_crop
)

# from utils.visualizing.visualize_utils import (
#     to_numpy,
#     numpy_to_img,
#     VIZ_IMAGE_SIZE,
#     RED,
#     GREEN,
#     BLUE,
#     CYAN,
#     YELLOW,
#     MAGENTA,
# )
#import utils.action_utils as au

# from data.data_utils import(
#     _normalize_pose,
#     _denormalize_pose,
#     _get_rel_pose_se2,
#     resize_and_aspect_crop
# )

# UTILS
from deployment.src.topic_names import (RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO,
                        WAYPOINT_TOPIC,
                        ODOM_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)

VIZ_IMG_SIZE = (120, 160)

# CONSTANTS
DEPLOYMENT_DIR = BASE_DIR + "/deployment"
TOPOMAP_IMAGES_DIR = "%s/topomaps" % DEPLOYMENT_DIR
MODEL_WEIGHTS_PATH = "%s/model_weights"%DEPLOYMENT_DIR
ROBOT_CONFIG_PATH = "%s/config/robot.yaml"%DEPLOYMENT_DIR
MODEL_CONFIG_PATH = "%s/config/models.yaml"%DEPLOYMENT_DIR
MODEL_PARAM_PATH = '%s/config/depth_nav.yaml'%BASE_DIR
DATA_CONFIG_PATH = '%s/config/data_config.yaml'%BASE_DIR

with open(MODEL_PARAM_PATH, "r") as f:
    model_params = yaml.safe_load(f)

test_params = model_params['test']
deployment_params = model_params['deployment']

with open(DATA_CONFIG_PATH, "r") as f:
    data_config = yaml.safe_load(f)

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

img_width, img_height = model_params["image_size"]
normalize = model_params['normalize']
len_traj_pred = model_params['len_traj_pred']

def predict_collision(model: nn.Module, context_depth_queue):
    img_width, img_height = model_params["image_size"]
    context_size = model_params['context_size']
    dataset_name = deployment_params['dataset_name']
    model_type = deployment_params['col_model_type']
    waypoint_spacing = deployment_params['waypoint_spacing']
    max_frame_dist = model_params['distance']['max_frame_dist']
    learn_angle = model_params['learn_angle']

    obs_queue_size = context_size + 1
    ts_obs_depths = torch.zeros([obs_queue_size, img_height, img_width])
    for idx in range(0, obs_queue_size):
        pil_depth = context_depth_queue[idx]
        ts_resized_depth = resize_and_aspect_crop(pil_depth, [img_width, img_height])  # [C, h, w]
        assert ts_resized_depth.max() <= 1, f"obs depths not normalized well {ts_resized_depth.max()}"
        ts_obs_depths[idx] = ts_resized_depth

    ts_obs_depths = ts_obs_depths[None, ...]
    ts_obs_depths = ts_obs_depths.to(device)
    ts_obs_depth_curr = ts_obs_depths[:, -1:, :, :]

    if model_type == 'depth_coll':
        model_outputs = model(ts_obs_depth_curr)
    else:
        raise ValueError("unknown model type")

    ts_coll_logit = model_outputs  # (1,5,4),  (1,2)
    coll_logit = ts_coll_logit.item()

    return coll_logit

def predict_actions(model: nn.Module, context_depth_queue, context_pose_queue,
                    sg_depth, curr_rel_sg_pose, curr_robot_pose_corrected,
                    normalize = True):
    img_width, img_height = model_params["image_size"]
    context_size = model_params['context_size']
    dataset_name = deployment_params['dataset_name']
    model_type = deployment_params['nav_model_type']
    waypoint_spacing = deployment_params['waypoint_spacing']
    max_frame_dist = model_params['distance']['max_frame_dist']
    learn_angle = model_params['learn_angle']

    obs_queue_size = context_size + 1
    ts_obs_depths = torch.zeros([obs_queue_size, img_height, img_width])
    for idx in range(0, obs_queue_size):
        pil_depth = context_depth_queue[idx]
        ts_resized_depth = resize_and_aspect_crop(pil_depth, [img_width, img_height])  # [C, h, w]
        assert ts_resized_depth.max() <= 1, f"obs depths not normalized well {ts_resized_depth.max()}"
        # print("dep shape: ", ts_resized_depth[0].shape)
        # print("obs depth shape: ", ts_obs_depths[0].shape)
        ts_obs_depths[idx] = ts_resized_depth
    ts_obs_depths = ts_obs_depths[None, ...]
    ts_obs_depths = ts_obs_depths.to(device)

    # process goal depth
    # print(self.sg_depth.size)   # (64, 85)
    # print(self.sg_depth.mode)   # I;16
    ts_goal_depth = resize_and_aspect_crop(sg_depth, [img_width, img_height])  # [h, w]
    assert ts_goal_depth.max() <= 1, f"goal depth not normalized well {ts_resized_depth.max()}"
    ts_goal_depth = ts_goal_depth[None, ...]
    ts_goal_depth = ts_goal_depth.to(device)

    # process odom context:
    # (1) pose correction (if necessary), (2) tform to local coord,
    # (3) normalize and (4) convert to tensor
    # correct context odom pose
    # corrected_odom_context = odom_context_correction (context_pose_queue, correcting_htm)

    np_context_pose = np.zeros([obs_queue_size, 3])  # x, y, th
    [xc, yc, thc] = curr_robot_pose_corrected

    for ii in range(0, obs_queue_size):
        [xcp, ycp, th_cp] = context_pose_queue[ii]
        # print("%f %f %f \n" %(x0, y0, th0))
        # print("%f %f %f \n" %(xc, yc, th_c))
        rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([xc, yc, thc]), np.array([xcp, ycp, th_cp]),
                                                    dataset_name)

        if normalize:
            pose_norm = _normalize_pose(np.array([rel_x, rel_y, rel_theta]),
                                        waypoint_spacing=waypoint_spacing,
                                        dataset_name=dataset_name)
            np_context_pose[ii] = pose_norm
        else:
            np_context_pose[ii] = np.array([rel_x, rel_y, rel_theta])

    # process goal pose:
    # (2) normalize and (3) convert to tensor
    if normalize:
        [rel_x, rel_y, rel_theta] = curr_rel_sg_pose
        sg_pose = _normalize_subgoal(np.array([rel_x, rel_y, rel_theta]),
                                     max_frame_dist=max_frame_dist, dataset_name=dataset_name)
    else:
        sg_pose = np.array([rel_x, rel_y, rel_theta])

    if learn_angle:
        np_context_action = np.zeros([obs_queue_size, 4], dtype='float32')
        for ii in range(0, obs_queue_size):
            # print(np_context_pose.shape)
            q_a = rm.rpy2quat(0, 0, np_context_pose[ii, 2])
            assert round(math.sqrt(q_a[0] * q_a[0] + q_a[-1] * q_a[-1]),
                         3) == 1.0, f"Is contxt q_a: {q_a} unit quat ? "
            np_context_action[ii] = np.concatenate(
                (np_context_pose[ii, :2], np.array([q_a[0], q_a[-1]])), axis=0)

        q_sg = rm.rpy2quat(0, 0, sg_pose[2])
        assert round(math.sqrt(q_sg[0] * q_sg[0] + q_sg[-1] * q_sg[-1]),
                     3) == 1.0, f"Is SG q_a: {q_sg} unit quat ? "
        np_sg_pose = np.array([sg_pose[0], sg_pose[1], q_sg[0], q_sg[-1]], dtype='float32')

    else:
        np_context_action = np_context_pose
        np_sg_pose = sg_pose[:2]

    ts_context_action = torch.as_tensor(np_context_action, dtype=torch.float32)
    ts_context_action = ts_context_action[None, ...]
    ts_context_action = ts_context_action.to(device)

    ts_sg_pose = torch.as_tensor(np_sg_pose, dtype=torch.float32)
    ts_sg_pose = ts_sg_pose[None, ...]
    ts_sg_pose = ts_sg_pose.to(device)

    # predict distances and waypoints
    # print("contxt + obs depth shape:", ts_obs_depths.shape)         # B(1), Contxt+curr (6), h, w
    # print("goal depth shape:", ts_goal_depth.shape)                 # B(1), 1, h, w
    # print("contxt + cur actions shape:", ts_context_action.shape)   # B(1), Contxt+curr (6), num_params
    # print("goal shape", ts_goal_pose.shape)                         # B(1), num_params

    if model_type == 'image_pose_rnn':
        model_outputs = model(ts_obs_depths, ts_goal_depth, ts_context_action, ts_sg_pose)
    elif model_type == 'dev_gru':
        model_outputs = model(ts_obs_depths, ts_goal_depth, ts_context_action, ts_sg_pose)
    else:
        raise ValueError("unknown model type")

    ts_waypoints, ts_pose_diff = model_outputs  # (1,5,4),  (1,2)

    np_pose_diff = to_numpy(ts_pose_diff.squeeze())  # (2, ) remaining dist to subgoal
    np_waypoints = to_numpy(ts_waypoints.squeeze())  # (5, 4)

    if normalize:
        np_out_waypoints = _denormalize_pose(np_waypoints, waypoint_spacing=waypoint_spacing,
                                             dataset_name=dataset_name)
        np_out_pose_diff = _denormalize_subgoal(np_pose_diff, max_frame_dist=max_frame_dist,
                                                dataset_name=dataset_name)
    else:
        np_out_waypoints = np_waypoints
        np_out_pose_diff = np_pose_diff

    return np_out_waypoints, np_out_pose_diff

def load_pose_data( pose_file ):
    pose_raw = np.loadtxt(pose_file)
    num_data, _ = pose_raw.shape

    xy0 = pose_raw[0, 4:6]
    quat0 = pose_raw[0, [10, 7, 8, 9]]
    wHb = np.zeros((num_data, 4, 4), dtype=float)
    wHb[0] = rm.quat_to_htm( quat0 )
    wHb[0, 0:2, 3] = xy0
    xy = np.zeros((num_data, 2), dtype=float)
    xy[0, :] = xy0

    for idx in range(1, num_data):
        # MATLAB: xy_line = pose_raw(idx, 5:6)
        xy_line = pose_raw[idx, 4:6]
        # MATLAB: quat = pose_raw(idx, [11, 8:10])
        quat = pose_raw[idx, [10, 7, 8, 9]]
        wHb[idx, :, :] = rm.quat_to_htm(quat)
        wHb[idx, 0:2, 3] = xy_line
        xy[idx, :] = xy_line

    return pose_raw, xy, wHb


def load_slam_subgoals( curr_nav_idx: int,
                        resolution: float,
                        topomap_dir: str, sync_metadata_dir: str ):

    topo_tf_m2b_file = topomap_dir + '/topo_tf_m2b.txt'
    topo_m2b_raw, topo_m2b_xy, topo_m1Hb = load_pose_data(topo_tf_m2b_file)
    num_node = topo_m2b_xy.shape[0]

    nav_m2b_file = sync_metadata_dir + '/sync_tf_m2b.txt'
    nav_m2b_raw,  nav_m2b_xy,  nav_m2Hb = load_pose_data(nav_m2b_file)

    sgs_xyzq_corrected_m = np.zeros([num_node, 7], dtype=float)
    m2Hb_curr = nav_m2Hb[curr_nav_idx]
    bHm2_curr = np.linalg.inv(m2Hb_curr)

    #print(bHm2_curr)

    for next_sg_idx in range(0, num_node):
        bHsg = np.matmul(bHm2_curr, topo_m1Hb[next_sg_idx])
        sgs_xyzq_corrected_m[next_sg_idx, 0:2] = bHsg[0:2, 3]
        sgs_xyzq_corrected_m[next_sg_idx, 3:] = rm.htm_to_quat(bHsg)

    sgs_xyzq_corrected_px = sgs_xyzq_corrected_m.copy()
    sgs_xyzq_corrected_px[:, 0:2] = sgs_xyzq_corrected_px[:, 0:2] / resolution

    return sgs_xyzq_corrected_m, sgs_xyzq_corrected_px

def to_numpy_image(img):
    if isinstance(img, torch.Tensor):
        # detach, move to CPU, and convert to numpy
        img = img.detach().cpu()
        # if CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
        img = img.numpy()
    return img

def main(args: argparse.Namespace):

    dataset_name = test_params["dataset_name"]
    base_data_folder = '/media/data/mydata/former_datasets/colldata/colldata-all/bag_2025-11-20-15-48-59'  #bag_2025-11-20-15-48-59'
    topomap_dir = base_data_folder #'/home/hankm/python_ws/viznav/depth-nav/deployment/topomaps/T9'
    sync_metadata_dir = base_data_folder #'/media/results/navdata_collector/colldata/processed/T9_coll/coll_2025-11-21-15-20/bag_2025-11-21-15-20-52/synced'
    vizout_dir = '/media/results/devgru'

    if not os.path.exists( topomap_dir ):
        raise FileNotFoundError(f'Cannot find the {topomap_dir}. Make sure to run create_sync_topomap.py to generate a topomap\n')

    transform = ([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    #    transform = ([transforms.Normalize(mean=[1, 1, 1], std=[0, 0, 0]),])
    transform = transforms.Compose(transform)  # data preprocessing ( e.g. data augmentation )

    #################################################################################################
    # 1. load model weights
    #################################################################################################
    nav_model_type = deployment_params['nav_model_type']
    col_model_type = deployment_params['col_model_type']
    waypoint_spacing = deployment_params['waypoint_spacing']
    nav_ckpth_name = model_params['deployment']["nav_ckpth_name"]
    col_ckpth_name = model_params['deployment']["col_ckpth_name"]
    nav_ckpth_path = MODEL_WEIGHTS_PATH + "/" + nav_ckpth_name
    col_ckpth_path = MODEL_WEIGHTS_PATH + "/collision/" + col_ckpth_name

    if os.path.exists(col_ckpth_path):
        print(f"Loading model from {col_ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {col_ckpth_path}")
    col_model = load_model(
        col_ckpth_path,
        col_model_type,
        model_params,
        device,
    )
    col_model = col_model.to(device)
    col_model.eval()

    if os.path.exists(nav_ckpth_path):
        print(f"Loading model from {nav_ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {nav_ckpth_path}")
    nav_model = load_model(
        nav_ckpth_path,
        nav_model_type,
        model_params,
        device,
    )
    nav_model = nav_model.to(device)
    nav_model.eval()

    ###################################################################################################
    ## 2. Visualizer
    ###################################################################################################
    cm_params = CostmapParams(resolution=0.05,       # 5cm per pixel
                               map_size_m=10.0,        # costmap size = 4m x 4m
                               robot_radius_m=0.25,   # 20cm
                               inflation_radius_m=1.0,
                               max_range_m=25.0)
    viz = Visualizer(cm_params)

    ###################################################################################################
    ## 3. Dataset
    ###################################################################################################
    dataset = Collision_Dataset_Simple(
        base_data_folder=base_data_folder,
        dataset_name=dataset_name,
        image_size=model_params["image_size"],
        waypoint_spacing=model_params['deployment']["waypoint_spacing"],   # 1 by default b/c waypoint spacing is not set in the config file
        min_frame_dist=model_params["distance"]["min_frame_dist"],
        max_frame_dist=model_params["distance"]["max_frame_dist"],
        negative_mining=False, #model_params["negative_mining"],
        len_traj_pred=model_params["len_traj_pred"],
        learn_angle=model_params["learn_angle"],
        context_size=model_params["context_size"],
        context_type=model_params["context_type"],
        goal_dist_type=model_params["goal_dist_type"],
        normalize=model_params["normalize"],
        goal_type=model_params["goal_type"],
    )

    #################################################################################################
    # 4. begin the testing loop
    #################################################################################################
    cnt = 0
    for data in dataset:
        (
            ts_obs_images,
            ts_obs_depths,
            ts_goal_image,
            ts_goal_depth,
            ts_action_label,
            ts_context_action,
            ts_goal_pose,
            ts_pose_diff_label,
            ts_dataset_index,
            ts_nav_seq_idx,
            ts_target_sg_idx,
            ts_action_mask,
            ts_collision_label,  # true (bool) if collision
            str_data_info,
        ) = data
        ts_obs_images = ts_obs_images.unsqueeze(0) # [B, cxC, H, W ]
        ts_obs_depths = ts_obs_depths.unsqueeze(0)
        ts_goal_image = ts_goal_image.unsqueeze(0)
        ts_goal_depth = ts_goal_depth.unsqueeze(0)
        #ts_action_label = ts_action_label.unsqueeze(0)
        ts_context_action = ts_context_action.unsqueeze(0)
        ts_goal_pose = ts_goal_pose.unsqueeze(0)

        nav_seq_idx = ts_nav_seq_idx.item()
        target_sg_idx = ts_target_sg_idx.item()
    # context action, goal pose
        ts_context_action = ts_context_action.to(device)
        ts_goal_pose = ts_goal_pose.to(device)

    # TODO: check if dim=0 is correct for torch.split() and torch.cat() below
        assert (ts_obs_images.max() <= 1)
        assert (ts_goal_image.shape[1] == 3), f" rgb img shape is {ts_goal_image.shape}"
    # ts_obs_images.shape  is [B, Contxt*C, H, W],  ex) [256, 18, 64, 85]
        tuple_ts_obs_images = torch.split(ts_obs_images, 3, dim=1)  # tuple  of  obs image imgs (CxC, H, W)
        ts_curr_obs_image_vz   = TF.resize(tuple_ts_obs_images[-1], VIZ_IMG_SIZE)  # current obs img (160, 120)
        ls_ts_obs_images = [transform(ts_obs_image).to(device) for ts_obs_image in tuple_ts_obs_images]
        ts_obs_images = torch.cat(ls_ts_obs_images, dim=1)

    # obs_depth
        tuple_obs_depths = torch.split(ts_obs_depths, 1, dim=1)  # convert tensor to tuple
        ts_curr_obs_depth_vz = TF.resize(tuple_obs_depths[-1], VIZ_IMG_SIZE) # current obs
        ts_obs_depths = ts_obs_depths.to(device)
    # goal_image
        assert (ts_goal_image.max() <= 1)
        ts_goal_rgb_vz = TF.resize(ts_goal_image, VIZ_IMG_SIZE)
        ts_goal_image = transform(ts_goal_image).to(device)  # transform() does img normalization (data preprocessing)

    # goal_depth
        ts_goal_depth_vz = TF.resize(ts_goal_depth, VIZ_IMG_SIZE)
        ts_goal_depth = ts_goal_depth.to(device)

    # SGs
        data_dir = str_data_info.split(' ')[1]
        name_split = data_dir.split('/')
        data_id = name_split[-1]
        navtime_id = name_split[-2]

        old_sg_m = np.loadtxt(data_dir+'/old_subgoal_m.txt')
        gt_sg_m  = np.loadtxt(data_dir+'/new_subgoal_m.txt')

    # action label (GT waypoints)
        np_action_label_m = ts_action_label.numpy()

    #######################################################################################################
    #  5. proceed to model step
    #######################################################################################################

        ts_obs_depth_curr = ts_obs_depths[:, -1, ...].unsqueeze(0)
        col_model_outputs = col_model(ts_obs_depth_curr)
        coll_logit = col_model_outputs.item()
        pred_coll_prob = 1.0 / (1.0 + np.exp(-coll_logit))
        # nav model step FF
        nav_model_outputs = nav_model(ts_obs_depths, ts_goal_depth, ts_context_action, ts_goal_pose)
        ts_action_pred, ts_pose_diff_pred = nav_model_outputs

    #######################################################################################################
    # 6. Visualization
    #######################################################################################################
    # load costmap

        costmap_i8 = np.loadtxt(data_dir+'/costmap_i8.txt', dtype=np.int32, delimiter=",")
    # convert data to viz form
        # corrected (slam) sg poses
        if nav_seq_idx >= 0:
            sgs_xyzq_corrected_m, sgs_xyzq_corrected_px = load_slam_subgoals(
                curr_nav_idx=nav_seq_idx, resolution=0.05, topomap_dir=topomap_dir, sync_metadata_dir=sync_metadata_dir)
        else:
            sgs_xyzq_corrected_m = None
            sgs_xyzq_corrected_px = None

        pose_context_m   = ts_context_action.squeeze().cpu().numpy()
        pred_waypoints_m = ts_action_pred.squeeze().detach().cpu().numpy()
        pred_pose_diff_m = ts_pose_diff_pred.squeeze().detach().cpu().numpy()

        print("processing %d data"%cnt)
        cnt += 1
        # print("sgs shape: ", sgs_xyzq_corrected_m.shape)
        # print("pose context shape: ", pose_context_m.shape)
        # print("pred_waypoints_m shape: ", pred_waypoints_m.shape )
        # print("pred_pose_diff_m shape: ", pred_pose_diff_m.shape )

        np_curr_obs_rgb_vz = to_numpy_image(ts_curr_obs_image_vz.squeeze())
        np_curr_obs_depth_vz = to_numpy_image(ts_curr_obs_depth_vz.squeeze()) * MAX_DEPTH / 1000
        np_goal_rgb_vz = to_numpy_image(ts_goal_rgb_vz.squeeze())
        np_goal_depth_vz = to_numpy_image(ts_goal_depth_vz.squeeze()) * MAX_DEPTH / 1000

        viz.draw(
                str_data_info = str_data_info,
                rgb_img=np_curr_obs_rgb_vz,
                rgb_sg =np_goal_rgb_vz,
                depth_img=np_curr_obs_depth_vz,
                depth_sg=np_goal_depth_vz,
                costmap_i8=costmap_i8,               # raw costmap
                sgs_xyzq_corrected_m=sgs_xyzq_corrected_m,
                pose_context_m=pose_context_m,    # (x,y,qw,qz)
                waypt_label_m=np_action_label_m,    # label waypt
                rx=100,
                ry=100,
                old_sg_m=old_sg_m,
                gt_sg_m =gt_sg_m,
                pred_waypoints_m=pred_waypoints_m,
                pred_pose_diff_m=pred_pose_diff_m,
                pred_collprob=pred_coll_prob,
                save_path=f"{vizout_dir}/{data_id}.png",
                show=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    #print(f"img topic {RGB_TOPIC}")
    main(args)


