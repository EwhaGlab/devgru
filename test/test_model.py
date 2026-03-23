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


def to_numpy_image(img):
    if isinstance(img, torch.Tensor):
        # detach, move to CPU, and convert to numpy
        img = img.detach().cpu()
        # if CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
        img = img.numpy()
    return img


def _quat_ang_dist(q1: torch.Tensor, q2: torch.Tensor):
    """
    Unsigned angular distance between quaternions (radians).
    Inputs: (...,4) = [qw,qx,qy,qz] or (...,2) = [qw,qz]
    Returns:
      sin2_half: sin^2(theta/2) in [0,1], shape q1.shape[:-1]
      angle    : theta in [0, pi],        shape q1.shape[:-1]
    """
    assert q1.shape == q2.shape and q1.shape[-1] in (2, 4), f"{q1.shape} vs {q2.shape}"

    # Normalize along last dim (robust even if inputs are near-unit)
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-12)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-12)

    # Dot product along last dim; use |dot| so q ≡ -q
    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)  # |cos(theta/2)|

    sin2_half = 1.0 - dot * dot                          # in [0,1]
    angle = 2.0 * torch.acos(dot)                        # in [0, pi]

    return sin2_half, angle

def _compute_errors(
    pose_diff_label: torch.Tensor,  # (B, 4)
    action_label: torch.Tensor,
    pose_diff_pred: torch.Tensor,   # (B, 4)
    action_pred: torch.Tensor,
    action_mask: torch.Tensor = None, # 1 if the pred action (wpt) is bounded btwn 0 ~ 10
    test_angle: bool = False,
):
    """
    Compute losses for distance and action prediction.
    """
    # position loss
    dx_l = F.smooth_l1_loss(pose_diff_pred[0], pose_diff_label[0], reduction='mean')
    dy_l = F.smooth_l1_loss(pose_diff_pred[1], pose_diff_label[1], reduction='mean')
    sg_pos_diff_loss = dx_l + dy_l
    # ang loss
    if test_angle:
        qw_p, qz_p = pose_diff_pred[2:]
        qw_g, qz_g = pose_diff_label[2:]
        p_q = torch.stack([qw_p, torch.zeros_like(qw_p), torch.zeros_like(qw_p), qz_p])
        g_q = torch.stack([qw_g, torch.zeros_like(qw_g), torch.zeros_like(qw_g), qz_g])
        dot = (p_q * g_q).sum(dim=-1).abs().clamp(0, 1)  # |cos(dθ/2)|
        yaw_loss = 1.0 - dot ** 2  # smooth ≈ sin^2(dθ/2)
        sg_pose_diff_loss = sg_pos_diff_loss.mean()
        sg_ang_diff_loss = yaw_loss.mean()
    else:
        sg_pose_diff_loss = sg_pos_diff_loss.mean()

    len_traj_pred = model_params['len_traj_pred']
    use_time_weight = model_params['use_time_weight']

    if use_time_weight == True:
        pred_weights = torch.linspace(1.0, len_traj_pred, len_traj_pred, device=action_pred.device)
        pred_weights = (pred_weights / pred_weights.mean()).view(len_traj_pred, 1)
    else:
        pred_weights = torch.ones([1, len_traj_pred], device=action_pred.device)
        pred_weights = pred_weights.view(len_traj_pred, 1)

    def action_reduce(unreduced_loss: torch.Tensor):
        # takes [B, X, Y, theta], then repeats mean() of the last dim until the shape becomes [B]
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"  # action_label.shape
    action_mse = F.mse_loss(action_pred[..., :2], action_label[..., :2], reduction="none")  # [B, len_traj_pred, 2], 2 b/c qz and qw only

    weighted_action_loss = action_mse * pred_weights
    action_loss = action_reduce( weighted_action_loss ) # action waypt loss

    #device = action_pred.device
    #TODO: Find a better way to compute angle diff btwn two quaternions
    if test_angle:
        qw_pred = action_pred[..., 2].clone()
        qz_pred = action_pred[..., 3].clone()
        quat_pred = torch.zeros([len_traj_pred, 4]).to(device)

        quat_pred[..., 0] = qw_pred
        quat_pred[..., 3] = qz_pred
        quat_pred = torch.nn.functional.normalize(quat_pred, dim=1)

        qw_label = action_label[..., 2].clone()  # [x y qw qz]
        qz_label = action_label[..., 3].clone()
        quat_label = torch.zeros([len_traj_pred, 4]).to(device)
        quat_label[..., 0] = qw_label
        quat_label[..., 3] = qz_label

        q_dist, q_ang = _quat_ang_dist(quat_pred, quat_label)  # q_dist (B, len_pred),
        #orient_mse = F.mse_loss(quat_label, quat_pred, reduction="none")
        p = F.normalize(action_pred[..., 2:4], dim=-1, eps=1e-8)
        g = F.normalize(action_label[..., 2:4], dim=-1, eps=1e-8)
        dot = (p * g).sum(dim=-1).abs().clamp(0, 1)  # |cos(dθ/2)|
        wp_yaw_loss_per = 1.0 - dot ** 2
        weighted_orient_loss = wp_yaw_loss_per * pred_weights.squeeze(-1)
        wp_orient_loss = action_reduce( weighted_orient_loss )

        wp, _, _, zp = p_q.unbind(dim=-1) # each (B,)
        wg, _, _, zg = g_q.unbind(dim=-1) # each (B,)
        w_rel = wp * wg + zp * zg
        z_rel = zp * wg - wp * zg
        theta_err = 2.0 * torch.atan2(z_rel, w_rel)
        # wrap to (-pi, pi]
        theta_err = (theta_err + torch.pi) % (2 * torch.pi) - torch.pi
        total_err = [sg_pos_diff_loss, action_loss, wp_orient_loss]
    else:
        total_err = [sg_pos_diff_loss, action_loss]

    return total_err


def main(args: argparse.Namespace):

    dataset_name = deployment_params["dataset_name"]
    base_data_folder = test_params['test_dataset_path']
    vizout_dir = test_params['out_res_path']

    transform = ([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    transform = transforms.Compose(transform)  # data preprocessing ( e.g. data augmentation )

    #################################################################################################
    # 1. load model weights
    #################################################################################################
    nav_model_type = deployment_params['nav_model_type']
    col_model_type = deployment_params['col_model_type']
    waypoint_spacing = deployment_params['waypoint_spacing']
    nav_ckpth_name = deployment_params["nav_ckpth_name"]
    col_ckpth_name = deployment_params["col_ckpth_name"]
    nav_ckpth_path = MODEL_WEIGHTS_PATH + "/noncol_X-col_Y/" + nav_ckpth_name
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
        waypoint_spacing=deployment_params["waypoint_spacing"],   # 1 by default b/c waypoint spacing is not set in the config file
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

    num_data = len(dataset)
    np_tot_error = np.zeros((num_data, 3), dtype=float)
    np_coll_prob = np.zeros((num_data, 1), dtype=float)
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
        ) = data # normalized vals
        ts_obs_images = ts_obs_images.unsqueeze(0) # [B, cxC, H, W ]
        ts_obs_depths = ts_obs_depths.unsqueeze(0)
        ts_goal_image = ts_goal_image.unsqueeze(0)
        ts_goal_depth = ts_goal_depth.unsqueeze(0)
        #ts_action_label = ts_action_label.unsqueeze(0)
        #ts_pose_diff_label = ts_pose_diff_label.unsqueeze(0)
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

    # label action, sgs
        ts_action_label = ts_action_label.to(device)
        ts_pose_diff_label = ts_pose_diff_label.to(device)

        ts_action_mask = ts_action_mask.repeat(len_traj_pred)
        ts_action_mask = ts_action_mask.to(device)
    # SGs
        data_dir = str_data_info.split(' ')[1]
        name_split = data_dir.split('/')
        data_id = name_split[-1]
        navtime_id = name_split[-2]

        old_sg_m = np.loadtxt(data_dir+'/old_subgoal_m.txt')
        gt_sg_m  = np.loadtxt(data_dir+'/new_subgoal_m.txt')

    # action label (GT waypoints)
        np_action_label_m = ts_action_label.cpu().numpy()

    #######################################################################################################
    #  5-1. proceed to nav model step
    #######################################################################################################
        ts_obs_depth_curr = ts_obs_depths[:, -1, ...].unsqueeze(0)
        if col_model_type == 'depth_coll':  # curr depth only collision model
            col_model_outputs = col_model(ts_obs_depth_curr)
        elif col_model_type == 'depth_sg_coll':
            col_model_outputs = col_model(ts_obs_depth_curr, ts_goal_depth)
        else:
            raise NotImplementedError

        coll_logit = col_model_outputs.item()
        pred_coll_prob = 1.0 / (1.0 + np.exp(-coll_logit))
        # nav model step FF
        nav_model_outputs = nav_model(ts_obs_depths, ts_goal_depth, ts_context_action, ts_goal_pose)
        ts_action_pred, ts_pose_diff_pred = nav_model_outputs

    #######################################################################################################
    #  6. compute loss
    #######################################################################################################
        ts_action_pred = ts_action_pred.squeeze()
        ts_pose_diff_pred = ts_pose_diff_pred.squeeze()
        pred_error = _compute_errors(
            pose_diff_label = ts_pose_diff_label,
            action_label   = ts_action_label,
            pose_diff_pred = ts_pose_diff_pred,  # (B, 4)
            action_pred = ts_action_pred,
            action_mask = ts_action_mask,  # 1 if the pred action (wpt) is bounded btwn 0 ~ 10
            test_angle  = True)

        np_tot_error[cnt, 0] = pred_error[0].item()
        np_tot_error[cnt, 1] = pred_error[1].item()

        if len(pred_error) == 3:
            np_tot_error[cnt, 2] = pred_error[2].item()

        np_coll_prob[cnt] = pred_coll_prob
    #######################################################################################################
    # 7. Visualization
    #######################################################################################################
    # load costmap

        costmap_file = data_dir+'/costmap_i8.txt'
        if os.path.exists(costmap_file):
            costmap_i8 = np.loadtxt(costmap_file, dtype=np.int32, delimiter=",")
        else:
            costmap_i8 = None

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

    # stats
    mean_error = np.mean(np_tot_error, axis=0)
    med_error = np.median(np_tot_error, axis=0)
    data = np.stack([mean_error, med_error], axis=1)

    outfile = f"{vizout_dir}/data_error.txt"
    np.savetxt(outfile, np_tot_error, fmt='%.5f',  delimiter=' ')

    outfile = f"{vizout_dir}/coll_prob.txt"
    np.savetxt(outfile, np_coll_prob, fmt='%.5f', delimiter=' ')

    outfile = f"{vizout_dir}/tot_error.txt"
    all_row_labels = ['sg pos diff', 'action (pos) error', 'action (ang) error']
    row_labels = all_row_labels[:data.shape[0]]
    col_labels = ['mean', 'median']
    with open(outfile, "w") as f:
        # header
        f.write(f"{'':25}" + " ".join(f"{c:>12}" for c in col_labels) + "\n")

        # rows
        for rlabel, row in zip(row_labels, data):
            f.write(f"{rlabel:<25}" + " ".join(f"{v:12.4f}" for v in row) + "\n")

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


