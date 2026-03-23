import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
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


def main(args: argparse.Namespace):

    dataset_name = 'former'  # model_params["datasets"]
    subgoal_spacing = deployment_params['subgoal_spacing'] #int(model_params['distance']['max_frame_dist'] / 2)
    topomap_dir = DEPLOYMENT_DIR + "/topomaps/" + deployment_params['topomap_name']

    if not os.path.exists( topomap_dir ):
        raise FileNotFoundError(f'Cannot find the {topomap_dir}. Make sure to run create_sync_topomap.py to generate a topomap\n')

    #################################################################################################
    # 0. load model weights
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

    #################################################################################################
    # 1. load data
    #################################################################################################
    data_dir = BASE_DIR + '/test/data'
    data_all = os.listdir(data_dir)
    # image_type = model_params['goal_type']
    context_rgb_files = [f for f in data_all if re.match(r"rgb\d{5}", f) ]
    context_depth_files = [f for f in data_all if re.match(r"depth\d{5}", f) ]
    rgb_sg_file_path = data_dir +'/rgb_sg.png'
    depth_sg_file_path = data_dir+'/depth_sg.png'
    pil_depth_sg = PILImage.open(depth_sg_file_path)
    pil_rgb_sg = PILImage.open(rgb_sg_file_path)

    context_depth_queue = []
    context_rgb_queue = []
    context_pose_queue = []
    for depth_file in context_depth_files:
        depth_file_path = data_dir + '/' + depth_file
        pil_depth = PILImage.open(depth_file_path)
        context_depth_queue.append(pil_depth)

    for rgb_file in context_rgb_files:
        rgb_file_path = data_dir + '/' + rgb_file
        pil_rgb = PILImage.open(rgb_file_path)
        context_rgb_queue.append(pil_rgb)

    context_pose_xyzquat = np.loadtxt(data_dir+'/pose_context_m.txt')
    context_pose_queue = []
    for idx in range(0, len(context_pose_xyzquat)):
        pose_xyzquat = context_pose_xyzquat[0]
        x, y = pose_xyzquat[0:2]
        qw, _, _, qz = pose_xyzquat[3:]
        theta = 2 * np.arctan2(qz, qw)
        context_pose_queue.append([x, y, theta])

    curr_robot_pose_corrected = context_pose_queue[-1]
    sg_pose = np.loadtxt(data_dir+'/old_subgoal_m.txt')
    th = 2 *  np.arctan2(sg_pose[3], sg_pose[2])
    curr_rel_sg_pose = [sg_pose[0], sg_pose[1], th]

    coll_logit = predict_collision(col_model, context_depth_queue)
    np_waypoints_m, np_pose_diff_m = predict_actions(nav_model, context_depth_queue, context_pose_queue,
                    pil_depth_sg, curr_rel_sg_pose, curr_robot_pose_corrected, True)

    coll_prob = 1.0 / (1.0 + np.exp(-coll_logit))
    print("coll prob: %.2f"% coll_prob)
    print(np_waypoints_m)
    print(np_pose_diff_m)
    
    # save output
    f = open(data_dir+'/pred_coll.txt', 'w')
    f.write('%.02f'%coll_prob)
    f.close()
    np.savetxt(data_dir + '/pred_waypoints.txt', np_waypoints_m)
    np.savetxt(data_dir + '/pred_pose_diff.txt', np_pose_diff_m)


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


