import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt

# ROS
import rospy
import message_filters
from message_filters import TimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension

#from data.ipython_collision_dataset import max_frame_dist
from nav_utils import msg_to_pil, to_numpy, transform_images, load_model, msg_to_caminfo, set_tform
import tf2_ros

import torch
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
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.append(BASE_DIR)
import utils.rigid_motion as rm
from typing import Optional
from navigator import navigator

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

# UTILS
from topic_names import (RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO,
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

def _to_uint8_image(img):
    """Convert PIL / NumPy / Torch image to uint8 grayscale or RGB NumPy array."""
    # ✅ PIL.Image
    if isinstance(img, PILImage.Image):
        arr = np.asarray(img)
        if arr.ndim == 2:  # grayscale depth
            # robust normalize even for uint16/float
            arr = arr.astype(np.float32)
            finite = np.isfinite(arr)
            if finite.any():
                mn, mx = arr[finite].min(), arr[finite].max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = np.zeros_like(arr)
            else:
                arr = np.zeros_like(arr)
            return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return np.asarray(img.convert("RGB"))

    # ✅ torch.Tensor
    if torch.is_tensor(img):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):   # (C,H,W)
            x = x.permute(1, 2, 0)
        x = x.numpy()
        if x.dtype != np.uint8:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # if looks like depth, scale per-frame
            if x.ndim == 2 or x.shape[2] == 1:
                vmin, vmax = np.min(x), np.max(x)
                x = (x - vmin) / (vmax - vmin + 1e-8)
            x = (np.clip(x, 0, 1) * 255.0).astype(np.uint8)
        return x

    # ✅ numpy arrays
    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale depth
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = np.min(arr), np.max(arr)
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr)
        return (arr * 255).astype(np.uint8)

    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    return arr

def show_status_rgb_depth(
    depth_img: np.ndarray,
    rgb_img: np.ndarray,
    is_collision: bool,
    prob: Optional[float] = None,
    logit: Optional[float] = None,
    window_name: str = "Navigator RGB + Depth Status",
    use_colormap: bool = False,
    save_path: Optional[str] = None,   # <-- NEW
):
    """
    Show RGB and Depth images side by side with a colored status banner on top.
    Left  = RGB image
    Right = Depth image (gray or colormapped)

    Red banner  = COLLISION
    Green banner = SAFE
    """

    # --- 1. Prepare RGB image (ensure 3-channel uint8) ---
    print(rgb_img.shape )
    print(depth_img.shape )
    rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)

    # --- 2. Prepare Depth image (uint8 -> 3-channel) ---
    depth_u8 = _to_uint8_image(depth_img)  # your existing helper

    if depth_u8.ndim == 2:
        if use_colormap:
            depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
        else:
            depth_vis = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
    else:
        depth_vis = depth_u8

    # --- 3. Resize depth to match RGB size (for clean hstack) ---
    H_rgb, W_rgb = rgb.shape[:2]
    depth_vis = cv2.resize(depth_vis, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST)

    # --- 4. Concatenate side by side: [ RGB | DEPTH ] ---
    img = np.hstack([rgb, depth_vis])

    # --- 5. Draw status banner on top of the combined image ---
    H, W = img.shape[:2]
    banner_h = max(40, H // 16)

    COLOR_COLLISION = (40, 40, 220)   # BGR (red)
    COLOR_SAFE      = (40, 200, 40)   # BGR (green)
    color = COLOR_COLLISION if is_collision else COLOR_SAFE

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, banner_h), color, -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

    # --- 6. Build status text ---
    status = "COLLISION" if is_collision else "SAFE"
    text_items = [status]
    if prob is not None:
        text_items.append(f"p={prob:.2f}")
    if logit is not None:
        text_items.append(f"logit={logit:.2f}")
    text = " | ".join(text_items)

    # Put text near the left side of the banner
    cv2.putText(
        img,
        text,
        (12, banner_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --- 7. Show window ---
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

    if save_path is not None:
        cv2.imwrite(save_path, img)
        # (optional) print a confirmation
        # print(f"[Saved] {save_path}")

    return img


def main(args: argparse.Namespace):

    dataset_name = 'former'  # model_params["datasets"]
    subgoal_spacing = deployment_params['subgoal_spacing'] #int(model_params['distance']['max_frame_dist'] / 2)
    topomap_dir = DEPLOYMENT_DIR + "/topomaps/" + deployment_params['topomap_name']

    if not os.path.exists( topomap_dir ):
        raise FileNotFoundError(f'Cannot find the {topomap_dir}. Make sure to run create_sync_topomap.py to generate a topomap\n')

    #################################################################################################
    # 0. load model weights
    #################################################################################################
    model_type = deployment_params['col_model_type']
    waypoint_spacing = deployment_params['waypoint_spacing']
    ckpth_name = model_params['deployment']["col_ckpth_name"]
    ckpth_path = MODEL_WEIGHTS_PATH + "/collision/" + ckpth_name

    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_type,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    #################################################################################################
    # 1. load topomap imgs
    #################################################################################################
    synced_data_all = os.listdir(topomap_dir)
    # image_type = model_params['goal_type']
    topomap_rgb_files = [f for f in synced_data_all if 'rgb' in f]
    topomap_depth_files = [f for f in synced_data_all if 'depth' in f]

    topomap_rgb_filenames = sorted(topomap_rgb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    topomap_depth_filenames = sorted(topomap_depth_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    num_nodes = len(topomap_depth_filenames)
    #num_nodes = tot_nodes #len( list(range(0, tot_nodes)) ) #int(tot_nodes / subgoal_spacing)

    topomap_depth = []
    topomap_rgb = []
    cnt = 0
    for i in range(0, num_nodes):
        rgb_path = os.path.join(topomap_dir, topomap_rgb_filenames[i])
        depth_path = os.path.join(topomap_dir, topomap_depth_filenames[i])
        cv_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        pil_depth = PILImage.fromarray(cv_depth, mode="I;16")
        topomap_depth.append(pil_depth)
        topomap_rgb.append(PILImage.open(rgb_path))
        #cv2.imwrite("/home/glab/results/dep_gru/sg_rgb/%05d.png"%cnt, cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED))
        #cnt += 1

    assert -1 <= args.goal_node < len(topomap_depth), "Invalid goal index"
    if args.goal_node == -1:
        goal_node_idx = len(topomap_depth) - 1
    else:
        goal_node_idx = args.goal_node

    is_goal_reached = False

    #################################################################################################
    # 2. load topomap odom and transform them to local coord frame
    #################################################################################################
    topomap_odom = os.path.join(topomap_dir + '/topo_odom.txt')
    odom_info = np.loadtxt(topomap_odom)

    print(odom_info.shape)
    topomap_odom = []
    w1Hb = np.tile(np.eye(4), (num_nodes, 1, 1))
    cnt = 0
    for idx in range(0, num_nodes):
        x = odom_info[idx, 4]  # x
        y = odom_info[idx, 5]  # y
        q = odom_info[idx, 7:11]  # qx, qy, qz, qw
        quat = [q[-1], q[0], q[1], q[2]]
        w1Hb[idx] = rm.quat_to_htm(quat)
        # _,_,_,_,_,yaw = rm.htm_to_xyzrpy(w1Hb[cnt])
        # yaw_n = rm.normalizeAngle(yaw)
        w1Hb[idx,0,3] = x
        w1Hb[idx,1,3] = y
        #print(w1Hb[idx])
#        print("sg %d: x:%f  y:%f  \n" % (idx, x, y ))

    b0Hw1 = np.linalg.inv(w1Hb[0].copy())
    print(b0Hw1)
    for i in range(0, num_nodes):
        b0Hb_i = np.matmul( b0Hw1, w1Hb[i] )
        #print(b0Hb_i)
        [xt, yt, _, _, _, yaw_t] = rm.htm_to_xyzrpy(b0Hb_i)
        #yaw_n = rm.normalizeAngle(yaw)
        topomap_odom.append([xt, yt, yaw_t]) # [x0, y0, th0] == [0, 0, 0]
        #print("tformed sg %d: x:%f  y:%f  th:%f \n"%(i, xt, yt, yaw_t) )

    # begin ROS
    rospy.init_node("navigator_parent_node", anonymous=False)
    rate = rospy.Rate(RATE)

    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    # navigator instance
    nav = navigator(name='viz_navigator', dataset_name='former',
                             topomap_rgb = topomap_rgb,
                             topomap_depth = topomap_depth,
                             topomap_odom = topomap_odom,
                             config = model_params, data_config=data_config, robot_config=robot_config)
    # navigation loop
    #################################################################################################
    # Make sure to have the robot's init pose is aligned with the starting position of the sug-goal trajectory
    #################################################################################################

    nav.update_subgoal() # set subgoal == 1

    while not rospy.is_shutdown():
        # Image Goal Conditioned Navigation
        # chosen_waypoint = np.zeros(4)
        reached_goal = nav._is_finalgoal_reached
        if reached_goal:
            print("\n Reached the final goal! <%d> Stopping... \n"%nav.curr_sg_idx)
            goal_pub.publish(reached_goal)
            rospy.signal_shutdown("Shutting down navigator\n")
            break
        #print("Q size %d "%len(nav.context_depth_queue) )

        #print("curr target SG idx is: <%d> \n" % (nav.curr_sg_idx))
        if len(nav.context_depth_queue) > model_params["context_size"]:

            #if model_params["model_type"] == "image_pose_rnn":  # (len(context_queue) > model_params["context_size"]):
            nav._is_sg_reached = False
            xy_dist = 2.
            orient_dist = 0.
            print("Moving toward the new subgoal \n")
            while not nav._is_sg_reached:

                # for i, sg_depth in enumerate(topomap[start: end + 1]):
                # transf_obs_img = transform_images(context_queue, model_params["image_size"])
                # goal_data = transform_images(sg_img, model_params["image_size"])
                if nav.got_new_metadata(): # new rgb-d data is received
                    # estimate denormalized waypoints and dx, dy to the subgoal
                    coll_logit = nav.predict_collision(model, model_type, dataset_name)
                    coll_prob = 1.0 / (1.0 + np.exp(-coll_logit))
                    if coll_prob > 0.5: # meaning the robot is about to collide..
                        nav.set_collision_status(True)
                        print('\033[91mcoll prob: %.2f  %.2f\033[0m' % (coll_prob, coll_logit))
                    else:
                        # ordinary case. just follow the curr sg
                        nav.set_collision_status(False)
                        print('coll prob: %.2f  %.2f' % (coll_prob, coll_logit))

                    if len(nav.context_depth_queue) > 0:
                        np_curr_depth = np.array( nav.context_depth_queue[-1], dtype=np.float32 )  # latest RGB image
                        np_curr_rgb = np.array(nav.viz_curr_rgb, dtype=np.uint8)
                        save_file = f"/home/glab/results/dev_gru/{cnt:05d}.png"
                        show_status_rgb_depth(np_curr_depth,
                                              np_curr_rgb, nav._is_collision,
                                              coll_prob, coll_logit,
                                              save_path=save_file)
                        cnt += 1
                # chose subgoal and output waypoints
                # print(f"is_subgoal_reached: {nav._is_subgoal_reached}")
                if ( nav.is_subgoal_reached() == True):
                    print("Updating SG \n")
                    nav.update_subgoal()
                    break

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


