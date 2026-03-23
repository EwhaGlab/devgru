import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable


import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml
import cv2 

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.append(BASE_DIR)

import utils.rigid_motion as rm

from data.data_utils import(
    _normalize_pose,
    _denormalize_pose,
    _get_rel_pose_se2,
    resize_and_aspect_crop
)

# ROS
import rospy
import message_filters
from message_filters import TimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray
from nav_utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
#from train.depth_nav_train.train_utils import get_action

import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
import math
from cv_bridge import CvBridge
bridge = CvBridge()


# UTILS
from topic_names import (DEPTH_TOPIC, RGB_TOPIC, #IMAGE_TOPIC,
                        ODOM_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)

from os.path import dirname, abspath
# CONSTANTS
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
DEPLOYMENT_DIR = BASE_DIR + "/deployment"
TOPOMAP_IMAGES_DIR = "%s/topomaps" % DEPLOYMENT_DIR
MODEL_WEIGHTS_PATH = "%s/model_weights"%DEPLOYMENT_DIR
ROBOT_CONFIG_PATH ="%s/config/robot.yaml"%DEPLOYMENT_DIR
MODEL_CONFIG_PATH = "%s/config/models.yaml"%DEPLOYMENT_DIR
MODEL_PARAM_PATH = '%s/config/depth_nav.yaml'%BASE_DIR

print(TOPOMAP_IMAGES_DIR)
print(MODEL_WEIGHTS_PATH)
print(ROBOT_CONFIG_PATH)

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# load model parameters
# with open(MODEL_CONFIG_PATH, "r") as f:
#     model_paths = yaml.safe_load(f)
#
# model_config_path = model_paths[args.model]["config_path"]
# with open(model_config_path, "r") as f:
#     model_params = yaml.safe_load(f)

with open(MODEL_PARAM_PATH, "r") as f:
    model_params = yaml.safe_load(f)

# GLOBALS
context_rgb_queue = []
context_depth_queue = []
context_odom_queue = []
context_size = None  
obs_queue_size = None
subgoal = []
prev_sg_idx = None
curr_sg_idx = None
correcting_htm = np.eye(4)
curr_topomap_idx = None
learn_angle = True


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image

def callback_curr_topomap(msg):
    global curr_topomap_idx
    curr_topomap_idx = msg.data

# The training data ( /rgbd_throttle/rgbd ) was collected @ 10 hz
# see navdata_collector/launch/include/start_bag_async.laucn.. wanted 12 hz but record @ 10 ~ 11 hz
waypoint_spacing = model_params['datasets']['former']['waypoint_spacing']
callback_fn_freq = 10. / float(waypoint_spacing)
callback_fn_interval = 1. / callback_fn_freq
last_callback_time = 0

print("obs freq and obs time interval: %f, %f" % (callback_fn_freq, callback_fn_interval))


def main(args: argparse.Namespace):
    global context_size

    context_size = model_params["context_size"]
    obs_queue_size = context_size + 1
    
    subgoal_spacing = int( model_params['distance']['max_frame_dist'] / 2 )
    topomap_dir = model_params['topomap_dir']
    dataset_name = 'former' #model_params["datasets"]
    normalize = model_params['normalize']

    #################################################################################################
    # 0. load model weights
    #################################################################################################
    ckpth_path = DEPLOYMENT_DIR+"/model_weights/dep_gru_ws3.pth" #model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    #################################################################################################
    # 1. load topomap imgs
    #################################################################################################
    synced_images_all = os.listdir(topomap_dir)
    #image_type = model_params['goal_type']
    topomap_rgb_files = [f for f in synced_images_all if 'rgb' in f ]
    topomap_depth_files = [f for f in synced_images_all if 'depth' in f ]

    topomap_rgb_filenames = sorted(topomap_rgb_files, key=lambda x: int(''.join(filter(str.isdigit, x))) )
    topomap_depth_filenames = sorted(topomap_depth_files, key=lambda x: int(''.join(filter(str.isdigit, x))) )

    tot_nodes = len( topomap_depth_filenames)
    num_nodes = int( tot_nodes / subgoal_spacing )

    topomap_depth = []
    topomap_rgb  = []
    for i in range(0, tot_nodes, subgoal_spacing):
        rgb_path = os.path.join(topomap_dir, topomap_rgb_filenames[i])
        depth_path = os.path.join(topomap_dir, topomap_depth_filenames[i])
        cv_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        pil_depth = PILImage.fromarray(cv_depth, mode="I;16")
        topomap_depth.append( pil_depth )
        topomap_rgb.append(PILImage.open(rgb_path))

    closest_node_idx = 0
    assert -1 <= args.goal_node < len(topomap_depth), "Invalid goal index"
    if args.goal_node == -1:
        goal_node_idx = len(topomap_depth) - 1
    else:
        goal_node_idx = args.goal_node
    is_goal_reached = False

    #################################################################################################
    # 2. load topomap odom and transform them to local coord frame
    #################################################################################################
    synced_odom = os.path.join(topomap_dir+'/sync_odom.txt')
    odom_info = np.loadtxt(synced_odom)
    topo_odom = []

    for i in range(0, tot_nodes, subgoal_spacing):
        x = odom_info[i, 4]    # x
        y = odom_info[i, 5]    # y
        quat = odom_info[i, 7:11] # qw, qx, qy, qz
        _, _, _, rol, pit, yaw = rm.htm_to_xyzrpy( rm.quat_to_htm(quat) )
        topo_odom.append([x, y, yaw])

    # transform topomap coords wrt the first coord
    # x0, y0, th0 = topo_odom[0]
    # for i in range(1, num_nodes):
    #     x, y, th = topo_odom[i]
    #     rel_x, rel_y, rel_theta_p = _get_rel_pose_se2(np.array([x0, y0, th0]), np.array([x, y, th]), dataset_name)
    #     topo_odom[i] = [rel_x, rel_y, rel_theta_p]
    # topo_odom[0] = [0, 0, 0]


     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    rgb_curr_msg = rospy.Subscriber(RGB_TOPIC, Image, callback_curr_obs, queue_size=1)
    #rgb_sub   = message_filters.Subscriber(RGB_TOPIC, Image)
    depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)
    odom_sub = message_filters.Subscriber(ODOM_TOPIC, Odometry)
    sync = message_filters.ApproximateTimeSynchronizer([depth_sub, odom_sub], queue_size=10, slop=0.1) # 100ms tolerance
    sync.registerCallback(synced_data_callback)

    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    img_width, img_height = model_params["image_size"]
    curr_sg_idx = 1  # next sg to visit
    prev_sg_idx = 0  # last sg is the origin of the curr robot pose (odom)

    # navigation loop
    #################################################################################################
    # Make sure to have the robot's init pose is aligned with the starting position of the sug-goal trajectory
    #################################################################################################
    while not rospy.is_shutdown():
        # Image Goal Conditioned Navigation
        chosen_waypoint = np.zeros(4)

        reached_goal = prev_sg_idx == goal_node_idx
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
            break
        #print("Q size %d "%len(context_depth_queue) )
        if len(context_depth_queue) > model_params["context_size"]:
            
            if model_params["model_type"] == "image_pose_rnn":  #(len(context_queue) > model_params["context_size"]):
                # start   = max(closest_node_idx - args.radius, 0)
                # end     = min(closest_node_idx + args.radius + 1, goal_node_idx)
                # distance  = []
                # waypoints = []
                # update correcting transformation !!!
                # correcting_htm = update_correcting_htm( )

                sg_depth = topomap_depth[curr_sg_idx]
                sg_rgb   = topomap_rgb[curr_sg_idx]
                x_sg, y_sg, th_sg  = topo_odom[curr_sg_idx]
                x0, y0, th0 = topo_odom[prev_sg_idx]
                is_sg_reached = False

                print("I am starting from <%d>th sg. Now, my curr sg idx is: <%d>, and the final goal node idx <%d>\n"%(prev_sg_idx, curr_sg_idx, goal_node_idx) )
                while is_sg_reached == False:

#                   # for i, sg_depth in enumerate(topomap[start: end + 1]):
                    # transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    # goal_data = transform_images(sg_img, model_params["image_size"])

                    # process obs depth contexts
                    ts_obs_depths = torch.zeros( [obs_queue_size, img_height, img_width] )
                    for idx in range(0, obs_queue_size ):
                        ros_depth = context_depth_queue[idx]
                        cv_depth = bridge.imgmsg_to_cv2(ros_depth, desired_encoding="passthrough") 
                        pil_depth = PILImage.fromarray(cv_depth).convert("I;16")
                        ts_resized_depth = resize_and_aspect_crop(pil_depth, [img_width, img_height])  # [C, h, w]
                        #print("dep shape: ", ts_resized_depth[0].shape)
                        #print("obs depth shape: ", ts_obs_depths[0].shape)
                        ts_obs_depths[idx] = ts_resized_depth
                    ts_obs_depths = ts_obs_depths[None, ...]
                    ts_obs_depths = ts_obs_depths.to(device)
                    
                    # process goal depth
                    #print(sg_depth.size)   (64, 85)
                    #print(sg_depth.mode)   I;16
                    ts_goal_depth = resize_and_aspect_crop( sg_depth, [img_width, img_height] ) #[h, w]
                    ts_goal_depth = ts_goal_depth[None, ...]
                    ts_goal_depth = ts_goal_depth.to(device)
                    
                    # process odom context:
                    # (1) pose correction (if necessary), (2) tform to local coord,
                    # (3) normalize and (4) convert to tensor
                    # correct context odom pose
                    # corrected_odom_context = odom_context_correction (context_odom_queue, correcting_htm)

                    np_context_pose = np.zeros( [obs_queue_size, 3] )  # x, y, th
                    for ii in range(0, obs_queue_size):
                        odom_pos = context_odom_queue[ii].pose.pose.position
                        odom_orient = context_odom_queue[ii].pose.pose.orientation
                        xc = odom_pos.x
                        yc = odom_pos.y
                        qx = odom_orient.x 
                        qy = odom_orient.y
                        qz = odom_orient.z 
                        qw = odom_orient.w 
                        [_, _, _, _, _, th_c] = rm.htm_to_xyzrpy( rm.quat_to_htm([qw, qx, qy, qz]) )
                        
                        #print("%f %f %f \n" %(x0, y0, th0))
                        #print("%f %f %f \n" %(xc, yc, th_c))
                        rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([x0, y0, th0]), np.array([xc, yc, th_c]),
                                                                    dataset_name)
                        
                        if normalize:
                            pose_norm = _normalize_pose( np.array( [rel_x, rel_y, rel_theta] ),
                                                  waypoint_spacing=waypoint_spacing, eta=1.0, dataset_name=dataset_name)
                            np_context_pose[ii] = pose_norm
                        else:
                            np_context_pose[ii] = np.array( [rel_x, rel_y, rel_theta] )

                        # process goal pose:
                        # (1) tform to local coord and (2) normalize and (3) convert to tensor
                        rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([x0, y0, th0]), np.array([x_sg, y_sg, th_sg]),
                                                                    dataset_name)
                        if normalize:
                            sg_pose = _normalize_pose(np.array([rel_x, rel_y, rel_theta]),
                                                                waypoint_spacing=waypoint_spacing, eta=1.0, dataset_name=dataset_name)
                        else:
                            sg_pose = np.array([rel_x, rel_y, rel_theta])

                        if learn_angle:
                            np_context_action = np.zeros([obs_queue_size, 4], dtype='float32')
                            for ii in range(0, obs_queue_size):
                                #print(np_context_pose.shape) 
                                q_a = rm.rpy2quat(0, 0, np_context_pose[ii, 2])
                                assert round(math.sqrt(q_a[0] * q_a[0] + q_a[-1] * q_a[-1]), 3) == 1.0, f"Is contxt q_a: {q_a} unit quat ? "
                                np_context_action[ii] = np.concatenate((np_context_pose[ii, :2], np.array([q_a[0], q_a[-1]])), axis=0)
                                
                            sg_pose_4d = np.zeros([4], dtype = 'float32')
                            q_a = rm.rpy2quat(0,0, sg_pose[2])
                            assert round(math.sqrt(q_a[0] * q_a[0] + q_a[-1] * q_a[-1]), 3) == 1.0, f"Is SG q_a: {q_a} unit quat ? "
                            np_sg_pose = np.array([sg_pose[0], sg_pose[1], q_a[0], q_a[-1]], dtype='float32')
                            
                        else:
                            np_context_action = np_context_pose
                            np_sg_pose = sg_pose

                    ts_context_action = torch.as_tensor(np_context_action, dtype=torch.float32)
                    ts_context_action = ts_context_action[None, ...]
                    ts_context_action = ts_context_action.to(device)

                    ts_goal_pose = torch.as_tensor(np_sg_pose, dtype=torch.float32)
                    ts_goal_pose = ts_goal_pose[None, ...]
                    ts_goal_pose = ts_goal_pose.to(device)

                    # predict distances and waypoints

                    #print("contxt + obs depth shape:", ts_obs_depths.shape)         # B(1), Contxt+curr (6), h, w
                    #print("goal depth shape:", ts_goal_depth.shape)                 # B(1), 1, h, w
                    #print("contxt + cur actions shape:", ts_context_action.shape)   # B(1), Contxt+curr (6), num_params
                    #print("goal shape", ts_goal_pose.shape)                         # B(1), num_params
                    
                    model_outputs = model(ts_obs_depths, ts_goal_depth, ts_context_action, ts_goal_pose)
                    ts_waypoints, ts_distance = model_outputs  # (1,5,4),  (1,2)

                    np_distance = to_numpy(ts_distance.squeeze())       #  (2, ) remaining dist to subgoal
                    np_waypoints = to_numpy(ts_waypoints.squeeze())     #  (5, 4)

                    # look for closest node
                    #closest_node = np.argmin(distances)
                    xy_distance = np_distance[0]
                    # chose subgoal and output waypoints

                    if xy_distance < 0.1:     # sg is reached.
                        print("<%d> th sg is reached "%curr_sg_idx )
                        is_sg_reached = True
                        curr_sg_idx = curr_sg_idx + 1
                        prev_sg_idx = prev_sg_idx + 1

                    else:
                        #print("dist to sg: %f"%xy_distance )
                        chosen_waypoint = np_waypoints[args.waypoint]
                        # convert x,y, qw, qz to (x, y, th)
                        quat = [chosen_waypoint[2], 0, 0, chosen_waypoint[3]]
                        _, _, _, rol, pit, yaw = rm.htm_to_xyzrpy( rm.quat_to_htm(quat) )
                        chosen_waypoint_se2 = np.array([x, y, yaw])
                        if model_params["normalize"]:
                            chosen_waypoint_se2 = _denormalize_pose( chosen_waypoint_se2, waypoint_spacing =waypoint_spacing,
                                                                 eta = 1.0, dataset_name = dataset_name  )

                        waypoint_msg = Float32MultiArray()
                        waypoint_msg.data = chosen_waypoint_se2
                        waypoint_pub.publish(waypoint_msg)

                # # TODO: set dist condition
                # # if XY reached
                # if xy_distance > xy_dist_thr:   #distances[closest_node] > args.close_threshold:
                #     chosen_waypoint = waypoints[closest_node][args.waypoint]
                #     sg_depth = topomap[start + closest_node]
                # else:
                #     chosen_waypoint = waypoints[min(
                #         closest_node + 1, len(waypoints) - 1)][args.waypoint]
                #     sg_depth = topomap[start + min(closest_node + 1, len(waypoints) - 1)]
            else:
                print("unknown model type: %s" %  model_params["model_type"])
                raise NotImplementedError

        # RECOVERY MODE
        # if model_params["normalize"]:
        #     chosen_waypoint[:2] *= (MAX_V / RATE)
        # waypoint_msg = Float32MultiArray()
        # waypoint_msg.data = chosen_waypoint
        # waypoint_pub.publish(waypoint_msg)
        # reached_goal = closest_node == goal_node
        # goal_pub.publish(reached_goal)
        # if reached_goal:
        #     print("Reached goal! Stopping...")
        # rate.sleep()


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
    print(f"depth topic {DEPTH_TOPIC}")
    main(args)


