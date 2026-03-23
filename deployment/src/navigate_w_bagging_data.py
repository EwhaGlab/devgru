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
from sensor_msgs.msg import Image, CameraInfo, Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension
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

from navdata_collector.msg import waypoint_stamped
from navigator import navigator


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

DEADMAN_BUTTON = robot_config["deadman_button"]

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

img_width, img_height = model_params["image_size"]
normalize = model_params['normalize']
datacollector_start_msg = None
joy_flag = None
finish_flag = None
joy_msg = None

def joy_callback(msg):
    global joy_msg
    joy_msg = msg

def joyflag_callback(msg):
    global joy_flag
    joy_flag = msg.data

def main(args: argparse.Namespace):
    global joy_flag, finish_flag

    dataset_name = 'former'  # model_params["datasets"]
    topomap_dir = args.topomap_dir #DEPLOYMENT_DIR + "/topomaps/" + deployment_params['topomap_name']

    subgoal_spacing = deployment_params['subgoal_spacing'] #int(model_params['distance']['max_frame_dist'] / 2)
    use_coll_pred_model = deployment_params['use_coll_pred_model'] # whether to employ coll_pred_model or not when driving the robot
    #################################################################################################
    # 1. load model weights
    #################################################################################################
    nav_model_type = deployment_params['nav_model_type']
    col_model_type = deployment_params['col_model_type']
    waypoint_spacing = model_params['deployment']['waypoint_spacing']

    # 1-1 loading the main control model
    nav_ckpth_name = model_params['deployment']["nav_ckpth_name"]
    nav_ckpth_path = MODEL_WEIGHTS_PATH + "/" + nav_ckpth_name

    if os.path.exists(nav_ckpth_path):
        print(f"Loading model from {nav_ckpth_path}")
    else:
        raise FileNotFoundError(f"Nav Model weights not found at {nav_ckpth_path}")
    nav_model = load_model(
        nav_ckpth_path,
        nav_model_type,
        model_params,
        device,
    )
    nav_model = nav_model.to(device)
    nav_model.eval()

    # 1-2 loading the collision detection model if necessary
    if use_coll_pred_model:
        col_ckpth_name = model_params['deployment']["col_ckpth_name"]
        col_ckpth_path = MODEL_WEIGHTS_PATH + "/collision/" + col_ckpth_name

        if os.path.exists(col_ckpth_path):
            print(f"Loading model from {col_ckpth_path}")
        else:
            raise FileNotFoundError(f"Collision Model weights not found at {col_ckpth_path}")
        col_model = load_model(
            col_ckpth_path,
            col_model_type,
            model_params,
            device,
        )
        col_model = col_model.to(device)
        col_model.eval()

    #################################################################################################
    # 2. load topomap imgs
    #################################################################################################
    synced_images_all = os.listdir(topomap_dir)
    # image_type = model_params['goal_type']
    topomap_rgb_files = [f for f in synced_images_all if 'rgb' in f]
    topomap_depth_files = [f for f in synced_images_all if 'depth' in f]

    topomap_rgb_filenames = sorted(topomap_rgb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    topomap_depth_filenames = sorted(topomap_depth_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    num_nodes = len(topomap_depth_filenames)

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

    assert -1 <= args.goal_node < len(topomap_depth), "Invalid goal index"
    if args.goal_node == -1:
        goal_node_idx = len(topomap_depth) - 1
    else:
        goal_node_idx = args.goal_node

    is_goal_reached = False

    #################################################################################################
    # 3. load topomap odom and transform them to local coord frame
    #################################################################################################
    #topomap_odom = os.path.join(topomap_dir + '/topo_odom.txt')
    topomap_odom = os.path.join(topomap_dir + '/topo_tf_m2b.txt')
    odom_info = np.loadtxt(topomap_odom)

    print(odom_info.shape)
    topomap_odom = []
    w1Hb = np.tile(np.eye(4), (num_nodes, 1, 1))
    cnt = 0
    for idx in range(0, num_nodes):
        #print("%d .. %d / %d %d \n"%(cnt, i, num_nodes, tot_nodes) )
        x = odom_info[idx, 4]  # x
        y = odom_info[idx, 5]  # y
        q = odom_info[idx, 7:11]  # qx, qy, qz, qw
        quat = [q[-1], q[0], q[1], q[2]]
        w1Hb[idx] = rm.quat_to_htm(quat)
        # _,_,_,_,_,yaw = rm.htm_to_xyzrpy(w1Hb[cnt])
        # yaw_n = rm.normalizeAngle(yaw)
        w1Hb[idx,0,3] = x
        w1Hb[idx,1,3] = y
        #print("sg %d: x:%f  y:%f  th: %f\n" % (i, x, y, yaw_n))

    b0Hw1 = np.linalg.inv(w1Hb[0].copy())
    for i in range(0, num_nodes):
        b0Hb_i = np.matmul( b0Hw1, w1Hb[i] )
        [xt, yt, _, _, _, yaw_t] = rm.htm_to_xyzrpy(b0Hb_i)
        #yaw_n = rm.normalizeAngle(yaw)
        topomap_odom.append([xt, yt, yaw_t]) # [x0, y0, th0] == [0, 0, 0]
        #print("tformed sg %d: x:%f  y:%f  th:%f \n"%(i, xt, yt, yaw_t) )

    # begin ROS
    rospy.init_node("navigator_parent_node", anonymous=False)
    rate = rospy.Rate(RATE)

    # start the data collector node
    #tmp = "/home/glab/catkin_ws/src/navdata_collector/run_script/data_collector"
    #datacollection_node_path = os.path.join(tmp, 'run_colldata_collection.py')
    #process = subprocess.Popen(['python3', datacollection_node_path])
    print("waiting for the data_collection node... \n")
    rospy.wait_for_message('/navdata_collector/colldata_collection_node_is_up', Bool, timeout=None)
    print('***********************************************************************')
    print("\n colldata_collector node is up! Starting the navigation process ! \n")

    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    
    joy_sub = rospy.Subscriber("/joy", Joy, joy_callback)
    joyflag_sub = rospy.Subscriber('/joy_flag', Bool, joyflag_callback)
    
    process_finish_pub = rospy.Publisher("/finish_data_collection", Bool, queue_size=1, latch=True)
    
    # navigator instance
    nav = navigator(name='viz_navigator', dataset_name='former',
                             topomap_rgb = topomap_rgb,
                             topomap_depth = topomap_depth,
                             topomap_odom = topomap_odom,
                             topomap_m1_to_sg = None,
                             config = model_params, data_config=data_config, robot_config=robot_config)

    #################################################################################################
    # 4. Start the main navigation loop
    #################################################################################################

    # Make sure to have the robot's init pose is aligned with the starting position of the sug-goal trajectory
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
        if (finish_flag == True):
            rospy.logerr("Got finishing cmd. Shutting down the process \n")
            process_finish_pub.publish(Bool(data=True))
            break

        #print("curr target SG idx is: <%d> \n" % (nav.curr_sg_idx))
        if len(nav.context_depth_queue) > model_params["context_size"]:

            #if model_params["model_type"] == "image_pose_rnn":  # (len(context_queue) > model_params["context_size"]):
            nav._is_sg_reached = False
            xy_dist_m = 1.
            orient_dist = 0.
            print("Moving toward the new subgoal \n")
            while not nav._is_sg_reached:
                if rospy.is_shutdown():
                    break

                if nav.got_new_metadata(): # new rgb-d data is received
                    if use_coll_pred_model:
                        np_waypoints_m, np_pose_diff_m, _ = nav.navstep(
                            nav_model, nav_model_type,
                            col_model, col_model_type,
                            dataset_name)

                    else:
                        raise NotImplementedError

                    # compute distance to sg
                    xy_dist_m = np.linalg.norm(np_pose_diff_m[:2])
                    orient_dist = 2.0 * np.arctan2(np_pose_diff_m[3], np_pose_diff_m[2]) #
                    np_distance_m = np.array([xy_dist_m, orient_dist], dtype=np.float32)
                    nav.publish_navdata(joy_flag, np_waypoints_m, np_pose_diff_m, np_distance_m, nav.get_collision_status())
                    nav.publish_curr_rel_subgoal()

                nav.update_subgoal_status( xy_dist_m ) #, np_pose_diff_m , orient_dist )
                # chose subgoal and output waypoints
                # print(f"is_subgoal_reached: {nav._is_subgoal_reached}")
                if ( nav.is_subgoal_reached() == True):
                    print("Updating SG \n")
                    nav.update_subgoal()
                    nav._correction_latch = False
                    break
                
                if joy_msg.buttons[DEADMAN_BUTTON] == 1 and joy_msg.buttons[DEADMAN_BUTTON+1] == 1 and joy_msg.buttons[DEADMAN_BUTTON+2] == 1:
                    finish_flag = True
                    break
                
                
        # # RECOVERY MODE
        # if model_params["normalize"]:
        #     chosen_waypoint[:2] *= (MAX_V / RATE)
        # waypoint_msg = Float32MultiArray()
        # waypoint_msg.data = chosen_waypoint
        # waypoint_pub.publish(waypoint_msg)
        # reached_goal = closest_node == goal_node  # this is wrong b/c closest_node is always bounded in 0 ~ 5
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
    
    parser.add_argument(
        "--topomap_dir",
        "-i",
        type=str,
        required=True,
        help="Base path to saved map (no extension)"
    )
    
    args = parser.parse_args()
    print(f"Using {device}")
    print(f"img topic {RGB_TOPIC}")
    main(args)


