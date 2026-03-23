import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import yaml

import action_utils as au

# ROS
import rospy
import tf2_ros
from tf.transformations import quaternion_matrix
import geometry_msgs.msg
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32MultiArray
#from ros_numpy import numpify

from utils import msg_to_pil, to_numpy, transform_images, load_model, msg_to_caminfo, set_tform

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

import cv2

# UTILS
from topic_names import (IMAGE_TOPIC, CAMERA_INFO,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)
from visualize_utils import (
    to_numpy,
    numpy_to_img,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)

# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
#cam_info = CameraInfo
context_queue = []
context_size = None  
subgoal = []
#base2cam = None

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_caminfo():
    dataset_name = 'former'
    with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
        data_config = yaml.safe_load(f)
        
    camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
    camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]
    fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
    fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
    cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
    cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
    camera_matrix = au.gen_camera_matrix(fx, fy, cx, cy)
    K = camera_matrix
    D = np.zeros((1,4), dtype=float)
    return K, D, camera_height, camera_x_offset

def read_caminfo():
    print("waiting for %s msg..."%CAMERA_INFO)
    msg=rospy.wait_for_message(CAMERA_INFO, CameraInfo)
    cam_info = msg_to_caminfo(msg)
    return cam_info

def read_base2cam():
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    # lookup_transform(a, b, time) # pose of frame b w.r.t frame a
    bl2cl = tfBuffer.lookup_transform('base_link', 'camera_link', rospy.Time(), timeout=rospy.Duration(1.0) ) 
    bl2o = tfBuffer.lookup_transform('base_link', 'camera_color_optical_frame', rospy.Time(), timeout=rospy.Duration(1.0))
    q0 = bl2cl.transform.rotation
    q1 = bl2o.transform.rotation
    bHc = quaternion_matrix([q0.x,q0.y,q0.z,q0.w])
    bHo = quaternion_matrix([q1.x,q1.y,q1.z,q1.w])
    tvec_base2cam = bl2cl.transform.translation
    tform_base2opt = set_tform(bl2o)
    bHo[:3,3] = np.array([tvec_base2cam.x, tvec_base2cam.y, tvec_base2cam.z])
    #print(bHo)
    # base2opt refers to cam coord frame (z front and y down) w.r.t base base_link
    # base2cam refers to camlink (z up and x front)
    return tform_base2opt, tvec_base2cam, bHo
   
def callback_obs(msg):
    #print("got an image")
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

def pinhole_projection(xy, K, cHw ):
    P = np.matmul(K, cHw[:3, ...])
    uvS = np.matmul(P, np.array([xy[0], xy[1], 0.0, 1.0], dtype='float') )
    u = uvS[0]/uvS[2]
    v = uvS[1]/uvS[2]
#    print("uv: (%f %f)"%(u,v) )
    return (u,v)


def main(args: argparse.Namespace):
    global context_size
    
     # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
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

     # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
    #cam_info, K, D, R, P = read_caminfo()
    K, D, cam_height, cam_x_offset = load_caminfo()
    #print(K)
    #print(R)
    #print(P)
    #print(D)
    #tform_base2opt, tvec_base2cam, bHc = read_base2cam()
    #cam_height = 0.17 #tvec_base2cam.z
    #cam_x_offset = tvec_base2cam.x
    print("cam_height/cam x offset: %f %f"%(cam_height, cam_x_offset))

    wHc = np.array([ [0.0, 0.0, 1.0, cam_x_offset], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, cam_height], [0.0, 0.0, 0.0, 1.0] ] )
    cHw = np.linalg.inv(wHc)

    totframecnt = 0 #,
    capture_time = time.time()
    # navigation loop

    closest_node = 0
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            #elif (len(context_queue) > model_params["context_size"]):   # ViNT model
            start = max(closest_node - args.radius, 0)              # start, end  should increases over time
            end   = min(closest_node + args.radius + 1, goal_node)
            distances = []
            waypoints = []
            batch_obs_imgs = []
            batch_goal_data = []
            for i, sg_img in enumerate(topomap[start: end + 1]):
                transf_obs_img = transform_images(context_queue, model_params["image_size"]) # transf_obs_img stays the same
                goal_data = transform_images(sg_img, model_params["image_size"]) # goal_data changes. t-4,t-3,t-2,t-1,t,t+1,t+2,t+3 
                batch_obs_imgs.append(transf_obs_img)
                batch_goal_data.append(goal_data)
                
            #print("transf_obs_img size: ", to_numpy(transf_obs_img).squeeze().shape)  # (18, 64, 85)
            #print("goal_data size:", to_numpy(goal_data).squeeze().shape)             # (3, 64, 85)
                #print("batch goal data size: ", len(batch_goal_data))
                # predict distances and waypoints

            # batch_obs = context_queue
            # batch_goals = topomap[start: end + 1]
            # for i in range(0, len(batch_obs)):
            #     fig_obs = plt.figure()
            #     obs_ = batch_obs[i]
            #     plt.imshow(obs_)
            #     outfile = '/media/data/results/ViNT/obs%04d_%04d.png'%(totframecnt,i)
            #     fig_obs.savefig(outfile)
            #     plt.close(fig_obs)
            #
            #     fig_goal = plt.figure()
            #     goal_ = batch_goals[i]
            #     plt.imshow(goal_)
            #     outfile = '/media/data/results/ViNT/goal%04d_%04d.png'%(totframecnt,i)
            #     fig_goal.savefig(outfile)
            #     plt.close(fig_goal)
                
            batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
            batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)
            
            # save batch obs and goals
            #print("context_queue:", len(context_queue))
            inf_time = time.time()
            distances, waypoints = model(batch_obs_imgs, batch_goal_data) # returns dist (temporal) and actions(wpts) for each batch
            print("inference time: %f"%( time.time() - inf_time ))
            distances = to_numpy(distances)
            print("quesize: %d batchsize: %d  out wps len: %d %d"%(len(context_queue), len(batch_obs_imgs), len(waypoints), len(distances)))
            waypoints = to_numpy(waypoints)
    # display projected project_points in obs image
            # look for closest node
            min_idx = np.argmin(distances)
            closest_node = start + min_idx # 0 ~ 5 the one marked shortest distance is the closest neighboring node
            # chose subgoal and output waypoints
            sg_idx = 0 #subgoal idx

            sg_idx = closest_node
            chosen_waypoint = waypoints[min_idx][args.waypoint]
            sg_img = topomap[sg_idx]

            # if distances[min_idx] > args.close_threshold:   # 3 by default... temporal distance
            #     chosen_topomap_idx = closest_node
            #     chosen_waypoint = waypoints[min_idx][args.waypoint]
            #     sg_img = topomap[chosen_topomap_idx]
            # else:
            #     chosen_topomap_idx = min(closest_node + 1, len(waypoints) - 1)
            #     chosen_waypoint = waypoints[chosen_topomap_idx][args.waypoint]
            #     sg_img = topomap[chosen_topomap_idx]

            #print("distances: ", distances.transpose())
            #print("chosen waypoint: ",chosen_waypoint) 
            (batchsize, hsize, wpndim) = waypoints.shape
            ppts = np.zeros( (2, hsize), dtype=float)
            curr_img = context_queue[-1]
            chosen_waypoint_horiz = waypoints[min_idx] # (5,4)
            #for bidx in range(0, batchsize):
            for nidx in range(0, hsize): # 5 pred horiz
                wp = chosen_waypoint_horiz[nidx] * MAX_V
                (u, v) = pinhole_projection(wp[:2], K, cHw)
                #print("wp uv (%f %f)  (%f %f)"%(wp[0], wp[1], u, v))
                ppts[0, nidx] = np.clip(u, 0, 640-1)
                ppts[1, nidx] = np.clip(v, 0, 480-1)
            # print("km ppts:")
            # print(ppts)
            #if( abs(capture_time - time.time()) > 0.0): #0.2 ): # 5 hz, 6 imgs per sec
                #print(chosen_waypoint[np.newaxis,np.newaxis,:2].shape)
                #print(waypoints[...,:2].shape)
#                uv = project_points(chosen_waypoint_horiz[np.newaxis,...,:2], cam_height, cam_x_offset, K, D)
#                uv_target = project_points(chosen_waypoint[np.newaxis,np.newaxis,:2], cam_height, cam_x_offset, K, D)
                #print(uv.shape)
                #print(uv_target.shape)
            fig = plt.figure( figsize=(12, 6))
            ax0, ax1 = fig.subplots(1, 2)
            #ax.imshow(curr_img)
            #print(len(list(waypoints)))
            au.plot_trajs_and_points_on_image(ax0, curr_img, 'former', list(waypoints*MAX_V), list(chosen_waypoint_horiz*MAX_V), [YELLOW]*20, [RED, MAGENTA, GREEN, CYAN, BLUE] )
            au.plot_trajs_and_points_on_image(ax0, curr_img, 'former', [chosen_waypoint_horiz*MAX_V], [], [CYAN], [RED, GREEN] )

            # ax0.plot(ppts[0,0], ppts[1,0], "xr", markersize=10)
            # ax0.plot(ppts[0,1], ppts[1,1], "xm", markersize=10)
            # ax0.plot(ppts[0,2], ppts[1,2], "xg", markersize=10)
            # ax0.plot(ppts[0,3], ppts[1,3], "xc", markersize=10)
            # ax0.plot(ppts[0,4], ppts[1,4], "xb", markersize=10)

            ax1.imshow(sg_img)
            ax1.text(50, 50, 'start idx: %d'%(start), bbox=dict(fill=True, color = 'yellow', edgecolor='red', linewidth=2))
            ax1.text(200, 50, 'selected topoimg idx: %d'%(sg_idx), bbox=dict(fill=True, color = 'yellow', edgecolor='red', linewidth=2))
            #for bidx in range(0,batchsize):
            #plt.scatter(uv.squeeze()[...,0], uv.squeeze()[...,1], marker="x", color="red", s=10)
            #plt.scatter(uv_target.squeeze()[0], uv_target.squeeze()[1], marker="o", color="cyan", s=20)
            #(uc, vc) = pinhole_projection(chosen_waypoint[:2], K, bHc, cam_height)
            #plt.scatter(uc, vc,   marker=MarkerStyle("o", fillstyle="full"), color='green', s=20) 
            outfile = '/media/data/results/ViNT/img%04d.png'%(totframecnt)
            fig.savefig(outfile)
            plt.close(fig)
            capture_time = time.time()
            totframecnt = totframecnt + 1
            #print(waypoints)
            #print(distances)
            print("start idx: %d, end idx: %d, and closest topmap node idx: %d" %(start, end, closest_node) )
            # save to metadata for debugging
            mdfile = open('/media/data/results/ViNT/md%04d.txt'%(totframecnt), 'w')
            strtxt = '%d %d %d %d\n'%(start, end, min_idx, closest_node)
            mdfile.write(strtxt)  # start, end, selected topomap idx   
            dist_str = " ".join(str(element[0]) for element in distances)
            mdfile.write(dist_str) # distances

            ##########################################################################################
            # tform the local idx closest_node  to global idx
            ##########################################################################################
            #closest_node = start + closest_node


        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)
        reached_goal = closest_node == goal_node  # this is wrong b/c closest_node is always bounded in 0 ~ 5
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
        rate.sleep()


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
    print(f"img topic {IMAGE_TOPIC}")
    main(args)


