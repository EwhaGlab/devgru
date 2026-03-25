import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
import matplotlib

from utils.rigid_motion import htm_to_xyzrpy
BASE_DIR = dirname(dirname(abspath(__file__)))
#from data.ipython_collision_dataset import dataset_name

matplotlib.use("Agg")

# ROS
import rospy
import message_filters
from message_filters import TimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo, Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32, Float32MultiArray, MultiArrayDimension
from nav_utils import msg_to_pil, msg_to_pil_depth, to_numpy, transform_images, load_model, msg_to_caminfo, set_tform
import tf2_ros

from PIL import Image as PILImage
import yaml
import time

import cv2
import argparse

import math
from cv_bridge import CvBridge
bridge = CvBridge()

#from navdata_collector.msg import waypoint_stamped
from os.path import dirname, abspath
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.append(BASE_DIR)
import utils.rigid_motion as rm
from visualize_utils import (
    #plot_pred_cam_heading_and_points_simple,
    #plot_projected_pred_trajs_and_points_on_image,
    visualize_nav_status,
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

from matplotlib.gridspec import GridSpec

# UTILS
from topic_names import (RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO,
                        NAVDATA_TOPIC,
                        RELATIVE_SG_TOPIC,
                        ODOM_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_VIZ_START_OFFSET = 0.6
VIZ_SIZE = (320, 240) #(160, 120)

class navigator():
    def __init__(self, name, dataset_name: str,
                 topomap_rgb: list,     # PILImage
                 topomap_depth: list,   # PILImage
                 topomap_odom: list,    #
                 config, data_config, robot_config):
        self.name = name
        self.model_params = config
        self.data_config = data_config
        self.robot_config = robot_config
        self.max_frame_dist = self.model_params['distance']['max_frame_dist']
        # GLOBALS
        self.context_rgb_queue = []
        self.context_depth_queue = []
        self.context_pose_queue = []
        self.context_size = self.model_params['context_size']
        self.obs_queue_size = self.context_size + 1  # context + curr
        self.subgoal = []
        self.prev_sg_idx = 0
        self.curr_sg_idx = 0
        self.learn_angle = self.model_params['learn_angle']
        self.img_width = 0
        self.img_height = 0

        self.curr_obs_rgb = []
        self.sg_rgb = []
        self.sg_depth = []

        self.viz_curr_rgb = []
        self.viz_curr_depth = []
        self.viz_sg_rgb = []
        self.viz_sg_depth = []
        self.prev_sg_pose = []
        self.curr_rel_sg_pose = []   # SE2 SG pose wrt the curr robot pose
        self.curr_robot_pose = []    # SE2 robot pose
        self.curr_robot_pose_corrected = []  # corrected SE2 robot pose

        self.correcting_htm = np.eye(4)  #  w1_H_w2 # curr world's orig w.r.t prev world's orig
        self.waypoint_callback_cnt = 0
        self._is_subgoal_reached = False
        self._is_finalgoal_reached = False
        self.xy_dist_thr = self.model_params['deployment']['xy_dist_thr']
        self.ang_dist_thr = self.model_params['deployment']['ang_dist_thr']
        self._is_collision = False
        self._correction_latch = False
        self._metadata_counter = 0
 #       self.ax0 = None
        fig_title = 'fig_res'
        self.fig = plt.figure(layout="constrained")
        #self.fig.suptitle(fig_title, fontsize=12)
        self.fig.set_size_inches(12, 6)

        gs = GridSpec(2, 3,
                      figure=self.fig,
                      width_ratios=[1.2, 1.0, 1.0],
                      height_ratios=[1.0, 1.0] )

        self.ax0 =  self.fig.add_subplot(gs[:, 0])
        self.ax01 = self.fig.add_subplot(gs[0, 1])
        self.ax02 = self.fig.add_subplot(gs[0, 2])
        self.ax11 = self.fig.add_subplot(gs[1, 1])
        self.ax12 = self.fig.add_subplot(gs[1, 2])
        self.status_text_obj = None

        # print("got synced odom and depth image %d contxt size" %context_size )
        assert self.context_size > 0

        self.img_width, self.img_height = self.model_params["image_size"]
        self.normalize = self.model_params['normalize']
        self.dataset_name = dataset_name

        self.waypoint_spacing = self.model_params['datasets']['former']['waypoint_spacing']
        self.callback_fn_freq = 10. / float(self.waypoint_spacing)
        self.last_callback_time = 0
        self.curr_callback_time = time.time()
        self._got_new_metadata = False

        self.topomap_depth = topomap_depth
        self.topomap_rgb = topomap_rgb
        self.topomap_odom = topomap_odom.copy()    # tformed w.r.t the 1st robot pose. i.e, the coord starts from [0, 0, 0]
        self.topomap_odom_old = topomap_odom.copy()

        self.num_nodes = len(self.topomap_depth)

        _,_,_,_,_,yaw  = rm.htm_to_xyzrpy(w2Hr)
        yaw_n = rm.normalizeAngle(yaw)
        w2Hr[0,3] = rp.x
        w2Hr[1,3] = rp.y

        self.correcting_htm = np.linalg.inv(w2Hr) #np.matmul(w1Hr0, np.linalg.inv( w2Hr ) )  # w1Hw2
        self.curr_robot_pose_corrected = [0, 0, 0]

        # subscribers
        self.rgb_curr_msg = rospy.Subscriber(RGB_TOPIC, Image, self.callback_curr_obs, queue_size=1)
        #self.draw_waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, waypoint_stamped, self.draw_simple_waypoint_callback, queue_size=1)
        self.draw_navdata_sub = rospy.Subscriber(NAVDATA_TOPIC, navdata_stamped, self.draw_simple_waypoint_callback,
                                                  queue_size=1)

        self.depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)
        self.odom_sub = message_filters.Subscriber(ODOM_TOPIC, Odometry)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.odom_sub],
                                                                queue_size=100, slop=0.1)  # 100ms tolerance
        self.sync.registerCallback(self.synced_data_callback)
        
        # publisher
        self.rel_sg_publisher = rospy.Publisher(RELATIVE_SG_TOPIC, Odometry, queue_size=1)
        #self.waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, waypoint_stamped, queue_size=1)
        self.navdata_pub = rospy.Publisher(NAVDATA_TOPIC, navdata_stamped, queue_size=1)

#        [x0, y0, th0] = self.topomap_odom[0]
#        w1Hr0 = rm.xyzrpy_to_htm([x0, y0, 0, 0, 0, th0])
        # wait for 1 odom message to compute the correcting htm
        print("\n Check for essential metadata topics ... \n")
        try:
            rospy.wait_for_message(DEPTH_TOPIC, Odometry, timeout=5)
            print("\033[92mDepth topic: %s is avaiable\033[0m" % DEPTH_TOPIC)
        except rospy.ROSException:
            sys.exit("\033[91mDepth topic: %s is NOT avaiable\033[0m" % DEPTH_TOPIC)
        try:
            rospy.wait_for_message(RGB_TOPIC, Odometry, timeout=5)
            print("\033[92mRGB topic: %s is avaiable\033[0m"%RGB_TOPIC)
        except rospy.ROSException:
            sys.exit("\033[91mRGB topic: %s is NOT avaiable\033[0m" % RGB_TOPIC)

        print("\n Check for joy message ... \n")
        try:
            rospy.wait_for_message('joy', Joy, timeout=5)
            print("\033[92mJoystick is connected \033[0m")
        except rospy.ROSException:
            sys.exit("\033[91mMake sure to have the joystick is connected\033[0m")

    def set_collision_status(self, is_collision = False):
        self._is_collision = is_collision

    def get_collision_status(self):
        return self._is_collision

    def correct_robot_pose(self, in_robot_pose):
        [x_w2, y_w2, th_w2] = in_robot_pose
        w2Hr = rm.xyzrpy_to_htm( [x_w2, y_w2, 0, 0, 0, th_w2] )
        r0Hr = np.matmul( self.correcting_htm, w2Hr )
        [x, y, _, _, _, th] = rm.htm_to_xyzrpy(r0Hr)
        return [x, y, th]

    def read_caminfo(self):
        print("waiting for %s msg..." % CAMERA_INFO)
        msg = rospy.wait_for_message(CAMERA_INFO, CameraInfo)
        cam_info = msg_to_caminfo(msg)
        return cam_info

    def callback_curr_obs(self, msg):
        # print("got a rgb image")
        rgb = msg_to_pil(msg)
        self.viz_curr_rgb = rgb.resize(VIZ_SIZE)

    def pinhole_projection(self, xy, K, cHw):
        P = np.matmul(K, cHw[:3, ...])
        uvS = np.matmul(P, np.array([xy[0], xy[1], 0.0, 1.0], dtype='float'))
        u = uvS[0] / uvS[2]
        v = uvS[1] / uvS[2]
        #    print("uv: (%f %f)"%(u,v) )
        return (u, v)

    # The training data ( /rgbd_throttle/rgbd ) was collected @ 10 hz
    # see navdata_collector/launch/include/start_bag_async.laucn.. wanted 12 hz but record @ 10 ~ 11 hz

    def synced_data_callback(self, depth_msg, odom_msg):
        now = time.time()
        #print("cb freq: %f" % self.callback_fn_freq)
        if (now - self.last_callback_time) < (1. / self.callback_fn_freq):
            self._got_new_metadata = False
            return

        if self.curr_sg_idx >= self.num_nodes:
            return

        odom = odom_msg
        q = odom.pose.pose.orientation
        quat = [q.w, q.x, q.y, q.z]
        _,_,_,_,_,yaw = rm.htm_to_xyzrpy( rm.quat_to_htm(quat) )
        yaw_norm = rm.normalizeAngle(yaw)
        self.curr_robot_pose = [odom.pose.pose.position.x, odom.pose.pose.position.y, yaw_norm]
        self.curr_robot_pose_corrected = self.correct_robot_pose(self.curr_robot_pose)

        xc, yc, thc         = self.curr_robot_pose_corrected
        x_sg, y_sg, th_sg   = self.topomap_odom[self.curr_sg_idx]
        rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([xc, yc, thc]),
                                                    np.array([x_sg, y_sg, th_sg]),
                                                    self.dataset_name)
        self.curr_rel_sg_pose = np.array([rel_x, rel_y, rel_theta])
        fxy_dist = (rel_x**2 + rel_y**2) ** 0.5

        elapsed_time = now - self.last_callback_time
        metadata_cb_freq = 1.0 / elapsed_time
        print("rel sg <%d> pose: %.3f(m) %.3f(m) %.3f(deg)  dist to sg: %.3f \n"
              %(self.curr_sg_idx, rel_x, rel_y, rel_theta * 180/math.pi, fxy_dist ) )

        # if (fxy_dist < 0.1):
        #     self._is_subgoal_reached = True

        obs_depth = depth_msg
        obs_rgb = self.viz_curr_rgb #rgb_msg

        pil_depth = msg_to_pil_depth(depth_msg)
        np_depth = np.array(pil_depth)
        depth_normalized = np.array( pil_depth.resize(VIZ_SIZE) ).astype(np.float32) / MAX_DEPTH * 255.
        depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)
        #print("msg depth minmax: %f %f " % (np_depth.min(), np_depth.max()))
        #print("viz depth minmax: %d %d \n"%(depth_normalized.min(), depth_normalized.max()) )
        self.viz_curr_depth  = PILImage.fromarray(depth_normalized)

        if len(self.context_depth_queue) < self.context_size + 1:
            self.context_depth_queue.append(pil_depth)
            self.context_pose_queue.append(self.curr_robot_pose_corrected)
            self.context_rgb_queue.append(obs_rgb)
        else:
            self.context_depth_queue.pop(0) # removes the first itme
            self.context_pose_queue.pop(0)
            self.context_depth_queue.append(pil_depth) # add to the end
            self.context_pose_queue.append(self.curr_robot_pose_corrected)
            # context_rgb_queue.pop(0)
            # context_rgb_queue.append(obs_rgb)
        # print("[depth, odom] callback processed @ %f Hz \t queue_size %d" % (1. / callback_fn_interval, len(context_depth_queue)) )
        self.last_callback_time = now
        self._got_new_metadata = True
        self._metadata_counter += 1

    def got_new_metadata(self):
        return self._got_new_metadata

    def update_subgoal(self):
        self.curr_sg_idx += 1
        if self.curr_sg_idx >= self.num_nodes:
            self._is_finalgoal_reached = True
            self._is_subgoal_reached = True
            return

        self.sg_depth = self.topomap_depth[self.curr_sg_idx]  #
        self.sg_rgb   = self.topomap_rgb[self.curr_sg_idx]
        np_sg_depth = np.array( self.sg_depth.resize(VIZ_SIZE) ).astype(np.float32)
        depth_normalized= np_sg_depth / MAX_DEPTH * 255.
        #print("viz sg depth minmax: %d %d \n"%(np_sg_depth.min(), np_sg_depth.max()) )
        depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)
        self.viz_sg_depth = PILImage.fromarray(depth_normalized)
        self.viz_sg_rgb   = self.sg_rgb.resize(VIZ_SIZE)

        xc, yc, thc         = self.curr_robot_pose_corrected
        x_sg, y_sg, th_sg   = self.topomap_odom[self.curr_sg_idx]
        rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([xc, yc, thc]),
                                                    np.array([x_sg, y_sg, th_sg]),
                                                    self.dataset_name)
        self.curr_rel_sg_pose = np.array([rel_x, rel_y, rel_theta])

        # print("<%d>th target rel sg pose: %f(m) %f(m) %f(deg)"%(self.curr_sg_idx, self.curr_rel_sg_pose[0],
        #                                     self.curr_rel_sg_pose[1], self.curr_rel_sg_pose[2] * 180/math.pi) )
        [x_g, y_g, th_g] = self.topomap_odom[-1]
        rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([xc, yc, thc]),
                                                    np.array([x_g, y_g, th_g]),
                                                    self.dataset_name)
        dist_to_goal = ( rel_x**2 + rel_y**2 ) ** 0.5
        print("dist to the final sg: %f" %(dist_to_goal) )

    def update_subgoal_status(self, xy_dist): #, np_pose_diff, orient_dist):
        if self.curr_sg_idx >= self.num_nodes:
            print("cannot update sg anymore b/c the final goal is reached")
            self._is_subgoal_reached = True
            self._is_finalgoal_reached = True
            return

        #x_sg, y_sg, th_sg = self.topomap_odom[self.curr_sg_idx]
        #dist = np.linalg.norm( np.array([x_sg, y_sg]) - self.curr_robot_pose_corrected[:2])
        #print("Dist to <%d>th SG / THR is <%f> (%.2f %.2f) / <%f> \n"
        #      %(self.curr_sg_idx, xy_dist, np_pose_diff[0], np_pose_diff[1], self.xy_dist_thr ) )
        if xy_dist < self.xy_dist_thr:
            print("\033[36m <%d>th SG is reached \033[0m\n",self.curr_sg_idx)
            self._is_subgoal_reached = True
        else:
            self._is_subgoal_reached = False
        return

    def is_subgoal_reached(self) -> bool:
        return bool(self._is_subgoal_reached)

    # def topomap_update_callback(self, msg):
    #     self.curr_topomap_idx = msg.data
    # def is_subgoal_reached(self):
    #     x0, y0, th0         = self.curr_robot_pose_corrected
    #     x_sg, y_sg, th_sg   = self.topomap_odom[self.curr_sg_idx]
    #     rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([x0, y0, th0]),
    #                                                 np.array([x_sg, y_sg, th_sg]),
    #                                                 self.dataset_name)
    #     self.curr_rel_sg_pose = np.array([rel_x, rel_y, rel_theta])
    #     fxy_dist = (rel_x**2 + rel_y**2) ** 0.5
    #     print("dist to curr sg: %.3f"% fxy_dist)
    #     print("[%.4f, %.4f ] / [%.4f, %.4f ]"%( x0, y0, x_sg, y_sg) )
    #     if (fxy_dist < 0.1):
    #         return True
    #     else:
    #         return False

    def draw_simple_waypoint_callback(self, navdata_msg): #waypoint_msg):
        data1_raw = np.array(navdata_msg.waypoints.data, dtype=np.float32)
        dims = navdata_msg.waypoints.layout.dim

        if len(dims) < 2:
            rospy.logwarn("Waypoint layout dimension is less than 2. Cannot reshape.")
            return

        rows = dims[0].size
        cols = dims[1].size
        pred_waypoint = data1_raw.reshape((rows, cols))  # should be denormalized before
        pose_diff_pred = np.array(navdata_msg.pose_diff.data, dtype=np.float32)

        # We consider cam orig as the start_pos for the visualization
        start_pos = np.array([0.0, 0.0, 0.0, 0.0])
        # goal_pos = self.topomap_odom[self.curr_sg_idx]
        goal_pos = self.curr_rel_sg_pose

        # print("wpt shape: ", pred_waypoint.shape)
        if len(pred_waypoint.shape) > 2:
            raise Exception(" Must be only single way point \n")
            # list_waypoints = [*pred_waypoint, label_waypoint]
        else:
            list_waypoints = pred_waypoint

        b_is_collision = navdata_msg.is_collision.data
        print("received coll: ",b_is_collision)

        status_str = "COLLISION" if b_is_collision else "SAFE"
        latch_str = "TRUE" if self._correction_latch else "FALSE"

        info = (
            f"STATUS: {status_str}  |  "
            f"SG ID: <{self.curr_sg_idx}> / <{len(self.topomap_odom)}>  \n"
            f"CORR LATCH: <{latch_str}>"
        )

        if b_is_collision:
            status_color = 'red'
        else:
            status_color = 'green'

        if self.status_text_obj is not None:
            self.status_text_obj.remove()
        self.status_text_obj = self.fig.text(
            0.2,  # center horizontally
            0.95,  # slightly below the title
            info,
            ha='center',
            va='top',
            fontsize=12,
            weight="bold",
            color=status_color,
        )

        visualize_nav_status(
            self.fig,
            self.ax0, self.ax01, self.ax02, self.ax11, self.ax12,
            info,
            self.viz_curr_depth,  # obs_img,
            self.viz_sg_depth,  # goal_img,
            self.viz_curr_rgb,
            self.viz_sg_rgb,
            self.dataset_name,  # robot name "former"
            goal_pos,
            pred_waypoint,  # (len_pred, num_param)
            pose_diff_pred,
            b_is_collision
        )


        imgfile = '/home/glab/results/dev_gru/%05d.png' % self.waypoint_callback_cnt
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        self.fig.tight_layout()
        self.fig.savefig(imgfile)
        self.fig.canvas.draw()
        img_np = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np = img_np.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Visualization", img_bgr)
        cv2.waitKey(1)

        self.waypoint_callback_cnt += 1


    # def draw_waypoint_callback(self, waypoint_msg):
    #     data_raw = np.array(waypoint_msg.data, dtype=np.float32)
    #     dims = waypoint_msg.layout.dim
    #
    #     rows = dims[0].size
    #     cols = dims[1].size
    #     pred_waypoint = data_raw.reshape((rows, cols))  # should be denormalized before
    #
    #     self.ax0.cla()
    #     self.ax1.cla()
    #
    #     # We consider cam orig as the start_pos for the visualization
    #     bHc = rm.xyzrpy_to_htm(self.data_config[self.dataset_name]["camera_matrics"]["cam_wrt_base"])
    #     x_offset = bHc[0, 3] + X_VIZ_START_OFFSET  # +0.6 projects uv at the bottom of the (isaac sim)imgs
    #     y_offset = bHc[1, 3]
    #
    #     start_pos = np.array([0, 0, 0.0, 0.0])
    #     start_pos_w_offset = np.array([x_offset, y_offset, 0.0, 0.0])  # start cam pose
    #
    #     #goal_pos = self.topomap_odom[self.curr_sg_idx]
    #     goal_pos = self.curr_rel_sg_pose
    #     print("sg pos %f %f"%(goal_pos[0], goal_pos[1]))
    #     goal_pos_w_offset = goal_pos.copy()
    #     goal_pos_w_offset[0] += x_offset
    #     goal_pos_w_offset[1] += y_offset
    #
    #     #print("wpt shape: ", pred_waypoint.shape)
    #     if len(pred_waypoint.shape) > 2:
    #         raise Exception(" Must be only single way point \n")
    #         # list_waypoints = [*pred_waypoint, label_waypoint]
    #     else:
    #         list_waypoints = pred_waypoint
    #         np_pred_waypoint_w_offset = pred_waypoint.copy()
    #         np_pred_waypoint_w_offset[:, 0] += x_offset
    #         np_pred_waypoint_w_offset[:, 1] += y_offset
    #
    #     plot_pred_cam_heading_and_points(
    #         self.ax0,  # ax[0][0],
    #         list_waypoints,  # scale up for vizualization
    #         [start_pos, goal_pos],  # wrt base
    #         waypoint_color= CYAN,
    #         point_colors=[GREEN, RED],
    #     )
    #
    #     #plot_projected_pred_trajs_and_points_on_image(self.ax01,
    #                                       #self.viz_curr_rgb,
    #                                       #self.dataset_name,
    #                                       #np_pred_waypoint_w_offset,
    #                                       #[start_pos, goal_pos], #list(chosen_waypoint_horiz * MAX_V),
    #                                       #waypts_color = CYAN,
    #                                       #point_colors = [GREEN, RED])
    #
    #                                       #[RED, MAGENTA, GREEN, CYAN, BLUE])
    #
    #     self.ax01.imshow(self.viz_curr_rgb)
    #     self.ax02.imshow(self.viz_sg_rgb)
    #     self.ax11.imshow(self.viz_curr_depth,  cmap='gray', vmin=0, vmax=255)
    #     self.ax12.imshow(self.viz_sg_depth, cmap='gray', vmin=0, vmax=255)
    #     self.ax0.set_title(f"Action Prediction")
    #     self.ax01.set_title(f"Observation")
    #     self.ax02.set_title(f"Goal")
    #
    #     imgfile = '/home/glab/results/dev_gru/%05d.png' % self.waypoint_callback_cnt
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()
    #     self.fig.tight_layout()
    #     self.fig.savefig(imgfile)
    #     self.waypoint_callback_cnt += 1

    def publish_curr_rel_subgoal(self):
        np_pose = self.curr_rel_sg_pose
        odom = Odometry()
        odom.pose.pose.position.x = np_pose[0]
        odom.pose.pose.position.y = np_pose[1]
        odom.pose.pose.position.z = 0

        q = rm.rpy2quat(0,0,np_pose[2])
        odom.pose.pose.orientation.w = q[0]
        odom.pose.pose.orientation.x = q[1]
        odom.pose.pose.orientation.y = q[2]
        odom.pose.pose.orientation.z = q[3]
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "base_link"
        self.rel_sg_publisher.publish(odom)

    def publish_navdata(self, joy_flag,
                        np_waypoints_m: np.ndarray,
                        np_pose_diff_m: np.ndarray,
                        np_distance_m: np.ndarray,
                        b_is_collision: bool):
        (num_traj_len, num_params) = np_waypoints_m.shape

        # out_pose_diff is reserved for the future
        # np_pose_diff # [dx, dy, dw, dz]
        out_pose_diff = np_pose_diff_m.copy()

        navdata_msg = navdata_stamped()
        navdata_msg.header.stamp = rospy.Time.now()
        navdata_msg.joystick = Bool(data=joy_flag)
        navdata_msg.sg_idx.data = int(self.curr_sg_idx)
        navdata_msg.xy_dist.data = np_distance_m[0]
        navdata_msg.orient_dist.data = np_distance_m[1]
        navdata_msg.is_collision.data = bool(b_is_collision)



        print("published coll: ",b_is_collision)
# waypoint
        flat_waypoints = np_waypoints_m.flatten().tolist()
        dim1 = MultiArrayDimension()
        dim1.label = 'rows'
        dim1.size = num_traj_len
        dim1.stride = num_traj_len * num_params  # x y th
        dim2 = MultiArrayDimension()
        dim2.label = 'cols'
        dim2.size = num_params
        dim2.stride = num_params

        layout = Float32MultiArray().layout
        layout.dim = [dim1, dim2]
        layout.data_offset = 0
        # Fill Float32MultiArray
        waypoint_array = Float32MultiArray()
        waypoint_array.layout = layout
        waypoint_array.data = flat_waypoints
        navdata_msg.waypoints = waypoint_array

# waypoint
        flat_pose_diff = np_pose_diff_m.flatten().tolist()
        # Fill Float32MultiArray
        navdata_msg.pose_diff  = Float32MultiArray()
        navdata_msg.pose_diff.data = flat_pose_diff
        self.navdata_pub.publish(navdata_msg)

    # def publish_waypoint(self, flag, np_waypoints, np_distance):
    #
    #     (num_traj_len, num_params) = np_waypoints.shape
    #     out_waypoints = np_waypoints.copy()
    #     for ii in range(0, num_traj_len):
    #         if self.normalize:
    #             out_waypoints[ii] = _denormalize_pose(np_waypoints[ii], waypoint_spacing= self.waypoint_spacing, dataset_name= self.dataset_name)
    #         else:
    #             out_waypoints[ii] = np_waypoints[ii]
    #
    #     waypoint_msg = waypoint_stamped()
    #     waypoint_msg.header.stamp = rospy.Time.now()
    #     waypoint_msg.joystick = Bool(data=flag)
    #     waypoint_msg.sg_idx.data = int(self.curr_sg_idx)
    #
    #     flat_waypoints = out_waypoints.flatten().tolist()
    #     dim1 = MultiArrayDimension()
    #     dim1.label = 'rows'
    #     dim1.size = num_traj_len
    #     dim1.stride = num_traj_len * num_params  # x y th
    #     dim2 = MultiArrayDimension()
    #     dim2.label = 'cols'
    #     dim2.size = num_params
    #     dim2.stride = num_params
    #
    #     layout = Float32MultiArray().layout
    #     layout.dim = [dim1, dim2]
    #     layout.data_offset = 0
    #     # Fill Float32MultiArray
    #     waypoint_array = Float32MultiArray()
    #     waypoint_array.layout = layout
    #     waypoint_array.data = flat_waypoints
    #     waypoint_msg.waypoints = waypoint_array
    #
    #     waypoint_msg.xy_dist.data = np_distance[0]
    #     waypoint_msg.orient_dist.data = np_distance[1]
    #     self.waypoint_pub.publish(waypoint_msg)

    def predict_actions(self, model: nn.Module, model_type: str, dataset_name: str):

        ts_obs_depths = torch.zeros([self.obs_queue_size, self.img_height, self.img_width])
        for idx in range(0, self.obs_queue_size):
            pil_depth = self.context_depth_queue[idx]
            ts_resized_depth = resize_and_aspect_crop(pil_depth, [self.img_width, self.img_height])  # [C, h, w]
            assert(ts_resized_depth.max() <= 1, f"obs depths not normalized well {ts_resized_depth.max()}" )
            # print("dep shape: ", ts_resized_depth[0].shape)
            # print("obs depth shape: ", ts_obs_depths[0].shape)
            ts_obs_depths[idx] = ts_resized_depth
        ts_obs_depths = ts_obs_depths[None, ...]
        ts_obs_depths = ts_obs_depths.to(device)

        # process goal depth
        #print(self.sg_depth.size)   # (64, 85)
        #print(self.sg_depth.mode)   # I;16
        ts_goal_depth = resize_and_aspect_crop(self.sg_depth, [self.img_width, self.img_height])  # [h, w]
        assert (ts_goal_depth.max() <= 1, f"goal depth not normalized well {ts_resized_depth.max()}")
        ts_goal_depth = ts_goal_depth[None, ...]
        ts_goal_depth = ts_goal_depth.to(device)

        # process odom context:
        # (1) pose correction (if necessary), (2) tform to local coord,
        # (3) normalize and (4) convert to tensor
        # correct context odom pose
        # corrected_odom_context = odom_context_correction (context_pose_queue, correcting_htm)

        np_context_pose = np.zeros([self.obs_queue_size, 3])  # x, y, th
        for ii in range(0, self.obs_queue_size):
            [xcp, ycp, th_cp] = self.context_pose_queue[ii]
            [xc, yc, thc ] = self.curr_robot_pose_corrected
            # print("%f %f %f \n" %(x0, y0, th0))
            # print("%f %f %f \n" %(xc, yc, th_c))
            rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([xc, yc, thc]), np.array([xcp, ycp, th_cp]),
                                                        self.dataset_name)

            if self.normalize:
                pose_norm = _normalize_pose(np.array([rel_x, rel_y, rel_theta]),
                                            waypoint_spacing=self.waypoint_spacing,
                                            dataset_name= self.dataset_name)
                np_context_pose[ii] = pose_norm
            else:
                np_context_pose[ii] = np.array([rel_x, rel_y, rel_theta])

        # process goal pose:
        # (2) normalize and (3) convert to tensor
        if self.normalize:
            [rel_x, rel_y, rel_theta] = self.curr_rel_sg_pose
            sg_pose = _normalize_subgoal(np.array([rel_x, rel_y, rel_theta]),
                                      max_frame_dist=self.max_frame_dist, dataset_name=dataset_name)
        else:
            sg_pose = np.array([rel_x, rel_y, rel_theta])


        if self.learn_angle:
            np_context_action = np.zeros([self.obs_queue_size, 4], dtype='float32')
            for ii in range(0, self.obs_queue_size):
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

        if self.normalize:
            np_out_waypoints = _denormalize_pose(np_waypoints, waypoint_spacing= self.waypoint_spacing,
                                             dataset_name = self.dataset_name)
            np_out_pose_diff = _denormalize_subgoal(np_pose_diff, max_frame_dist= self.max_frame_dist,
                                             dataset_name = self.dataset_name)
        else:
            np_out_waypoints = np_waypoints
            np_out_pose_diff = np_pose_diff

        return np_out_waypoints, np_out_pose_diff


    def predict_collision(self, model: nn.Module, model_type: str, dataset_name: str):

        ts_obs_depths = torch.zeros([self.obs_queue_size, self.img_height, self.img_width])
        for idx in range(0, self.obs_queue_size):
            pil_depth = self.context_depth_queue[idx]
            ts_resized_depth = resize_and_aspect_crop(pil_depth, [self.img_width, self.img_height])  # [C, h, w]
            assert(ts_resized_depth.max() <= 1, f"obs depths not normalized well {ts_resized_depth.max()}" )
            # print("dep shape: ", ts_resized_depth[0].shape)
            # print("obs depth shape: ", ts_obs_depths[0].shape)
            ts_obs_depths[idx] = ts_resized_depth
        ts_obs_depths = ts_obs_depths[None, ...]
        ts_obs_depths = ts_obs_depths.to(device)

        ts_obs_depth_curr = ts_obs_depths[:, -1:, :, :]
        # process goal depth
        #print(self.sg_depth.size)   # (64, 85)
        #print(self.sg_depth.mode)   # I;16
        #ts_goal_depth = resize_and_aspect_crop(self.sg_depth, [self.img_width, self.img_height])  # [h, w]
        # assert (ts_goal_depth.max() <= 1, f"goal depth not normalized well {ts_resized_depth.max()}")
        # ts_goal_depth = ts_goal_depth[None, ...]
        # ts_goal_depth = ts_goal_depth.to(device)

        # process odom context:
        # (1) pose correction (if necessary), (2) tform to local coord,
        # (3) normalize and (4) convert to tensor
        # correct context odom pose
        # corrected_odom_context = odom_context_correction (context_pose_queue, correcting_htm)

        if model_type == 'depth_coll':
            model_outputs = model(ts_obs_depth_curr)
        else:
            raise ValueError("unknown model type")

        print("model output shape: ", model_outputs.shape)
        ts_coll_logit = model_outputs  # (1,5,4),  (1,2)
        coll_logit = ts_coll_logit.item()

        return coll_logit

    def make_correction_to_subgoals(self, np_pose_diff):
        # make correction to topomap and the curr target sg

        nsg_x, nsg_y, nsg_qw, nsg_qz = np_pose_diff
        #dth = 2.0 * np.arctan2(qz, qw)
        osg_x, osg_y = self.curr_rel_sg_pose[0:2]
        dx = nsg_x - osg_x
        dy = nsg_y - osg_y
        # update topomap odom
        # for idx in range( len(self.topomap_odom) ):
        #     xsg, ysg, thsg = self.topomap_odom[idx]
        #     #th = np.arctan2(np.sin(th_rel + dth), np.cos(th_rel + dth))
        #     self.topomap_odom[idx] = np.array([xsg + dx, ysg + dy, thsg], dtype=float)

        # update curr sg
        x_sg, y_sg, th_sg   = self.topomap_odom[self.curr_sg_idx]
        xc, yc, thc = self.curr_robot_pose_corrected
        rel_x, rel_y, rel_theta = _get_rel_pose_se2(np.array([xc, yc, thc]),
                                                    np.array([x_sg + dx, y_sg + dy, th_sg]),
                                                    self.dataset_name)
        self.curr_rel_sg_pose = np.array([rel_x, rel_y, rel_theta])

    def navstep(self, nav_model, nav_model_type,
                                           col_model, col_model_type,
                                           dataset_name):

        np_waypoints_m, np_pose_diff_m = self.predict_actions(nav_model, nav_model_type, dataset_name)

        if self._correction_latch is False:  # This prevents subgoal pose correction
            coll_logit = self.predict_collision(col_model, col_model_type, dataset_name)
            coll_prob = 1.0 / (1.0 + np.exp(-coll_logit))
            if coll_prob > 0.5:  # meaning the robot is about to collide..
                self.set_collision_status(True)
            else:
                # ordinary case. just follow the curr sg
                self.set_collision_status(False)

            if self.get_collision_status():  # meaning the robot is about to collide..
                # need to update sub goal pose and topo map poses accordingly
                self._correction_latch = True
                self.make_correction_to_subgoals(np_pose_diff_m)

        return np_waypoints_m, np_pose_diff_m

    def navstep_wo_collsion_detection(self, nav_model, nav_model_type,
                                           col_model, col_model_type,
                                            dataset_name):

        np_waypoints_m, np_pose_diff_m = self.predict_actions(nav_model, nav_model_type, dataset_name)
        return np_waypoints_m, np_pose_diff_m