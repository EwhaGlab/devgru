import os
import csv
import yaml
import rospy
import numpy as np
import cv2
from datetime import datetime
from cv_bridge import CvBridge
import tf2_ros

# Standard messages inside your custom msg
from std_msgs.msg import Bool, Int32, Float32, Float32MultiArray
from sensor_msgs.msg import Image
from navdata_collector.msg import navstep_stamped
from nav_msgs.msg import Odometry

import torch

from pathlib import Path
import sys
from os.path import dirname, abspath
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(BASE_DIR)

print(BASE_DIR)
import utils.rigid_motion as rm

RATE = 20 #robot_config["frame_rate"]

from topic_names import (RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO,
                        WAYPOINT_TOPIC,
                        ODOM_TOPIC, REACHED_GOAL_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)

DEPL_CONFIG_PATH = f"{BASE_DIR}/config/depth_nav.yaml"
with open(DEPL_CONFIG_PATH, "r") as f:
    depth_nav_config = yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def stamp_to_float(stamp: rospy.Time) -> float:
    return stamp.secs + stamp.nsecs * 1e-9

def reshape_float32_multiarray(msg: Float32MultiArray) -> np.ndarray:
    """
    Convert std_msgs/Float32MultiArray to numpy array, respecting layout if present.
    If layout is empty/flat, returns 1D array.
    """
    data = np.array(msg.data, dtype=np.float32)

    # If layout provides dimensions, reshape accordingly
    if msg.layout and msg.layout.dim:
        dims = [d.size for d in msg.layout.dim if d.size > 0]
        if dims and int(np.prod(dims)) == data.size:
            return data.reshape(dims)
    return data


class NavstepDumperNode:
    def __init__(self):
        self.topic = "/navstep"
        self.odom_topic = ODOM_TOPIC
        self.base_out_dir = depth_nav_config["deployment"]["nav_log_dir"]
        self.write_png = True
        self.png_compression = 3
        self.flush_every = 50
        # numbering
        self.idx = 0
        self.sg_idx = 0
        self.num_tot_nodes = 0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tf_map_frame = rospy.get_param("~tf_map_frame", "map")
        self.tf_odom_frame = rospy.get_param("~tf_odom_frame", "odom")
        self.tf_base_frame = rospy.get_param("~tf_base_frame", "base_link")
        self.tf_timeout = rospy.Duration(0.2)
        self.odom_pose = None
        self.map_to_odom = None
        self.map_to_base = None
        self._navstep_time = None
        self._is_goal_reached = False
        self.nav_begin_time = rospy.Time.now()

        rostime = rospy.Time.now()
        sec = rostime.secs + rostime.nsecs * 1e-9
        dt = datetime.fromtimestamp(sec)
        timestr = dt.strftime("%Y-%m-%d-%H-%M")
        # dirs
        self.out_dir = "%s/devgru_res/%s"%(self.base_out_dir, timestr)
        if os.path.exists(self.out_dir):
            raise FileExistsError(f"Directory already exists: {self.out_dir}")
        else:
            os.makedirs(self.out_dir)

        self.rgb_dir = os.path.join(self.out_dir, "rgb")
        self.depth_dir = os.path.join(self.out_dir, "depth")
        self.traj_dir = os.path.join(self.out_dir, "traj")
        self.pose_diff_dir = self.traj_dir #os.path.join(self.out_dir, "pose_diff")
        self.waypoints_dir = self.traj_dir #os.path.join(self.out_dir, "waypoints")

        for d in [self.out_dir, self.rgb_dir, self.depth_dir, self.pose_diff_dir, self.traj_dir]:
            ensure_dir(d)

        self.index_csv_path = os.path.join(self.out_dir, "index.csv")
        self.index_csv_file = open(self.index_csv_path, "w", newline="")
        self.index_writer = csv.writer(self.index_csv_file)
        self.index_writer.writerow([
            "idx", "stamp_sec", "frame_id",
            "sg_idx", "xy_dist", "orient_dist",
            "joystick", "is_collision",
            "rgb_file", "depth_file", "pose_diff_file", "waypoints_file", "m2b_file"
        ])

        self.bridge = CvBridge()
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)
        self.sub_navstep = rospy.Subscriber(self.topic, navstep_stamped, self.navstep_cb, queue_size=10)
        self.sub_reached = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, self.reached_goal_cb, queue_size=1)

        rospy.loginfo(f"[navstep_dumper] Subscribing to {self.topic}")
        rospy.loginfo(f"[navstep_dumper] Writing to {self.out_dir}")

    def __del__(self):
        # measure tot nav time
        nav_end_time = rospy.Time.now()
        nav_elapsed = nav_end_time - self.nav_begin_time
        nav_elapsed_sec = nav_elapsed.to_sec()

        # cnt the total num topomap
        topomap_name = depth_nav_config["deployment"]["topomap_name"]
        topomap_path = f"{BASE_DIR}/deployment/topomaps/{topomap_name}"
        topo_tf_m2b_file = "%s/topo_tf_m2b.txt" % topomap_path
        topo_tf_m2b = np.loadtxt(topo_tf_m2b_file)
        num_nodes, _ = topo_tf_m2b.shape

        x0, y0 = topo_tf_m2b[0][4:6]
        cov_dist = 0
        for ii in range(1, self.sg_idx):
            m2b = topo_tf_m2b[ii]
            x1 = m2b[4]
            y1 = m2b[5]
            d  = ( (x1-x0)**2 + (y1-y0)**2 ) ** 0.5
            cov_dist += d
            x0 = x1
            y0 = y1

        nav_summary_path = self.out_dir + "/nav_summary.txt"
        with open(nav_summary_path, 'w') as file:
            file.write("devgru summary on %s \n"%topomap_name)
            file.write(f"final goal reached: {self._is_goal_reached}\n")
            file.write("covered node idx / last node idx : %d / %d \n"%( max(self.sg_idx-1,0), num_nodes-1))
            file.write("covered dist: %f (m) \n"%cov_dist)
            file.write("tot nav time: %.3f (s) \n"%nav_elapsed_sec)

    def lookup_map_to_odom(self, stamp=None):
        """
        Returns geometry_msgs/TransformStamped or None
        """
        try:
            if stamp is None:
                tf = self.tf_buffer.lookup_transform(
                    self.tf_map_frame,
                    self.tf_odom_frame,
                    rospy.Time(0),
                    self.tf_timeout
                )
            else:
                tf = self.tf_buffer.lookup_transform(
                    self.tf_map_frame,
                    self.tf_odom_frame,
                    stamp,
                    self.tf_timeout
                )
            return tf

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(
                2.0, f"[NavstepDumperNode] TF lookup map to odom failed: {e}"
            )
            return None

    def lookup_map_to_base(self, stamp=None):
        """
        Returns geometry_msgs/TransformStamped or None
        """
        try:
            if stamp is None:
                tf = self.tf_buffer.lookup_transform(
                    self.tf_map_frame,
                    self.tf_base_frame,
                    rospy.Time(0),
                    self.tf_timeout
                )
            else:
                tf = self.tf_buffer.lookup_transform(
                    self.tf_map_frame,
                    self.tf_base_frame,
                    stamp,
                    self.tf_timeout
                )
            return tf

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(
                2.0, f"[NavstepDumperNode] TF lookup failed: {e}"
            )
            return None

    def _increment_index(self) -> str:
        self.idx += 1

    def odom_cb(self, msg: Odometry):
        import tf.transformations as tft
        quat = msg.pose.pose.orientation
        yaw = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom_pose = np.asarray([x,y,yaw])

        tf = self.lookup_map_to_odom(msg.header.stamp)
        if tf is not None:
            t = tf.transform.translation
            q = tf.transform.rotation
            # Example: save map->odom pose (x, y, yaw)
            #import tf.transformations as tft
            yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.map_to_odom = np.asarray([t.x, t.y, yaw])

        tf = self.lookup_map_to_base(msg.header.stamp)
        if tf is not None:
            t = tf.transform.translation
            q = tf.transform.rotation
            # Example: save map->odom pose (x, y, yaw)
            #import tf.transformations as tft
            yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.map_to_base = np.asarray([t.x, t.y, yaw])

    def navstep_cb(self, msg: navstep_stamped):
        if self._is_goal_reached:
            self.sg_idx = self.num_tot_nodes
            return

        cb_start_time = rospy.Time.now()
        stamp_sec = stamp_to_float(msg.header.stamp)
        self._navstep_time = stamp_sec
        frame_id = msg.header.frame_id
        frame_idx = '%05d'%self.idx

        self.sg_idx = int(msg.sg_idx.data)
        # ---------- RGB ----------
        rgb_path = os.path.join(self.rgb_dir, f"rgb{frame_idx}.png")
        try:
            # Keep whatever encoding you receive; convert to bgr8 for OpenCV PNG
            # Typical encodings: "rgb8" or "bgr8"
            rgb_cv = self.bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
            cv2.imwrite(rgb_path, rgb_cv, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])
        except Exception as e:
            rospy.logwarn(f"[navstep_dumper] Failed to save RGB at idx={frame_idx}: {e}")
            rgb_path = ""

        # ---------- DEPTH ----------
        depth_path = os.path.join(self.depth_dir, f"depth{frame_idx}.png")
        try:
            # Preserve depth values: use "passthrough"
            # For 16UC1, this returns uint16 numpy array.
            depth_cv = self.bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")

            # If depth is float (32FC1), PNG isn't great; save .npy as well.
            if depth_cv.dtype == np.float32 or depth_cv.dtype == np.float64:
                npy_path = os.path.join(self.depth_dir, f"depth{frame_idx}.npy")
                np.save(npy_path, depth_cv)
                # Optional: also write a visualization PNG (scaled) if you want
                # Here we still attempt PNG by converting to uint16 if safe
                depth_vis = np.clip(depth_cv, 0, 65535).astype(np.uint16)
                cv2.imwrite(depth_path, depth_vis, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])
            else:
                # uint16 (16UC1) -> write directly
                cv2.imwrite(depth_path, depth_cv, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])
        except Exception as e:
            rospy.logwarn(f"[navstep_dumper] Failed to save depth at idx={frame_idx}: {e}")
            depth_path = ""

        # ---------- pose_diff ----------
        pose_diff_path = os.path.join(self.pose_diff_dir, f"rel_sg_pose_m_{frame_idx}.txt")
        try:
            pose_diff = reshape_float32_multiarray(msg.pose_diff)
            # expected shape: (3,) [dx, dy, dth], but we don't assume
            np.savetxt(pose_diff_path, pose_diff.reshape(-1), fmt="%.8f")
        except Exception as e:
            rospy.logwarn(f"[navstep_dumper] Failed to save pose_diff at idx={frame_idx}: {e}")
            pose_diff_path = ""

        # ---------- waypoints ----------
        waypoints_path = os.path.join(self.waypoints_dir, f"waypoints_m_{frame_idx}.txt")
        try:
            waypoints = reshape_float32_multiarray(msg.waypoints)
            # If it's Nx2 or Nx3 it'll save as a matrix; otherwise 1D
            np.savetxt(waypoints_path, waypoints, fmt="%.8f")
        except Exception as e:
            rospy.logwarn(f"[navstep_dumper] Failed to save waypoints at idx={frame_idx}: {e}")
            waypoints_path = ""

        # ---------- map 2 odom tform ----
        map_to_odom_path = os.path.join(self.traj_dir, f"m2o_{frame_idx}.txt")
        map_to_base_path = os.path.join(self.traj_dir, f"m2b_{frame_idx}.txt")
        odom_pose_path   = os.path.join(self.traj_dir, f"odom_pose_{frame_idx}.txt")

        # print(self.map_to_odom.shape)
        # print(self.map_to_base.shape)
        # print(self.odom_pose.shape)
        try:
            np.savetxt(map_to_odom_path, self.map_to_odom, fmt="%.8f")
            np.savetxt(map_to_base_path, self.map_to_base, fmt="%.8f")
            np.savetxt(odom_pose_path, self.odom_pose, fmt="%.8f")
        except Exception as e:
            rospy.logwarn(f"[navstep_dumper] Failed to save map 2 odom at idx={frame_idx}: {e}")

        # ---------- meta (yaml) ----------
        # meta_path = os.path.join(self.meta_dir, f"meta{frame_idx}.yaml")
        # try:
        #     meta = {
        #         "idx": int(self.idx),
        #         "stamp_sec": float(stamp_sec),
        #         "frame_id": str(frame_id),
        #
        #         "joystick": bool(msg.joystick.data) if isinstance(msg.joystick, Bool) else bool(msg.joystick),
        #         "is_collision": bool(msg.is_collision.data) if isinstance(msg.is_collision, Bool) else bool(msg.is_collision),
        #
        #         "sg_idx": int(msg.sg_idx.data) if isinstance(msg.sg_idx, Int32) else int(msg.sg_idx),
        #         "xy_dist": float(msg.xy_dist.data) if isinstance(msg.xy_dist, Float32) else float(msg.xy_dist),
        #         "orient_dist": float(msg.orient_dist.data) if isinstance(msg.orient_dist, Float32) else float(msg.orient_dist),
        #
        #         "rgb_encoding": getattr(msg.rgb, "encoding", ""),
        #         "depth_encoding": getattr(msg.depth, "encoding", ""),
        #         "rgb_size": [int(msg.rgb.height), int(msg.rgb.width)],
        #         "depth_size": [int(msg.depth.height), int(msg.depth.width)],
        #     }
        #     with open(meta_path, "w") as f:
        #         yaml.safe_dump(meta, f, sort_keys=False)
        # except Exception as e:
        #     rospy.logwarn(f"[navstep_dumper] Failed to write meta yaml at idx={frame_idx}: {e}")
        #     meta_path = ""

        # ---------- index.csv ----------
        self.index_writer.writerow([
            frame_idx, f"{stamp_sec:.9f}", frame_id,
            int(msg.sg_idx.data) if hasattr(msg.sg_idx, "data") else int(msg.sg_idx),
            float(msg.xy_dist.data) if hasattr(msg.xy_dist, "data") else float(msg.xy_dist),
            float(msg.orient_dist.data) if hasattr(msg.orient_dist, "data") else float(msg.orient_dist),
            int(msg.joystick.data) if hasattr(msg.joystick, "data") else int(bool(msg.joystick)),
            int(msg.is_collision.data) if hasattr(msg.is_collision, "data") else int(bool(msg.is_collision)),
            rgb_path, depth_path, pose_diff_path, waypoints_path #, meta_path
        ])

        if (self.idx % self.flush_every) == 0:
            self.index_csv_file.flush()

        self._increment_index()
        cb_end_time = rospy.Time.now()
        dt = (cb_end_time - cb_start_time).to_sec()
        print("navstep CB elapsed time [s]:", dt)

    def reached_goal_cb(self, msg: Bool):
        self._is_goal_reached = msg.data

    def shutdown(self):
        try:
            self.index_csv_file.flush()
            self.index_csv_file.close()
        except Exception:
            pass

def main():
    rospy.init_node("navstep_dumper", anonymous=False)
    node = NavstepDumperNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
