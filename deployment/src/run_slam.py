#!/usr/bin/env python3
"""
Launch move_former_sync_slam.launch from Python using roslaunch API,
passing args like: map_file_name:=<map_base>

Usage:
  python launch_slam.py --pkg_dir /path/to/catkin_ws/src/your_pkg \
                        --map_base /path/to/map.yaml

Notes:
- This runs roslaunch inside this Python process.
- Stop with Ctrl+C; it will shutdown the launch cleanly.
"""

import argparse
import os
import signal
import sys
import time

import roslaunch
import rospy
import rospkg
from sensor_msgs.msg import Joy
from pathlib import Path
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf.transformations as tft
import numpy as np
import subprocess

# CHANGE THESE to your controller mapping (check /joy)
BTN_L1 = 4
BTN_R1 = 5
BTN_L2 = 6  # sometimes L2 is an axis; see below

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEPL_CONFIG_PATH = (
    BASE_DIR / "config" / "depth_nav.yaml"
)
#print("config path: %s"%DEPL_CONFIG_PATH)

with open(DEPL_CONFIG_PATH, "r") as f:
    deployment_config = yaml.safe_load(f)

def resume_posegraph(map_base: str, timeout_s: float = 30.0):
    """
    Resume slam_toolbox from a serialized posegraph.
    Expects: <map_base>.posegraph
    """
    posegraph = map_base + ".posegraph"
    if not os.path.isfile(posegraph):
        rospy.logwarn(f"[run_slam] posegraph not found, skipping deserialize: {posegraph}")
        return False

    rospy.loginfo("[run_slam] Waiting for /slam_toolbox/deserialize_map ...")
    rospy.wait_for_service("/slam_toolbox/deserialize_map", timeout=timeout_s)

    cmd = f"rosservice call /slam_toolbox/deserialize_map \"filename: '{posegraph}'\""
    rospy.loginfo(f"[run_slam] {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        rospy.logerr(f"[run_slam] deserialize_map failed (exit={ret})")
        return False

    rospy.loginfo("[run_slam] deserialize_map success")
    return True

def publish_initialpose(x, y, yaw, frame_id="map"):
    pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    q = tft.quaternion_from_euler(0.0, 0.0, yaw)
    msg.pose.pose.orientation.x = q[0]
    msg.pose.pose.orientation.y = q[1]
    msg.pose.pose.orientation.z = q[2]
    msg.pose.pose.orientation.w = q[3]

    # modest covariance
    msg.pose.covariance[0] = 0.05**2
    msg.pose.covariance[7] = 0.05**2
    msg.pose.covariance[35] = 0.10**2

    for _ in range(3):
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rospy.sleep(0.1)

def load_start_pose_from_slam_poses(map_base: str):
    """
    map_base: .../<topomap_name>/map   (NO extension)
    expects:  .../<topomap_name>/slam_poses.txt
    returns: (x, y, yaw)
    """
    topomap_dir = os.path.dirname(map_base)   # folder containing map + slam_poses.txt
    pose_file = os.path.join(topomap_dir, "slam_poses.txt")
    if not os.path.isfile(pose_file):
        raise FileNotFoundError(f"Missing slam_poses.txt: {pose_file}")

    poses = np.loadtxt(pose_file)
    poses = np.atleast_2d(poses)

    # Your format: [node_id, x, y, yaw]  (based on your other script using poses[0][1:])
    x, y, yaw = poses[0][1], poses[0][2], poses[0][3]
    return float(x), float(y), float(yaw)

def launch_slam(pkg_dir: str, map_base: str):
    launch_path = os.path.join(pkg_dir, "launch", "includes", "move_former_sync_slam.launch")
    if not os.path.isfile(launch_path):
        raise FileNotFoundError(f"Launch file not found: {launch_path}")

    # cli_arg1 = [
    #     launch_path,
    #     f"map_file_name:={map_base}",  # NOTE: := not =
    # ]
    x0, y0, yaw0 = load_start_pose_from_slam_poses(map_base)
    cli_arg1 = [
        launch_path,
        f"map_file_name:={map_base}",
        f"start_x:={x0}",
        f"start_y:={y0}",
        f"start_yaw:={yaw0}",
    ]

    resolved = roslaunch.rlutil.resolve_launch_arguments(cli_arg1)[0]
    roslaunch1 = [(resolved, cli_arg1[1:])]

    # uuid must be created BEFORE ROSLaunchParent
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch1)

    # ----------------------------
    # Stop conditions
    # ----------------------------
    shutdown_requested = {"flag": False}

    def joy_cb(msg: Joy):
        l1 = msg.buttons[BTN_L1] if BTN_L1 < len(msg.buttons) else 0
        r1 = msg.buttons[BTN_R1] if BTN_R1 < len(msg.buttons) else 0
        l2_btn = msg.buttons[BTN_L2] if BTN_L2 < len(msg.buttons) else 0

        if l1 and r1 and l2_btn:
            rospy.logwarn("[run_slam] L1 + R1 + L2 pressed -> stopping SLAM")
            shutdown_requested["flag"] = True

    joy_topic = rospy.get_param("~joy_topic", "/joy")
    joy_sub = rospy.Subscriber(joy_topic, Joy, joy_cb, queue_size=1)

    def shutdown_roslaunch(reason: str):
        if shutdown_requested["flag"]:
            return
        shutdown_requested["flag"] = True
        print(f"\n[run_slam] Shutdown requested ({reason}).")

    # Ctrl+C / kill handling
    def _sig_handler(signum, _frame):
        shutdown_roslaunch(f"signal {signum}")
    signal.signal(signal.SIGINT, _sig_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, _sig_handler)  # kill <pid>

    # Start roslaunch
    parent.start()
    print(f"[run_slam] Started: {resolved}")
    print(f"[run_slam] Args: {cli_arg1[1:]}")
    print("[run_slam] Stop: Ctrl+C OR joystick combo (L1+R1+L2)")

    # ---- Load previous map graph + set start pose from slam_poses.txt
    print("here 0")
    try:
        time.sleep(1.0)  # give slam_toolbox a moment to advertise services

        # 1) load the saved posegraph (required)
        #resume_posegraph(map_base)

        # 2) set initial pose to the REAL start pose of this map
        # x0, y0, yaw0 = load_start_pose_from_slam_poses(map_base)
        # print("here \n")
        # print(f"[run_slam] initialpose from slam_poses.txt: {x0:.3f}, {y0:.3f}, {yaw0:.3f}")
        # publish_initialpose(x0, y0, yaw0)

    except Exception as e:
        print(f"[run_slam] resume/initpose exception: {e}")

    try:
        while not rospy.is_shutdown() and not shutdown_requested["flag"]:
            time.sleep(0.1)
    except KeyboardInterrupt:
        shutdown_roslaunch("KeyboardInterrupt")
    finally:
        print("[run_slam] Shutting down roslaunch...")
        try:
            parent.shutdown()
        except Exception as e:
            print(f"[run_slam] parent.shutdown() error: {e}")
        # give children a moment to exit cleanly
        time.sleep(0.2)

def main():
    rospy.init_node("run_slam", anonymous=True, disable_signals=True)

    topomap_path = f"{BASE_DIR}/deployment/topomaps"
    topomap_name = deployment_config["deployment"]["topomap_name"]
    map_base = "%s/%s/map"%(topomap_path, topomap_name)
    print(map_base)

    pkg_path = rospkg.RosPack().get_path("navdata_collector")
    launch_slam(pkg_path, map_base)

if __name__ == "__main__":
    main()

