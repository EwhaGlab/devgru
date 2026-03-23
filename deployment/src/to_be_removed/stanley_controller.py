import numpy as np
import yaml
from typing import Tuple

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC)
from ros_data import ROSData
from nav_utils import clip_angle
import utils.rigid_motion as rm

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1/robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None
np_waypoints = []

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      
def stanley_controller(waypoints):
    """
    Stanley controller for differential drive robot.
    Args:
        waypoints: Nx2 numpy array [[x, y], ...]
        robot_pose: (x, y, theta)
    Returns:
        (v, w)
    """
    
    v_ref = 0.3
    k = 1.0
    wheelbase=0.4
    
    x_r, y_r, theta_r = [0,0,0]

    waypoints = waypoints[:,:2]
    dists = ( (waypoints[:,0]-x_r)**2 + (waypoints[:,1]-y_r)**2 )**0.5   #np.linalg.norm(waypoints, axis=1)
    nearest_idx = np.argmin(dists)

    if nearest_idx < len(waypoints) - 1:
        wp1 = waypoints[nearest_idx]
        wp2 = waypoints[nearest_idx + 1]
    else:
        wp1 = waypoints[-2]
        wp2 = waypoints[-1]

    path_dx = wp2[0] - wp1[0]
    path_dy = wp2[1] - wp1[1]
    path_yaw = np.arctan2(path_dy, path_dx)

    # heading error
    heading_error = rm.normalizeAngle(path_yaw - theta_r)

    # cross track error
    path_normal = np.array([-np.sin(path_yaw), np.cos(path_yaw)])
    cross_track_vec = np.array([x_r - wp1[0], y_r - wp1[1]])
    cross_track_error = np.dot(cross_track_vec, path_normal)

    correction = np.arctan2(k * cross_track_error, v_ref)
    steering = heading_error + correction

    w = (v_ref / wheelbase) * np.tan(steering)
    v = v_ref

    return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	global np_waypoints
	print("seting waypoint")
	data_raw = np.array(waypoint_msg.data, dtype=np.float32)
	dims = waypoint_msg.layout.dim

	rows = dims[0].size
	cols = dims[1].size
	np_waypoints = data_raw.reshape((rows, cols))

	waypoint.set(waypoint_msg.data)
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	reached_goal = reached_goal_msg.data


def main():
	global vel_msg, reverse_mode
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	print("Registered with master node. Waiting for waypoints...")
	while not rospy.is_shutdown():
		vel_msg = Twist()
		if reached_goal:
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			return
		elif waypoint.is_valid(verbose=True):
			v, w = stanley_controller( np_waypoints ) #waypoint.get())
			if reverse_mode:
				v *= -1
			vel_msg.linear.x = v
			vel_msg.angular.z = w
			print(f"publishing new vel: {v}, {w}")
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
