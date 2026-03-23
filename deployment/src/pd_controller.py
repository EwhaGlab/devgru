import numpy as np
import yaml
from typing import Tuple

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (NAVDATA_TOPIC, #WAYPOINT_TOPIC,
			 			REACHED_GOAL_TOPIC)
from ros_data import ROSData
from nav_utils import clip_angle
from navdata_collector.msg import navdata_stamped #waypoint_stamped

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
np_waypoints = []		# waypoints (in meters)
finish_flag = None
joy_flag = False

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      
#def clip_angle(theta: float) -> float:
    #return np.arctan2(np.sin(theta), np.cos(theta))


def p_controller(waypoints: np.ndarray) -> Tuple[float]:
	"""PD controller for the robot"""
	waypoint = waypoints[2,:2]
	#print(waypoint)
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/DT		
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*DT)
	else:
		v = dx / DT
		w = np.arctan(dy/dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


def pd_controller(
    waypoints: np.ndarray,
    state: dict,
    config: dict
):
    """
    PD Controller with minimal arguments.
    waypoint: (2,) [dx, dy]
    state: dict holding 'prev_dx' and 'prev_heading'
    config: dict holding constants and gains
    """
    waypoint = waypoints[2,:2] #waypoints[2,:2]
    #print('target waypoint: (%f %f)'%(waypoint[0], waypoint[1]))
    dx, dy = waypoint
    heading_error = clip_angle(np.arctan2(dy, dx))

    # Derivatives
    if state.get("prev_dx") is None:
        d_dx = 0.0
    else:
        d_dx = (dx - state["prev_dx"]) / config["DT"]

    if state.get("prev_heading") is None:
        d_heading = 0.0
    else:
        d_heading = clip_angle(heading_error - state["prev_heading"]) / config["DT"]

    # Linear velocity
    if np.abs(dx) < config["EPS"]:
        v = 0.0
    else:
        v = config["Kp_linear"] * dx + config["Kd_linear"] * d_dx

    # Angular velocity
    if np.abs(dx) < config["EPS"] and np.abs(dy) < config["EPS"]:
        v = 0.0
        w = 0.0
    else:
        w = config["Kp_heading"] * heading_error + config["Kd_heading"] * d_heading

    # Clip
    v = np.clip(v, 0.0, config["MAX_V"])
    w = np.clip(w, -config["MAX_W"], config["MAX_W"])

    # Update state
    state["prev_dx"] = dx
    state["prev_heading"] = heading_error

    return v, w

def callback_drive(navdata_msg: navdata_stamped): #Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	global np_waypoints
	global joy_flag
	joy_flag = navdata_msg.joystick.data
	#print("seting  <%s> waypoint"%('incorrect' if joy is True else 'correct') )
	data_raw = np.array(navdata_msg.waypoints.data, dtype=np.float32)
	dims = navdata_msg.waypoints.layout.dim

	rows = dims[0].size
	cols = dims[1].size
	np_waypoints = data_raw.reshape((rows, cols))
	waypoint.set(navdata_msg.waypoints.data)


# def callback_drive(waypoint_msg: waypoint_stamped): #Float32MultiArray):
# 	"""Callback function for the waypoint subscriber"""
# 	global vel_msg
# 	global np_waypoints
# 	global joy_flag
# 	joy_flag = waypoint_msg.joystick.data
# 	#print("seting  <%s> waypoint"%('incorrect' if joy is True else 'correct') )
# 	data_raw = np.array(waypoint_msg.waypoints.data, dtype=np.float32)
# 	dims = waypoint_msg.waypoints.layout.dim
# 	rows = dims[0].size
# 	cols = dims[1].size
# 	np_waypoints = data_raw.reshape((rows, cols))
# 	waypoint.set(waypoint_msg.waypoints.data)
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	reached_goal = reached_goal_msg.data

def finish_callback(msg):
    global finish_flag
    finish_flag = msg.data


def main():
	global vel_msg, reverse_mode, joy_flag
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	#waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	
	process_finish_sub = rospy.Subscriber('/finish_data_collection', Bool, finish_callback)
	navdata_sub = rospy.Subscriber(NAVDATA_TOPIC, navdata_stamped, callback_drive, queue_size=1)
    
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	print("Registered with master node. Waiting for waypoints...")
	

	# Config dictionary holding all constants
	config = {"DT": 0.333, "EPS":1e-6, "MAX_V": 0.3,
              "MAX_W":0.6, "Kp_linear":10.0, "Kd_linear":0.4,
              "Kp_heading":1.0, "Kd_heading":0.4}


    # State to hold previous errors
	state = {"prev_dx": None, "prev_heading": None}
	
	while not rospy.is_shutdown():
		vel_msg = Twist()
		if reached_goal:
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			break
			
		elif finish_flag is True:	
			rospy.logerr("Got finishing cmd. Shutting down..")
			break
            
		elif waypoint.is_valid(verbose=True):
			v, w = pd_controller( np_waypoints, state, config ) #waypoint.get())
			#print(np_waypoints)
			#v, w = p_controller( np_waypoints )
			if reverse_mode:
				v *= -1
			vel_msg.linear.x = v 
			vel_msg.angular.z = w 
			print(f"publishing new vel: {v*100: .3f} (cm/s), {w: .3f} (rad/s). waypoint is < %s >"%('colliding !!' if joy_flag is True else 'collision-free' )  )
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
