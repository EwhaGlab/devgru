#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import Optional
import rospy
from sensor_msgs.msg import Joy

L1_BTN = 4
R1_BTN = 5

def tmux(*args, check=True):
    subprocess.run(["tmux", *args], check=check)


def pane_send(pane: str, line: str):
    tmux("send-keys", "-t", pane, line, "Enter")


def pane_ctrl_c(pane: str):
    # Send Ctrl+C to the pane (stops the foreground program)
    tmux("send-keys", "-t", pane, "C-c", check=False)

def wait_for_joy_start():
    rospy.loginfo("Waiting for joystick L1 + R1 to start...")

    rate = rospy.Rate(50)
    last_msg = None

    def joy_cb(msg):
        nonlocal last_msg
        last_msg = msg

    rospy.Subscriber("/joy", Joy, joy_cb)

    while not rospy.is_shutdown():
        if last_msg is None:
            rate.sleep()
            continue

        l1 = last_msg.buttons[L1_BTN]
        r1 = last_msg.buttons[R1_BTN]

        if l1 and r1:
            rospy.loginfo("L1 + R1 detected. Starting navigation.")
            return

        rate.sleep()

class JoyStopWatcher:
    """Sets stop_event when L1 + R1 + L2 are pressed together."""
    def __init__(
        self,
        stop_event: threading.Event,
        joy_topic: str,
        btn_l1: int,
        btn_r1: int,
        btn_l2: int,
        axis_l2: int,
        axis_l2_pressed_thr: float,
        hold_sec: float = 0.0,
    ):
        self.stop_event = stop_event
        self.btn_l1 = btn_l1
        self.btn_r1 = btn_r1
        self.btn_l2 = btn_l2
        self.axis_l2 = axis_l2
        self.axis_l2_pressed_thr = axis_l2_pressed_thr
        self.hold_sec = hold_sec
        self._t0 = None
        self._fired = False

        self._sub = rospy.Subscriber(joy_topic, Joy, self._cb, queue_size=1)

    @staticmethod
    def _btn(msg: Joy, idx: int) -> int:
        return msg.buttons[idx] if 0 <= idx < len(msg.buttons) else 0

    @staticmethod
    def _axis(msg: Joy, idx: int) -> float:
        return msg.axes[idx] if 0 <= idx < len(msg.axes) else 0.0

    def _cb(self, msg: Joy):
        if self._fired:
            return

        l1 = self._btn(msg, self.btn_l1)
        r1 = self._btn(msg, self.btn_r1)
        l2_btn = self._btn(msg, self.btn_l2)
        l2_axis_pressed = (self._axis(msg, self.axis_l2) < self.axis_l2_pressed_thr)

        combo = bool(l1 and r1 and (l2_btn or l2_axis_pressed))
        if not combo:
            self._t0 = None
            return

        if self.hold_sec <= 0.0:
            rospy.logwarn("[tmux_launcher] L1+R1+L2 detected -> stopping programs (Ctrl+C to panes)")
            self._fired = True
            self.stop_event.set()
            return

        now = time.time()
        if self._t0 is None:
            self._t0 = now
            return
        if (now - self._t0) >= self.hold_sec:
            rospy.logwarn("[tmux_launcher] L1+R1+L2 held -> stopping programs (Ctrl+C to panes)")
            self._fired = True
            self.stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="navigate_viz_stack")
    parser.add_argument("--catkin_setup", default="~/catkin_ws/install/setup.bash")
    parser.add_argument("--bashrc", default="~/.bashrc")

    # SLAM command (kept as run_slam.py like before)
    #parser.add_argument("--map_base", default=True)
    parser.add_argument("--run_slam_py", default="run_slam.py")
    parser.add_argument("--navigate_viz_py", default="navigate_viz.py")
    parser.add_argument("--recorder_py", default="record_navsteps.py")
    parser.add_argument("--pd_py", default="pd_controller.py")

    parser.add_argument("--conda_navdata", default="navdata")
    parser.add_argument("--conda_devgru_deployment", default="devgru_deployment")
    parser.add_argument("--conda_pd", default="devgru_deployment")
    parser.add_argument("--no_conda", action="store_true")

    # Joy stop config
    parser.add_argument("--joy_topic", default="/joy")
    parser.add_argument("--btn_l1", type=int, default=4)
    parser.add_argument("--btn_r1", type=int, default=5)
    parser.add_argument("--btn_l2", type=int, default=6)
    parser.add_argument("--axis_l2", type=int, default=2)
    parser.add_argument("--axis_l2_thr", type=float, default=-0.9)
    parser.add_argument("--hold_sec", type=float, default=0.0)

    args = parser.parse_args()

    session = args.session
    workdir = os.getcwd()  #os.path.abspath(os.path.expanduser(args.workdir))
    bashrc = os.path.abspath(os.path.expanduser(args.bashrc))
    catkin_setup = os.path.abspath(os.path.expanduser(args.catkin_setup))

    def abspath_in_workdir(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(workdir, p)

    navigate_viz_py = abspath_in_workdir(args.navigate_viz_py)
    recorder_py = abspath_in_workdir(args.recorder_py)
    pd_py = abspath_in_workdir(args.pd_py)
    run_slam_py = abspath_in_workdir(args.run_slam_py)

    rospy.init_node("navigate_viz_stack_tmux_launcher", anonymous=True, disable_signals=True)
    wait_for_joy_start()

    for p in [navigate_viz_py, recorder_py, pd_py, run_slam_py]:
        if not os.path.isfile(p):
            print(f"[ERROR] file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Fail if session exists
    has = subprocess.run(["tmux", "has-session", "-t", session],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if has.returncode == 0:
        print(f"[ERROR] tmux session already exists: {session}", file=sys.stderr)
        print(f"        Kill it manually if needed: tmux kill-session -t {session}", file=sys.stderr)
        sys.exit(1)

    # ROS for joystick watcher
    stop_event = threading.Event()
    JoyStopWatcher(
        stop_event=stop_event,
        joy_topic=args.joy_topic,
        btn_l1=args.btn_l1,
        btn_r1=args.btn_r1,
        btn_l2=args.btn_l2,
        axis_l2=args.axis_l2,
        axis_l2_pressed_thr=args.axis_l2_thr,
        hold_sec=args.hold_sec,
    )

    # Create tmux session + 2x2 panes
    tmux("new-session", "-d", "-s", session, "-n", "main")
    tmux("split-window", "-h", "-t", f"{session}:0.0")
    tmux("split-window", "-v", "-t", f"{session}:0.0")
    tmux("split-window", "-v", "-t", f"{session}:0.1")
    tmux("select-layout", "-t", session, "tiled")

    # Pane indices:
    P0 = f"{session}:0.0"  # former_client (left-top)
    P1 = f"{session}:0.1"  # SLAM (right-top)
    P2 = f"{session}:0.2"  # pd_controller (left-bottom)
    P3 = f"{session}:0.3"  # record_navsteps (right-bottom)

    def init_pane(pane: str, conda_env: Optional[str], cmd: str):
        pane_send(pane, f"source {shlex.quote(bashrc)}")
        if (not args.no_conda) and conda_env:
            pane_send(pane, f"conda activate {shlex.quote(conda_env)}")
        pane_send(pane, f"source {shlex.quote(catkin_setup)}")
        pane_send(pane, f"cd {shlex.quote(workdir)}")
        pane_send(pane, cmd)

    # Start SLAM first
    init_pane(P1, None if args.no_conda else args.conda_navdata,
              f"python3 {shlex.quote(run_slam_py)}")

    # Start former_client
    init_pane(P0, None if args.no_conda else args.conda_devgru_deployment,
              f"python {shlex.quote(navigate_viz_py)}")

    # Start recorder after a short delay (same spirit as your script)
    time.sleep(1.0)
    init_pane(P3, None if args.no_conda else args.conda_devgru_deployment,
              f"python {shlex.quote(recorder_py)}")

    # Start controller after another short delay
    time.sleep(1.0)
    init_pane(P2, None if args.no_conda else args.conda_pd,
              f"python {shlex.quote(pd_py)}")

    # When stop_event is set, send Ctrl+C to each pane (do NOT kill tmux)
    def stopper():
        stop_event.wait()
        # Send Ctrl+C to all panes (order doesn't matter)
        pane_ctrl_c(P0)
        pane_ctrl_c(P1)
        pane_ctrl_c(P2)
        pane_ctrl_c(P3)
        # Leave tmux session alive and panes open
        # Optionally: print a message in each pane
        pane_send(P0, 'echo "[STOP] joystick combo pressed - processes stopped"')
        pane_send(P1, 'echo "[STOP] joystick combo pressed - processes stopped"')
        pane_send(P2, 'echo "[STOP] joystick combo pressed - processes stopped"')
        pane_send(P3, 'echo "[STOP] joystick combo pressed - processes stopped"')

    threading.Thread(target=stopper, daemon=True).start()

    # Attach; Ctrl+C here will just detach/interrupt the attach (tmux stays)
    tmux("select-pane", "-t", P0)
    tmux("attach", "-t", session)


if __name__ == "__main__":
    main()
