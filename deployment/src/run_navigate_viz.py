#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import time


def tmux(*args):
    subprocess.run(["tmux", *args], check=True)


def pane_send(pane: str, line: str):
    tmux("send-keys", "-t", pane, line, "Enter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="devgru_nav")

    parser.add_argument("--catkin_setup", default="~/catkin_ws/install/setup.bash")
    parser.add_argument("--bashrc", default="~/.bashrc")

    parser.add_argument("--navigate_viz_py", default="navigate_viz.py")
    parser.add_argument("--pd_py", default="pd_controller.py")

    parser.add_argument("--conda_env", default="devgru_deployment")
    parser.add_argument("--no_conda", action="store_true")

    args = parser.parse_args()

    session = args.session
    workdir = os.getcwd()
    bashrc = os.path.abspath(os.path.expanduser(args.bashrc))
    catkin_setup = os.path.abspath(os.path.expanduser(args.catkin_setup))

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(workdir, p)

    navigate_viz_py = abspath(args.navigate_viz_py)
    pd_py = abspath(args.pd_py)

    for p in [navigate_viz_py, pd_py]:
        if not os.path.isfile(p):
            print(f"[ERROR] file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Check tmux session
    has = subprocess.run(["tmux", "has-session", "-t", session],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if has.returncode == 0:
        print(f"[ERROR] tmux session already exists: {session}")
        sys.exit(1)

    # Create session with 2 panes
    tmux("new-session", "-d", "-s", session, "-n", "main")
    tmux("split-window", "-h", "-t", f"{session}:0.0")
    tmux("select-layout", "-t", session, "tiled")

    P0 = f"{session}:0.0"
    P1 = f"{session}:0.1"

    def init_pane(pane, cmd):
        pane_send(pane, f"source {shlex.quote(bashrc)}")

        if not args.no_conda:
            pane_send(pane, f"conda activate {shlex.quote(args.conda_env)}")

        pane_send(pane, f"source {shlex.quote(catkin_setup)}")
        pane_send(pane, f"cd {shlex.quote(workdir)}")
        pane_send(pane, cmd)

    # Start navigate_viz
    init_pane(P0, f"python {shlex.quote(navigate_viz_py)}")

    # Small delay (important for ROS topics readiness)
    time.sleep(1.0)

    # Start PD controller
    init_pane(P1, f"python {shlex.quote(pd_py)}")

    # Attach session
    tmux("select-pane", "-t", P0)
    tmux("attach", "-t", session)


if __name__ == "__main__":
    main()
