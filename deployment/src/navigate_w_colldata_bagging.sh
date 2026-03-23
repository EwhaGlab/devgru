#!/bin/bash

# Create a new tmux session for collecting collision data. 
# It drives the robot w/ the trained DepGRU policy. 
# Bagging runs in parallel with a different ROS node to collect the collision node 
# Push L1, L2, R1, and R2 simultaneously to stop the process !!
# Last modified 2025.07.27 by hkm

# Check if map_file_name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <topomap_dir>"
    exit 1
fi

#TOPOMAP_DIR=$1 #"../topomaps"
TOPOMAP_DIR="$(realpath "$1")"

session_name="DevGRU_former_$(date +%s)"
tmux new-session -d -s $session_name

tmux split-window -h -p 66  # Pane 0 = 66%, Pane 1 = 34%

# Then: split pane 1 (right) into two equal horizontal panes
tmux select-pane -t 1
tmux split-window -h -p 50  # Pane 1 = 17%, Pane 2 = 17%

# Split the window into four panes
# Run the roslaunch command in the first pane

tmux select-pane -t 0
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
tmux send-keys "python navigate_w_bagging_data.py --topomap_dir=${TOPOMAP_DIR}" Enter

# Run the colldata_collector script in the 2nd pane
tmux select-pane -t 1
tmux send-keys "conda activate navdata" Enter
tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
tmux send-keys "python ~/catkin_ws/src/navdata_collector/run_script/data_collector/run_colldata_collection.py --topomap_dir=${TOPOMAP_DIR}" Enter

# Run the pd_controller.py script in the fourth pane

tmux select-pane -t 2
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
tmux send-keys "python pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
