#!/bin/bash

# Create a new tmux session

cd ../topomaps/bags
rosbag record /camera/color/image_raw -o $1

