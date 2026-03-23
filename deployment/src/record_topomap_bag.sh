#!/bin/bash

# Simply runs the python manual collection code located under ~/catkin_ws 
# Last modified 2025.07.27 by hkm

# Check if map_file_name is provided
#if [ -z "$1" ]; then
#    echo "Usage: $0 <topomap_dir>"
#    exit 1
#fi


#!/bin/bash

ORIG_DIR="$PWD"
CATKIN_WS="${CATKIN_WS:-$HOME/catkin_ws}"
NAVDATA_COLLECTOR_DIR="$CATKIN_WS/src/navdata_collector/run_script/data_collector"

# 1. Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate navdata

# 2. Source ROS setup
source "$CATKIN_WS/install/setup.bash"

# 3. Move into extractor dir (if needed)
cd "$NAVDATA_COLLECTOR_DIR"

# 4. Run the data collection node
rosrun navdata_collector run_manual_data_collection.py

# 5. Restore original directory
cd "$ORIG_DIR"
