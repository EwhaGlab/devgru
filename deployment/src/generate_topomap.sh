#!/bin/bash
set -euo pipefail

# -------------------------------
# Usage: ./create_topomap.sh <TOPOMAP_NAME> <EXTRACTED_DATA_DIR>
# Example: ./create_topomap.sh topomap0 ~/extracted_data
# -------------------------------

# Exit if no args provided
if [ $# -lt 1 ]; then
   echo "Usage: $0 <EXTRACTED_DATA_DIR> "
   exit 1
fi

EXTRACTED_DATA_DIR="$1"

ORIG_DIR="$PWD"

CATKIN_WS="${CATKIN_WS:-$HOME/catkin_ws}"
NAVDATA_EXTRACTOR_DIR="$CATKIN_WS/src/navdata_collector/run_script/data_extractor"
EXTRACTOR_SCRIPT="$NAVDATA_EXTRACTOR_DIR/script_extract_bags.py"
NAV_CFG="$CATKIN_WS/src/navdata_collector/param/navdata_collector.yaml" 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../..")"

# CONFIG_FILE lives under $PROJECT_DIR/config
DEPTH_CFG="$PROJECT_DIR/config/depth_nav.yaml"

TOPOMAP_NAME=$(grep -E '^[[:space:]]*topomap_name:' "$DEPTH_CFG" \
    | sed 's/#.*//' \
    | awk -F': *' '{print $2}' \
    | xargs)
 
# EXTRACT_DIR=$(grep -E '^[[:space:]]*topomap_name:' "$NAV_CFG" \
#     | sed 's/#.*//' \
#     | awk -F': *' '{print $2}' \
#     | xargs)

# if command -v yq >/dev/null 2>&1; then
#   TOPOMAP_NAME="$(yq '.depth_nav.topomap_name' "$CONFIG_FILE")"
# else
#   TOPOMAP_NAME="$(grep -E '^\s*topomap_name:' "$CONFIG_FILE" | awk '{print $2}')"
# fi

if [[ ! -f "$EXTRACTOR_SCRIPT" ]]; then
    echo "[ERROR] Extractor script not found: $EXTRACTOR_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$NAV_CFG" ]]; then
    echo "[ERROR] Navdata config not found: $NAV_CFG" >&2
    exit 1
fi

# --- Run extraction inside ROS + conda environment ---
source ~/catkin_ws/install/setup.bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navdata
cd "$NAVDATA_EXTRACTOR_DIR"

echo "executing $EXTRACTOR_SCRIPT"
#python $EXTRACTOR_SCRIPT "../../param/navdata_collector.yaml"


# 1. Make directory
echo "TOPOMAP name: $TOPOMAP_NAME"
echo "[INFO] Project dir     : $PROJECT_DIR"
echo "[INFO] Config file     : $DEPTH_CFG"
echo $NAVDATA_EXTRACTOR_DIR

TOPOMAP_DIR="$PROJECT_DIR/deployment/topomaps/$TOPOMAP_NAME"
echo "TOPOMAP_DIR:  $TOPOMAP_DIR"

mkdir -p "$TOPOMAP_DIR"

# 2. Copy extracted data
cp -r "$EXTRACTED_DATA_DIR"/* "$TOPOMAP_DIR"/

echo "finished copying topomap, creating slam_poses.txt"
# 3. gen slam_poses
PATH_TO_PGO="$TOPOMAP_DIR/map"

OUT_POSE_TXT="$TOPOMAP_DIR/slam_poses.txt"
rosrun navdata_collector dump_posegraph $PATH_TO_PGO $OUT_POSE_TXT

cd "$ORIG_DIR"
python create_synced_topomap.py

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vint_deployment
else
  echo "[WARN] conda not on PATH; skipping conda activate. Ensure deps are available."
fi
echo "[INFO] Done. new topomap created at $TOPOMAP_DIR"

# extract slam_poses.txt

