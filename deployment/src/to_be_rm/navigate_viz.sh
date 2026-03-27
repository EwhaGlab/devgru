#!/usr/bin/env bash
set -euo pipefail

# If --viz_topomap false (default):
#   - do NOT load config
#   - create 2 panes (navigate_viz, pd_controller)
# If --viz_topomap true:
#   - load ../../config/depth_nav.yaml -> /depth_nav
#   - create 3 panes (left-to-right)
#       Pane 0: run_localizer_from_topomap.py (FIRST)
#       Pane 1: navigate_viz.py
#       Pane 2: pd_controller.py
#
# Last modified 2025.12.19 by hkm (edited)

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

# --- find script dir (robust even if called from elsewhere) ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# depth_nav.yaml is located at: ../../config/depth_nav.yaml (relative to THIS .sh file)
NAV_CFG="$(realpath "${SCRIPT_DIR}/../../config/depth_nav.yaml")"

if [[ ! -f "$NAV_CFG" ]]; then
  echo "[ERROR] Config file not found: $NAV_CFG" >&2
  exit 1
fi

TOPOMAP_NAME="$(yq -r '.deployment.topomap_name' "$NAV_CFG")"

TOPOMAPS_BASE="$(realpath "${SCRIPT_DIR}/../topomaps")"
topomap_dir="${TOPOMAPS_BASE}/${TOPOMAP_NAME}"
# -----------------------------
# Parse args:
#   --viz_topomap <true|false>   (default true)
#   --topomap_dir <dir>          (required if viz_topomap=true)
# Everything else passes through to navigate_viz.py
# -----------------------------
viz_topomap="false"
passthrough_args=()
to_lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

if [[ $# -ge 1 ]]; then
  case "$(to_lower "$1")" in
    true|false)
      viz_topomap="$(to_lower "$1")"
      shift
      ;;
  esac
fi


while [[ $# -gt 0 ]]; do
  case "$1" in
    --viz_topomap)
      [[ $# -ge 2 ]] || { echo "[ERROR] --viz_topomap requires {true|false}" >&2; exit 1; }
      viz_topomap="$(to_lower "$2")"
      shift 2
      ;;
    --topomap_dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] --topomap_dir requires an argument" >&2; exit 1; }
      topomap_dir="$2"
      shift 2
      ;;
    *)
      passthrough_args+=("$1")
      shift
      ;;
  esac
done

echo $viz_topomap
echo $topomap_dir


if [[ "$viz_topomap" != "true" && "$viz_topomap" != "false" ]]; then
  echo "[ERROR] --viz_topomap must be {true|false}, got: $viz_topomap" >&2
  exit 1
fi

# IMPORTANT: set this to the ROS package that contains run_localizer_from_topomap.py
LOCALIZER_PKG="navdata_collector"
LOCALIZER_NODE="run_localizer_from_topomap.py"

session_name="DepGRU_former_deployment_$(date +%s)"

cleanup() {
  tmux has-session -t "${session_name}" 2>/dev/null && tmux kill-session -t "${session_name}" || true
}
trap cleanup INT TERM

tmux new-session -d -s "${session_name}"

if [[ "$viz_topomap" == "true" ]]; then
  # -----------------------------
  # Load config (only when viz_topomap=true)
  # -----------------------------
  if [[ ! -f "$NAV_CFG" ]]; then
    echo "[ERROR] Config file not found: $NAV_CFG" >&2
    exit 1
  fi

  echo "[INFO] viz_topomap=true"
  echo "[INFO] Using config: $NAV_CFG"

  # topomap_dir required when viz_topomap=true
  if [[ -z "${topomap_dir}" ]]; then
    echo "[ERROR] --topomap_dir is required when --viz_topomap true" >&2
    exit 1
  fi
  [[ -d "${topomap_dir}" ]] || { echo "[ERROR] topomap_dir not found: ${topomap_dir}" >&2; exit 1; }

  # -----------------------------
  # Create 3 equal panes (left-to-right)
  # -----------------------------
  tmux split-window -h -p 67 -t "${session_name}:0.0"   # left 33% | right 67%
  tmux split-window -h -p 50 -t "${session_name}:0.1"   # split right 67% into 33% and 33%

  # Pane 0: localizer FIRST
  tmux select-pane -t "${session_name}:0.0"
  tmux send-keys "source ~/.bashrc" Enter
  tmux send-keys "conda activate navdata" Enter
  tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
  tmux send-keys "cd /home/glab/catkin_ws/src/navdata_collector/run_script/data_collector" Enter
  tmux send-keys "python run_localizer_from_topomap.py --topomap_dir=\"${topomap_dir}\"" Enter
  
  echo $topomap_dir

  # Pane 1: navigate_viz
  tmux select-pane -t "${session_name}:0.1"
  tmux send-keys "source ~/.bashrc" Enter
  tmux send-keys "conda activate vint_deployment" Enter
  tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
  nav_cmd=(python navigate_viz.py "${passthrough_args[@]}")
  tmux send-keys "$(printf '%q ' "${nav_cmd[@]}")" Enter

  
  # Sleep 1 sec before starting controller
  sleep 1

  # Pane 2: pd_controller
  tmux select-pane -t "${session_name}:0.2"
  tmux send-keys "source ~/.bashrc" Enter
  tmux send-keys "conda activate vint_deployment" Enter
  tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
  tmux send-keys "python pd_controller.py" Enter

else
  echo "[INFO] viz_topomap=false (skip config + localizer)"

  # 2 equal panes (left/right)
  tmux split-window -h -p 50 -t "${session_name}:0.0"

  # Pane 0: navigate_viz
  tmux select-pane -t "${session_name}:0.0"
  tmux send-keys "source ~/.bashrc" Enter
  tmux send-keys "conda activate vint_deployment" Enter
  tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
  nav_cmd=(python navigate_viz.py "${passthrough_args[@]}")
  tmux send-keys "$(printf '%q ' "${nav_cmd[@]}")" Enter

  sleep 1

  # Pane 1: pd_controller
  tmux select-pane -t "${session_name}:0.1"
  tmux send-keys "source ~/.bashrc" Enter
  tmux send-keys "conda activate vint_deployment" Enter
  tmux send-keys "source ~/catkin_ws/install/setup.bash" Enter
  tmux send-keys "python pd_controller.py" Enter
fi

tmux -2 attach-session -t "${session_name}"

