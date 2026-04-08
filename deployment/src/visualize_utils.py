import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from typing import Optional, List
import wandb
import yaml

from PIL import Image
import torch
import torch.nn as nn
import math
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

from os.path import dirname
BASE_DIR = os.path.join(dirname(dirname(dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(BASE_DIR)
import time
import utils.rigid_motion as rm

X_VIZ_START_OFFSET = 1.0
VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])

MODEL_PARAM_PATH = '%s/config/devgru.yaml'%BASE_DIR
DATA_CONFIG_PATH = '%s/config/data_config.yaml'%BASE_DIR

with open(MODEL_PARAM_PATH, "r") as f:
    nav_config = yaml.safe_load(f)

with open(DATA_CONFIG_PATH, "r") as f:
    data_config = yaml.safe_load(f)

def numpy_to_img(arr: np.ndarray) -> Image:
    img = Image.fromarray(np.transpose(np.uint8(255 * arr), (1, 2, 0)))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()



# def plot_projected_pred_trajs_and_points_on_image(
#     ax: plt.Axes,  # ax01
#     viz_obs_img: np.ndarray,
#     dataset_name: str,
#     pred_waypts: np.ndarray,
#     list_points: list,      # [ start_pos,      goal_pos]  wrt robot base frame
#     waypts_color: CYAN,
#     point_colors: list = [GREEN, RED],
# ):
#     """
#     Plot the projected trajectories and points on an image.
#     If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
#     Args:
#         ax: matplotlib axis
#         viz_img: image to plot
#         dataset_name: name of the dataset found in data_config.yaml (e.g. "former")
#         list_waypts: list of waypoints, each waypoint is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
#         list_points: list of points, each point is a numpy array of shape (2,)
#         waypts_colors: list of colors for trajectories
#         point_colors: list of colors for points
#     """
#     assert len(list_points) <= len(point_colors), "Not enough colors for points"
#     assert (
#         dataset_name in data_config
#     ), f"Dataset {dataset_name} not found in data/data_config.yaml"
#
#     ax.imshow(viz_obs_img)  # img size : (640, 480)
#     if (
#         "camera_matrics" in data_config[dataset_name]
#         and "cam_wrt_base" in data_config[dataset_name]["camera_matrics"]       # extrinsic params
#         and "camera_matrix" in data_config[dataset_name]["camera_matrics"]      # intrinsic params
#         and "dist_coeffs" in data_config[dataset_name]["camera_matrics"]        # distortion coeff
#     ):
#         img_width = data_config[dataset_name]['img_width']
#         img_height= data_config[dataset_name]['img_height']
#         u_scale = img_width / VIZ_IMAGE_SIZE[0]  # 640
#         v_scale = img_height/ VIZ_IMAGE_SIZE[1]  # 480
#
#         bHc = rm.xyzrpy_to_htm( data_config[dataset_name]["camera_matrics"]["cam_wrt_base"] )
#         cHb = np.linalg.inv(bHc)
#         z_offset = bHc[2, 3]
#
#         fx = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["fx"]
#         fy = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["fy"]
#         cx = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["cx"]
#         cy = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["cy"]
#         K = gen_camera_matrix(fx, fy, cx, cy)
#
#         if False: # We consider distortion free cam only for now
#             k1 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["k1"]
#             k2 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["k2"]
#             p1 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["p1"]
#             p2 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["p2"]
#             k3 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["k3"]
#             dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])
#
#         #for i, waypts in enumerate(list_waypts): # way_points
#             #xy_coords = traj[:, :2]    # N x 2    (horizon, 2)
#         if len(pred_waypts.shape) == 1: # is 1 if we care about only one waypoint. i.e, the point btwn start_pos and goal_pos
#             # add a dimension to the front of point
#             pred_waypts = pred_waypts[None, ...]
#
#         num_pts = pred_waypts.shape[0]
#
#         xy   = pred_waypts[:, :2].transpose()  # (2,) 2 x N
#         N    = xy.shape[1]
#         xy_b = xy.copy()
#         z_h  = np.ones([2, N])
#         z_h[0] = z_h[0] * 0 #z_offset
#         xyz_h_b = np.concatenate( [xy_b, z_h], axis=0 )         # xyz wrt b
#         xyz_c = np.matmul( cHb[:3,...], xyz_h_b ).squeeze()     # xyz wrt c
#         #assert( not( xyz_c[0] == 0 and xyz_c[0] == 0 and xyz_c[0] == 0) )
#         uv_pred = rm.pinhole_projection(K, xyz_c)
#
#         # We assume distortion free cam for now...
#         ax.plot(
#             uv_pred[0,:] / u_scale,  # u
#             uv_pred[1,:] / v_scale,  # v
#             color=waypts_color,
#             marker='D',
#             markersize=8.0,
#             #lw = 2.5
#         )
#
#
#         for i, point in enumerate(list_points):
#             if len(point.shape) == 1:
#                 # add a dimension to the front of point
#                 point = point[None, :2]
#             else:
#                 point = point[:, :2]
#
#             xy   = point.transpose() # 2 x N
#             N    = xy.shape[1]
#             xy_b = xy.copy()
#             z_h  = np.ones([2, N])
#             z_h[0] = z_h[0] * 0
#             xyz_h_b = np.concatenate( [xy_b, z_h], axis=0 )         # xyz wrt b
#             xyz_c = np.matmul( cHb[:3,...], xyz_h_b ).squeeze()     # xyz wrt c
#             assert( not( xyz_c[0] == 0 and xyz_c[0] == 0 and xyz_c[0] == 0) )
#             uv = rm.pinhole_projection(K, xyz_c)
#
#             ax.plot(
#                 uv[0] / u_scale,
#                 uv[1] / v_scale,
#                 color=point_colors[i],
#                 marker="o",
#                 markersize=10.0,
#             )
#
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
#         ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
#         ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))

prev_waypoint_cb_time = time.time()
curr_waypoint_cb_time = time.time()

def _to_depth_uint8(img, vmin=None, vmax=None):
    """Convert PIL/np/torch depth to uint8 for viewing. Auto-scales if vmin/vmax not set."""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError("Depth must be HxW (grayscale)")

    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = img.min() if vmin is None else float(vmin)
    vmax = img.max() if vmax is None else float(vmax)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - vmin) / (vmax - vmin)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def _unit_bearing_from_yaw(yaw_rad: float):
    bx = np.cos(yaw_rad)
    by = np.sin(yaw_rad)
    return bx, by

    vmin = img.min() if vmin is None else float(vmin)
    vmax = img.max() if vmax is None else float(vmax)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - vmin) / (vmax - vmin)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def _unit_bearing_from_yaw(yaw_rad: float):
    return np.cos(yaw_rad), np.sin(yaw_rad)

def plot_image_and_nav_pose(
    ax0, ax1,
    waypoint_pred, list_points,
    waypoint_color=(0.0, 1.0, 1.0),
    point_colors=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    point_labels=("robot", "goal"),
    quiver_freq=1, learn_angle=False,
    collision_status=False,
    depth_img=None, depth_cmap="gray",
    depth_vmin=None, depth_vmax=None,
):
    import time
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    now = time.time()
    if not hasattr(plot_image_and_nav_pose, "_prev_t"):
        plot_image_and_nav_pose._prev_t = now
    elapsed = now - plot_image_and_nav_pose._prev_t
    plot_image_and_nav_pose._prev_t = now
    pub_freq = (1.0 / max(elapsed, 1e-6))

    # ---- Left: depth image ----
    ax0.clear()
    #ax0.set_axis_off()

    depth_aspect = 1.0
    if depth_img is not None:
        img8 = _to_depth_uint8(depth_img, vmin=depth_vmin, vmax=depth_vmax)  # your helper
        H, W = img8.shape
        depth_aspect = H / float(W)

        # Force same physical box shape for BOTH axes
        ax0.set_box_aspect(depth_aspect)
        ax1.set_box_aspect(depth_aspect)

        ax0.imshow(img8, cmap=depth_cmap, vmin=0, vmax=255)

        # Overlay status banner/text on depth
        status = "COLLISION" if collision_status else "SAFE"
        color_txt = "red" if collision_status else "green"
        banner_h = 0.14
        rect = Rectangle((0.0, 1.0 - banner_h), 1.0, banner_h,
                         transform=ax0.transAxes,
                         facecolor=(1, 0, 0, 0.35) if collision_status else (0, 1, 0, 0.35),
                         edgecolor='none')
        ax0.add_patch(rect)
        ax0.text(0.5, 1.0 - banner_h/2.0,
                 f"{status}\nWp period: {elapsed*1000:.2f} ms | Freq: {pub_freq:.2f} Hz",
                 ha="center", va="center", transform=ax0.transAxes,
                 fontsize=11, color=color_txt)

    # ---- Right: FULL-SIZE heading plot (no inset) ----
    waypoint_pred = np.asarray(waypoint_pred, dtype=float)
    if waypoint_pred.ndim == 1:
        waypoint_pred = waypoint_pred[None, :]
    start = np.asarray(list_points[0], dtype=float).ravel()
    goal  = np.asarray(list_points[1], dtype=float).ravel()

    ax1.clear()
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # convert to plotting frame: (plot_x, plot_y) = (-y, x)
    wpx = -waypoint_pred[:, 1]
    wpy =  waypoint_pred[:, 0]
    sx, sy = -start[1], start[0]
    gx, gy = -goal[1],  goal[0]

    ax1.plot(wpx, wpy, color=waypoint_color, marker="D", linestyle="-", linewidth=1.3, markersize=4, label="pred")
    ax1.plot(sx, sy, color=point_colors[0], marker="o", markersize=6, linestyle="None", label=point_labels[0] if point_labels else None)
    ax1.plot(gx, gy, color=point_colors[1], marker="o", markersize=6, linestyle="None", label=point_labels[1] if point_labels else None)

    if learn_angle and waypoint_pred.shape[1] >= 4:
        for ii in range(0, len(waypoint_pred), max(1, quiver_freq)):
            yaw = waypoint_pred[ii, 3]
            bx, by = np.cos(yaw), np.sin(yaw)
            qx, qy = -by, bx
            ax1.quiver(wpx[ii], wpy[ii], qx, qy,
                       angles='xy', scale_units='xy', scale=1.0,
                       width=0.004, alpha=0.8, color=waypoint_color)

    # Same data limits padding
    all_x = np.concatenate([wpx, [sx, gx]])
    all_y = np.concatenate([wpy, [sy, gy]])
    pad_x = max(0.1, 0.1 * (np.max(np.abs(all_x)) + 1e-6))
    pad_y = max(0.1, 0.1 * (np.max(np.abs(all_y)) + 1e-6))
    xlim = np.max(np.abs(all_x)) + pad_x
    ylim = np.max(np.abs(all_y)) + pad_y

    ax1.set_xlim([-xlim, xlim]); ax1.set_ylim([-ylim, ylim])
    ax1.set_aspect('equal', adjustable='box')  # equal units; 'box' respects set_box_aspect above
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{-v:.1f}"))  # flip sign on x labels
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))
    ax1.set_xlabel('Y (base_link)'); ax1.set_ylabel('X (base_link)')

    if point_labels:
        ax1.legend(loc='upper right', fontsize=8, frameon=True)

    ax0.figure.canvas.draw_idle()
    plt.pause(0.001)

def visualize_nav_status(
    fig,
    ax0: plt.Axes, ax01: plt.Axes, ax02: plt.Axes, ax11: plt.Axes, ax12: plt.Axes,
    info: tuple,
    viz_obs_depth,      # PILImage
    viz_goal_depth,     # PILImage
    viz_obs_img,
    viz_goal_img,
    dataset_name: str,
    goal_poses: np.ndarray,   # x, y, th
    pred_waypoint: np.ndarray,  # in meters  (len_pred, num_param)
    #old_waypoint: np.ndarray,
    pose_diff_pred: np.ndarray,
    is_collision: bool,   # 0 / 1
    sg_poses_wrt_base_link: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
    xy_dist_thr: float = 0.2,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization.
    This is a modification of the original code: "compare_waypoints_pred_to_label"

    Args:
        viz_obs_img: image of the observation (640, 480)
        viz_goal_img: image of the goal
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        goal_pos: goal position in the image
        pred_waypoints: predicted waypoints in the image
        label_waypoints: label waypoints in the image
        save_path: path to save the figure
        display: whether to display the figure
    """
    ax0.clear(); ax01.clear(); ax02.clear(); ax11.clear(); ax12.clear()

    # ---- big left: obs depth + status banner ----
    ax01.set_axis_off()
    ax02.set_axis_off()
    ax11.set_axis_off()
    ax12.set_axis_off()

    # We consider cam orig as the start_pos for the visualization
    bHc = rm.xyzrpy_to_htm(data_config[dataset_name]["camera_matrics"]["cam_wrt_base"])
    x_offset = bHc[0, 3] + X_VIZ_START_OFFSET # +0.6 projects uv at the bottom of the (isaac sim)imgs
    y_offset = bHc[1, 3]

    start_pos = np.array([0, 0, 0.0, 0.0])
    start_pos_w_offset = np.array([x_offset, y_offset, 0.0, 0.0])  # start cam pose
    goal_pos_w_offset  = goal_poses[0].copy()
    goal_pos_w_offset[0] += x_offset
    goal_pos_w_offset[1] += y_offset

    #load the corrected sg if exist one
    points_list = []
    points_list_w_offset = []
    ws = nav_config['datasets'][dataset_name]['waypoint_spacing']

    points_list_w_offset = [start_pos_w_offset, goal_pos_w_offset]
    #points_list = [start_pos, goal_pos]

    text_color = "black"
    dist_comp_txt = ""

#    _, _, yaw_pred = rm.quat2rpy( [pose_diff_pred[2],  0, 0, pose_diff_pred[3] ])

    if is_collision:
        status_color = 'red'
    else:
        status_color = 'green'

    if len(pred_waypoint.shape) > 2:
        raise Exception(" Must be only single way point \n")
    else:
        #list_waypoints = [pred_waypoint]
        np_pred_waypoint_w_offset = pred_waypoint.copy()
        np_pred_waypoint_w_offset[:,0] += x_offset
        np_pred_waypoint_w_offset[:,1] += y_offset
        #np_old_waypoint_w_offset = old_waypoint.copy()
        #np_old_waypoint_w_offset[:,0] += x_offset
        #np_old_waypoint_w_offset[:,1] += y_offset

    plot_cam_heading_and_points(
        ax =ax0,                # ax[0][0],
        pred_waypoint=pred_waypoint,      # in meters   scale up for vizualization
        pose_diff_pred=pose_diff_pred,
        start_point= start_pos,
        goal_points= goal_poses,
        #points_list,        # in meters   [start_pos, goal_pos],      # wrt base
        is_collision=is_collision,
        sg_poses_wrt_base_link=sg_poses_wrt_base_link,
        waypoint_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED, BLUE],
        xy_dist_thr = xy_dist_thr
    )

    width, height = viz_obs_img.size
    print(width, height)

    if dataset_name == 'former' or dataset_name == 'isaac_sim':
       plot_projected_trajs_and_points_on_image(
            ax01, #ax[0][1],
            viz_obs_img,    # (640, 480)
            #viz_obs_depth,
            dataset_name,
            np_pred_waypoint_w_offset,
            np_pred_waypoint_w_offset,
            points_list_w_offset, #[start_pos_w_offset, goal_pos_w_offset, corrected_goal_pos_w_offset],
            waypts_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED, BLUE],
       )
    elif dataset_name == 'thud':
        raise Exception("Not Implemented for 6d pose vec ")

    else:
        raise Exception("Unknown dataset_name <%s>"%dataset_name)

    ax02.imshow(viz_goal_img)
    ax11.imshow(viz_obs_depth,  cmap='gray', vmin=0, vmax=255)
    ax12.imshow(viz_goal_depth, cmap='gray', vmin=0, vmax=255)

    ax0.set_title(f"Action Prediction")
    ax01.set_title(f"Observation")
    ax02.set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    # if not display:
    #     plt.close(fig)


def plot_cam_heading_and_points(
        ax: plt.Axes,
        pred_waypoint: np.ndarray, # wpts
        pose_diff_pred: np.ndarray, # new sg pos w.r.t base
        #list_points: list,          # start & goal pts (pose)
        start_point: np.ndarray,    # start
        goal_points: np.ndarray,    # future sgs
        is_collision: bool,
        sg_poses_wrt_base_link: Optional[np.ndarray] = None,
        waypoint_colors: list = [CYAN, MAGENTA],
        point_colors: list = [GREEN, RED, BLUE],
        waypoint_labels: Optional[list] = ["prediction", "ground truth"],
        point_labels: Optional[list] = ["robot cent", "SG", "new SG"],
        waypoint_alphas: Optional[list] = None,
        point_alphas: Optional[list] = None,
        quiver_freq: int = 1,
        default_coloring: bool = True,
        xy_dist_thr: float = 0.2,
        dataset_name: str = 'former',
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        pred_waypoint:
        list_points: list of points: [start point and goal point], each point is a numpy array of shape (2,)
        waypoint_colors: list of colors for waypoints: waypoints[0]= pred (CYAN),  waypoints[1]= GT (MAGENTA)
        point_colors: list of colors for points:  start = RED,  goal = GREEN
        waypoint_labels: list of labels for waypoints
        point_labels: list of labels for points
        waypoint_alphas: list of alphas for waypoints
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    #assert point_labels is None or len(list_points) <= len(point_labels), "Not enough labels for points"

    waypoint_pred = np.atleast_2d(pred_waypoint)   # (len_pred, num_param)    # x, y

    #################################################################################
    #               base_link configuration
    #
    #                         ^  (X)
    #                         |
    #                         |
    #                         |
    #               Y <------(+) (Z)
    #################################################################################

    # swap X and Y and use -Y instead of Y to represent x,y based on the base_link config

    (len_pred, num_params) = waypoint_pred.shape
    assert(num_params == 2 or num_params == 4)
    if waypoint_labels is None:
        ax.plot(
            -waypoint_pred[:, 1],  # y  (negative y b/c y points to the right (see the base_link conf above)
            waypoint_pred[:, 0],   # x
            color=waypoint_colors[0],
            alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            marker="D",
        )

    else:
        ax.plot(
            -waypoint_pred[:, 1],
            waypoint_pred[:, 0],  # traj[:, 1],
            color=waypoint_colors[0],
            label=waypoint_labels[0],
            alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            marker="D",
        )
        if nav_config['learn_angle'] == True: #len(waypoint_pred) == 4:
            for ii in range(0, len_pred):
                bearing_pred = gen_bearing(waypoint_pred[ii])  # draw heading dir
                ax.quiver(
                    -waypoint_pred[ii,1],  # y
                    waypoint_pred[ii,0],  # x
                    -bearing_pred[1],
                    bearing_pred[0],
                    color=waypoint_colors[0] * 0.5,
                    scale=1.0,
                )

            #goal  = list_points[1]
            for goal in goal_points:
                bearing_goal = gen_bearing(goal)
                ax.quiver(
                    -goal[1],  # y
                    goal[0],  # x
                    -bearing_goal[1],
                    bearing_goal[0],
                    color=point_colors[1] * 0.5,
                    scale=1.0,
                )

    #for i, pt in enumerate(list_points):
    #start = list_points[0]
    #goal  = list_points[1]
    # new_goal = goal.copy()
    # if len(list_points) > 2:
    #     new_goal = list_points[2]

    start = start_point
    robot_cx = -start[1]
    robot_cy =  start[0]

    thr_circle = Circle(
        (robot_cx, robot_cy),
        radius=xy_dist_thr,
        fill=False,
        edgecolor=point_colors[0],   # or 'yellow'
        linewidth=2,
        linestyle='--',
        alpha=0.9,
        zorder=3,
        label="SG reach threshold"
    )
    ax.add_patch(thr_circle)

    # draw start pos and goal pos
    if point_labels is None:
        ax.plot(
            -start[1], # start cam x wrt base_link
            start[0], # start cam z wrt base_link
            color="darkgreen", #point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0
        )
        for goal in goal_points:
            ax.plot(
                -goal[1], # goal pos wrt curr base_link
                goal[0],
                color=point_colors[1],
                alpha=point_alphas[1] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
    else:
        ax.plot(
            -start[1],
            start[0],
            color="darkgreen", #point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0,
            label=point_labels[0],
        )
        goal = goal_points[0]
        ax.plot(
            -goal[1],
            goal[0],
            color=point_colors[1],
            alpha=1.0,
            marker="v",
            markersize=9.0,
            fillstyle='top',
            label="%s 0" % (point_labels[1]),
        )
        for ii in range(1, len(goal_points)):
            goal = goal_points[ii]
            ax.plot(
                -goal[1],
                goal[0],
                color="lightcoral",
                alpha=0.7,
                marker="v",
                markersize=7.0,
                fillstyle ='top',
                label="%s %d"%(point_labels[1], ii),
            )

        if is_collision:
            ax.plot(
                -pose_diff_pred[1],  # y  (negative y b/c y points to the right (see the base_link conf above)
                pose_diff_pred[0],   # x
                color=BLUE,
                alpha=1.0,
                marker="*",
                markersize=7.0,
                label=point_labels[2]
            )

        ax.grid(True)

    if sg_poses_wrt_base_link is not None:
        ax.plot(
            -sg_poses_wrt_base_link[:, 1],
            sg_poses_wrt_base_link[:, 0],  # traj[:, 1],
            color=BLUE,
            label="topomap",
            alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            marker="o",
        )

    # put the legend below the plot
    if waypoint_labels is not None or point_labels is not None:
        handles, labels = ax.get_legend_handles_labels()
        # Split
        handles_left = handles[:3]
        labels_left = labels[:3]
        handles_right = handles[3:]
        labels_right = labels[3:]
        # Left-bottom legend
        legend_left = ax.legend(
            handles_left,
            labels_left,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=True,
        )
        # Add it manually so it doesn't get overwritten
        ax.add_artist(legend_left)
        # Right-bottom legend
        ax.legend(
            handles_right,
            labels_right,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.0),
            frameon=True,
        )
        #ax.legend(loc='lower right', shadow=True, fontsize='medium')

    waypoint_speed = data_config[dataset_name]['max_speed'] / data_config[dataset_name]['img_fps'] * nav_config['datasets'][dataset_name]['waypoint_spacing']
    # if nav_config['normalize'] is True:
    #     plot_xlim = 1.2       # xy swapped
    #     plot_ylim = 1.2       #goal_dist_max / nav_config['datasets'][dataset_name]['waypoint_spacing'] + 0.1
    # else:
    #     plot_xlim = 3.       # xy swapped for drawing
    #     plot_ylim = 5. #3.4

    plot_xlim = 2.0# 0.8 #1.5   #max(1.0, plot_xlim) + 0.1
    plot_ylim = 2.0# 0.8 #1.5   #max(1.0, plot_ylim) + 0.1
    offset = 0.2
    # ----- Axis limits -----
    ax.set_xlim([-plot_xlim, plot_xlim])
    ax.set_ylim([-plot_ylim + offset, plot_ylim + offset])
    ax.set_aspect("equal", "box")

    # ----- Solid 2-pixel bounding box around axis -----
    rect = Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor="black",
        linewidth=2,
        zorder=-1  # <-- IMPORTANT
    )
    ax.add_patch(rect)

    # ----- Axis tick labels (meters) -----
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{-v:.1f}"))  # left-positive meters
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))  # forward meters

    # ----- Axis labels WITH units -----
    ax.set_xlabel('Y-axis (m)')
    ax.set_ylabel('X-axis (m)')

    # ----- Put Y-axis ticks on the right -----
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', labelright=True, labelleft=False)


    # if len(list_points) == 2:
    #     textstr0 = '\n'.join((
    #         r'Old goal(xy) = (%.2f, %.2f)'% (goal[0], goal[1]), # x y
    #         r'curr=(%.2f, %.2f)'% (start[0], start[1]), ) )
    # else:
    #     textstr0 = '\n'.join((
    #         r'Old goal(xy) = (%.2f, %.2f)'% (goal[0], goal[1]), # x y
    #         r'New goal(xy) = (%.2f, %.2f)'% (new_goal[0], new_goal[1]),
    #         r'curr=(%.2f, %.2f)'% (start[0], start[1]), ) )
    #
    # textstr1 = 'x_p, y_p, qw_p, qz_p (pred waypts)\n'
    # for idx, row in enumerate(waypoint_pred): #last_waypts_combined):
    #     (xp, yp, qw, qz) = row #(xp,yp,xg,yg) = row
    #     tmp = f'%.2f, %.2f, %.2f, %.2f\n'%(xp, yp, qw, qz)
    #     textstr1 +=tmp
    #
    # textstr = '\n'.join( (textstr0, textstr1) )
    #
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.0, 1.1, textstr, transform=ax.transAxes, fontsize=14,
    #          horizontalalignment='left', verticalalignment='bottom', bbox=props)



def plot_projected_trajs_and_points_on_image(
    ax: plt.Axes,  # ax01
    viz_obs_img: np.ndarray,
    dataset_name: str,
    pred_waypts: np.ndarray,
    label_waypts: np.ndarray,
    list_points: list,      # [ start_pos,      goal_pos]  wrt robot base frame
    waypts_colors: list = [CYAN, MAGENTA],
    point_colors: list = [GREEN, RED, BLUE],
):
    """
    Plot the projected trajectories and points on an image.
    If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
    Args:
        ax: matplotlib axis
        viz_img: image to plot
        dataset_name: name of the dataset found in data_config.yaml (e.g. "former")
        list_waypts: list of waypoints, each waypoint is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        waypts_colors: list of colors for trajectories
        point_colors: list of colors for points
    """
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        dataset_name in data_config
    ), f"Dataset {dataset_name} not found in data/data_config.yaml"

    W, H = viz_obs_img.size
    ax.imshow(viz_obs_img)  # img size : (640, 480)
    if (
        "camera_matrics" in data_config[dataset_name]
        and "cam_wrt_base" in data_config[dataset_name]["camera_matrics"]       # extrinsic params
        and "camera_matrix" in data_config[dataset_name]["camera_matrics"]      # intrinsic params
        and "dist_coeffs" in data_config[dataset_name]["camera_matrics"]        # distortion coeff
    ):
        img_width = data_config[dataset_name]['img_width']
        img_height= data_config[dataset_name]['img_height']
        u_scale = img_width / W
        v_scale = img_height/ H

        bHc = rm.xyzrpy_to_htm( data_config[dataset_name]["camera_matrics"]["cam_wrt_base"] )
        cHb = np.linalg.inv(bHc)
        z_offset = bHc[2, 3]

        fx = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["fx"]
        fy = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["fy"]
        cx = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["cx"]
        cy = data_config[dataset_name]["camera_matrics"]["camera_matrix"]["cy"]
        K = gen_camera_matrix(fx, fy, cx, cy)

        if False: # We consider distortion free cam only for now
            k1 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["k1"]
            k2 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["k2"]
            p1 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["p1"]
            p2 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["p2"]
            k3 = data_config[dataset_name]["camera_matrics"]["dist_coeffs"]["k3"]
            dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

        #for i, waypts in enumerate(list_waypts): # way_points
            #xy_coords = traj[:, :2]    # N x 2    (horizon, 2)
        if len(pred_waypts.shape) == 1: # is 1 if we care about only one waypoint. i.e, the point btwn start_pos and goal_pos
            # add a dimension to the front of point
            pred_waypts = pred_waypts[None, ...]

        num_pts = pred_waypts.shape[0]
        xy   = pred_waypts[:, :2].transpose()  # (2,) 2 x N
        N    = xy.shape[1]
        xy_b = xy.copy()
        z_h  = np.ones([2, N])
        z_h[0] = z_h[0] * 0 #z_offset
        xyz_h_b = np.concatenate( [xy_b, z_h], axis=0 )         # xyz wrt b
        xyz_c = np.matmul( cHb[:3,...], xyz_h_b ).squeeze()     # xyz wrt c
        #assert( not( xyz_c[0] == 0 and xyz_c[0] == 0 and xyz_c[0] == 0) )
        uv_pred = rm.pinhole_projection(K, xyz_c)

        # if len(label_waypts.shape) == 1: # is 1 if we care about only one waypoint. i.e, the point btwn start_pos and goal_pos
        #     # add a dimension to the front of point
        #     label_waypts = label_waypts[None, ...]
        #
        # xy   = label_waypts[:, :2].transpose()    # (2,) 2 x N
        # N    = xy.shape[1]
        # xy_b = xy.copy()
        # z_h  = np.ones([2, N])
        # z_h[0] = z_h[0] * 0 #z_offset
        # xyz_h_b = np.concatenate( [xy_b, z_h], axis=0 )         # xyz wrt b
        # xyz_c = np.matmul( cHb[:3,...], xyz_h_b ).squeeze()     # xyz wrt c
        # uv_gt = rm.pinhole_projection(K, xyz_c)

        # We assume distortion free cam for now...
        ax.plot(
            uv_pred[0,:] / u_scale,  # u
            uv_pred[1,:] / v_scale,  # v
            color=waypts_colors[0],
            marker='D',
            markersize=8.0,
        )

        for i, point in enumerate(list_points):
            if len(point.shape) == 1:
                # add a dimension to the front of point
                point = point[None, :2]
            else:
                point = point[:, :2]

            xy   = point.transpose() # 2 x N
            N    = xy.shape[1]
            xy_b = xy.copy()
            z_h  = np.ones([2, N])
            z_h[0] = z_h[0] * 0
            xyz_h_b = np.concatenate( [xy_b, z_h], axis=0 )         # xyz wrt b
            xyz_c = np.matmul( cHb[:3,...], xyz_h_b ).squeeze()     # xyz wrt c
            assert( not( xyz_c[0] == 0 and xyz_c[0] == 0 and xyz_c[0] == 0) )
            uv = rm.pinhole_projection(K, xyz_c)

            ax.plot(
                uv[0] / u_scale,
                uv[1] / v_scale,
                color=point_colors[i],
                marker="o",
                markersize=10.0,
            )

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, W - 0.5))
        ax.set_ylim((H - 0.5, 0.5))


def plot_pred_cam_heading_and_points_simple(
        ax0: plt.Axes,
        ax1: plt.Axes,
        waypoint_pred: list,    # wpts
        list_points: list,      # start & goal pts (pose)
        waypoint_color: CYAN,
        point_colors: list = [GREEN, RED],
        # waypoint_labels: Optional[list] = ["prediction", "ground truth"],
        point_labels: Optional[list] = ["robot", "goal"],
        waypoint_alphas: Optional[list] = None,
        point_alphas: Optional[list] = None,
        quiver_freq: int = 1,
        default_coloring: bool = True,
        dataset_name: str = 'former',
        collision_status: bool = False,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_waypoints: [ pred  and  gt ] waypoint, each waypoint is a numpy array of shape (2,) (if there is no yaw) or (4,) (if there is yaw)
        list_points: list of points: [start point and goal point], each point is a numpy array of shape (2,)
        waypoint_colors: list of colors for waypoints: waypoints[0]= pred (CYAN),  waypoints[1]= GT (MAGENTA)
        point_colors: list of colors for points:  start = RED,  goal = GREEN
        waypoint_labels: list of labels for waypoints
        point_labels: list of labels for points
        waypoint_alphas: list of alphas for waypoints
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    global prev_waypoint_cb_time, curr_waypoint_cb_time
    prev_waypoint_cb_time = curr_waypoint_cb_time
    curr_waypoint_cb_time = time.time()
    elapsed = curr_waypoint_cb_time - prev_waypoint_cb_time

    waypoint_pred = np.atleast_2d(waypoint_pred)  # (len_pred, num_param)    # x, y

    #################################################################################
    #               base_link configuration
    #
    #                         ^  (X)
    #                         |
    #                         |
    #                         |
    #               Y <------(+) (Z)
    #################################################################################

    # swap X and Y and use -Y instead of Y to represent x,y based on the base_link config

    (len_pred, num_params) = waypoint_pred.shape
    assert (num_params == 2 or num_params == 4)
    # if waypoint_labels is None:
    ax0.plot(
        -waypoint_pred[:, 1],  # y  (negative y b/c y points to the right (see the base_link conf above)
        waypoint_pred[:, 0],  # x
        color=waypoint_color,
        alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
        marker="D",
    )

    if nav_config['learn_angle'] == True:  # len(waypoint_pred) == 4:
        for ii in range(0, len_pred):
            # print(len(waypoint_pred[ii])) is 4
            bearing_pred = gen_bearing(waypoint_pred[ii])  # draw heading dir
            ax0.quiver(
                -waypoint_pred[ii, 1],  # y
                waypoint_pred[ii, 0],  # x
                -bearing_pred[1],
                bearing_pred[0],
                color=waypoint_color * 0.5,
                scale=1.0,
            )

        goal = list_points[1]
        bearing_goal = gen_bearing(goal)
        ax0.quiver(
            -goal[1],  # y
            goal[0],  # x
            -bearing_goal[1],
            bearing_goal[0],
            color=point_colors[1] * 0.5,
            scale=1.0,
        )

    start = list_points[0]
    goal = list_points[1]
    if point_labels is None:
        ax0.plot(
            -start[1],  # start cam x wrt base_link
            start[0],  # start cam z wrt base_link
            color=point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0
        )
        for goal in goal_points:
            ax0.plot(
                -goal[1],  # goal pos wrt curr base_link
                goal[0],
                color=point_colors[1],
                alpha=point_alphas[1] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
    else:
        ax0.plot(
            -start[1],
            start[0],
            color=point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0,
            label=point_labels[0],
        )
        for goal in goal_points:
            ax0.plot(
                -goal[1],
                goal[0],
                color=point_colors[1],
                alpha=point_alphas[1] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[1],
            )
        ax0.grid(True)

    # put the legend below the plot
    # if waypoint_labels is not None or point_labels is not None:
    # ax.legend(loc='lower right', shadow=True, fontsize='medium')

    print(waypoint_pred[..., 0])
    print(waypoint_pred[..., 1])

    max_x = max(abs(goal[0]), np.max(np.abs(waypoint_pred[..., 0])))
    max_y = max(abs(goal[1]), np.max(np.abs(waypoint_pred[..., 1]))) + 0.1
    max_x += 0.2 * max_x
    max_y += 0.2 * max_y
    # max_x = max(max_x, 1)

    waypoint_speed = data_config[dataset_name]['max_speed'] / data_config[dataset_name]['img_fps'] * \
                     nav_config['datasets'][dataset_name]['waypoint_spacing']
    goal_dist_max = waypoint_speed * nav_config['distance']['max_frame_dist']

    if nav_config['normalize'] is True:
        plot_xlim = max(goal[1], 1.1)  #1.0       # xy swapped
        plot_ylim = max(goal[0], 1.1)  #1.2       #goal_dist_max / nav_config['datasets'][dataset_name]['waypoint_spacing'] + 0.1
    else:
        plot_xlim = max(goal[1], 1.1)  #3.       # xy swapped for drawing
        plot_ylim = max(goal[0], 1.1)  #5. #3.4

    plot_xlim = max(1.0, plot_xlim) + 0.1
    plot_ylim = max(1.0, plot_ylim) + 0.1

    ax0.set_xlim([-plot_xlim, plot_xlim])
    ax0.set_ylim([-plot_ylim, plot_ylim])
    ax0.set_aspect("equal", "box")

    # X axis (horizontal) shows robot y (left positive) => flip sign in labels
    ax0.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{-v:.1f}"))
    # Y axis (vertical) already corresponds to robot x (forward positive)
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))
    ax0.set_xlabel('Y-axis (base_link)')
    ax0.set_ylabel('X-axis (base_link)')

    # Put Y ticks on the right if you like
    ax0.yaxis.tick_right()
    ax0.yaxis.set_label_position('right')
    ax0.tick_params(axis='y', labelright=True, labelleft=False)

    textstr0 = ('\n'.join((
        r'Wp pub period %.3f (ms)' % (elapsed * 1000),
        r'pub freq: %.3f (Hz)' % (1./elapsed),
        r'goal(xy) = (%.2f, %.2f)' % (goal[0], goal[1]),  # x y
        r'curr=(%.2f, %.2f)' % (start[0], start[1]),
        r'collision: %s' % collision_status )))

    # last_waypts_combined = np.concatenate( (waypoint_pred[...,:2], waypoint_gt[...,:2]), axis = 1 )
    textstr1 = 'xp, yp\n'
    for idx, row in enumerate(waypoint_pred[..., :2]):
        (xp, yp) = row
        tmp = f'%d: %.2f, %.2f\n' % (idx, xp, yp)
        textstr1 += tmp

    textstr = '\n'.join((textstr0, textstr1))
    text_color = "red" if collision_status else "black"
    ax1.text(
        0.01, 0.99, textstr,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        color = text_color,
    )


def plot_pred_cam_heading_and_points(
        ax: plt.Axes,
        waypoint_pred: list,       # wpts
        list_points: list,          # start & goal pts (pose)
        waypoint_color: CYAN,
        point_colors: list = [GREEN, RED],
        #waypoint_labels: Optional[list] = ["prediction", "ground truth"],
        point_labels: Optional[list] = ["robot cent", "goal"],
        waypoint_alphas: Optional[list] = None,
        point_alphas: Optional[list] = None,
        quiver_freq: int = 1,
        default_coloring: bool = True,
        dataset_name: str = 'former',
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_waypoints: [ pred  and  gt ] waypoint, each waypoint is a numpy array of shape (2,) (if there is no yaw) or (4,) (if there is yaw)
        list_points: list of points: [start point and goal point], each point is a numpy array of shape (2,)
        waypoint_colors: list of colors for waypoints: waypoints[0]= pred (CYAN),  waypoints[1]= GT (MAGENTA)
        point_colors: list of colors for points:  start = RED,  goal = GREEN
        waypoint_labels: list of labels for waypoints
        point_labels: list of labels for points
        waypoint_alphas: list of alphas for waypoints
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """

    waypoint_pred = np.atleast_2d(waypoint_pred)   # (len_pred, num_param)    # x, y
    
    #################################################################################
    #               base_link configuration
    #
    #                         ^  (X)
    #                         |
    #                         |
    #                         |
    #               Y <------(+) (Z)
    #################################################################################

    # swap X and Y and use -Y instead of Y to represent x,y based on the base_link config

    (len_pred, num_params) = waypoint_pred.shape
    assert(num_params == 2 or num_params == 4)
    #if waypoint_labels is None:
    ax.plot(
        -waypoint_pred[:, 1],  # y  (negative y b/c y points to the right (see the base_link conf above)
        waypoint_pred[:, 0],   # x
        color=waypoint_color,
        alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
        marker="D",
    )
    #else:
        #ax.plot(
            #-waypoint_pred[:, 1],
            #waypoint_pred[:, 0],  # traj[:, 1],
            #color=waypoint_colors[0],
            #label=waypoint_labels[0],
            #alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            #marker="D",
        #)
        # if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
    if nav_config['learn_angle'] == True: #len(waypoint_pred) == 4:
        # if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
        # bearings = gen_bearings_from_waypoints(traj)
        # ax.quiver(
        #     traj[::quiver_freq, 0],
        #     traj[::quiver_freq, 1],
        #     bearings[::quiver_freq, 0],
        #     bearings[::quiver_freq, 1],
        #     color=traj_colors[i] * 0.5,
        #     scale=1.0,
        # )
        for ii in range(0, len_pred):
            #print(len(waypoint_pred[ii])) is 4
            bearing_pred = gen_bearing(waypoint_pred[ii])  # draw heading dir
            ax.quiver(
                -waypoint_pred[ii,1],  # y
                waypoint_pred[ii,0],  # x
                -bearing_pred[1],
                bearing_pred[0],
                color=waypoint_color * 0.5,
                scale=1.0,
            )

        goal  = list_points[1]
        bearing_goal      = gen_bearing(goal)
        ax.quiver(
            -goal[1],  # y
            goal[0],  # x
            -bearing_goal[1],
            bearing_goal[0],
            color=point_colors[1] * 0.5,
            scale=1.0,
        )

    #for i, pt in enumerate(list_points):
    start = list_points[0]
    goal  = list_points[1]
    if point_labels is None:
        ax.plot(
            -start[1], # start cam x wrt base_link
            start[0], # start cam z wrt base_link
            color=point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0
        )
        ax.plot(
            -goal[1], # goal pos wrt curr base_link
            goal[0],
            color=point_colors[1],
            alpha=point_alphas[1] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0
        )
    else:
        ax.plot(
            -start[1],
            start[0],
            color=point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0,
            label=point_labels[0],
        )
        ax.plot(
            -goal[1],
            goal[0],
            color=point_colors[1],
            alpha=point_alphas[1] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0,
            label=point_labels[1],
        )
        ax.grid(True)

    # put the legend below the plot
    #if waypoint_labels is not None or point_labels is not None:
        #ax.legend(loc='lower right', shadow=True, fontsize='medium')

    max_x = max(abs(goal[0]), max(waypoint_pred[..., 0]))
    max_x += 0.2 * max_x

    max_y = max(abs(goal[1]), max(waypoint_pred[..., 1])) + 0.1
    max_y += 0.2 * max_y
    #max_x = max(max_x, 1)

    ax.set_aspect("equal", "box")
    ax.set_xlabel('Y-axis')
    ax.set_ylabel('X-axis')

    waypoint_speed = data_config[dataset_name]['max_speed'] / data_config[dataset_name]['img_fps'] * nav_config['datasets'][dataset_name]['waypoint_spacing']
    goal_dist_max = waypoint_speed * nav_config['distance']['max_frame_dist']
    if nav_config['normalize'] is True:
        plot_xlim = 1.       # xy swapped
        plot_ylim = 0.7 #goal_dist_max / nav_config['datasets'][dataset_name]['waypoint_spacing'] + 0.1
    else:
        plot_xlim = 3.       # xy swapped for drawing
        plot_ylim = 5. #3.4

    ax.set_xlim([-plot_xlim, plot_xlim])
    ax.set_ylim([-plot_ylim + 0.5, plot_ylim])

    textstr0 = '\n'.join((
        r'goal(xy) = (%.2f, %.2f)'% (goal[0], goal[1]), # x y
        r'curr=(%.2f, %.2f)'% (start[0], start[1]), ) )

    #last_waypts_combined = np.concatenate( (waypoint_pred[...,:2], waypoint_gt[...,:2]), axis = 1 )
    textstr1 = 'xp, yp\n'
    for idx, row in enumerate(waypoint_pred[...,:2]):
        (xp,yp) = row
        tmp = f'%.2f, %.2f\n'%(xp,yp)
        textstr1 +=tmp

    textstr = '\n'.join( (textstr0, textstr1) )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.0, 1.1, textstr, transform=ax.transAxes, fontsize=10,
             horizontalalignment='left', verticalalignment='bottom', bbox=props)



def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])

def gen_bearing(
    waypoint: np.ndarray,
    mag=0.1,
) -> np.ndarray:
    """Generate bearings from a point, (x, y, qs, qz)."""
    bearing = []
    #print(waypoint.shape)   # (len_pred, num_param)
    assert( len(waypoint.shape) == 1 )
    if len(waypoint) == 4:  # label is qs qz  represent

        qs = waypoint[2]
        qz = waypoint[3]
        quat = np.array([qs, 0.0, 0.0, qz])
        htm = rm.quat_to_htm(quat)
        x, y, z, rol, pit, yaw = rm.htm_to_xyzrpy(htm)
        assert( abs(rol) % 3.1415 < 0.015 and abs(pit) % 3.1415 < 0.015), f"rol is {rol} and pit is {pit} while they must be zeros"
        #pit_wrt_camz = -pit + math.pi/2.0   # negative if pitch axis goes down in cam frame
        yaw_wrt_basex = yaw  # positive since z goes up / negative if pitch axis goes down in cam frame
        v = mag * angle_to_unit_vector( yaw_wrt_basex ) # pit_wrt_camz) # Only 2D cam motion !!
    else:  # label is radians repr

        yaw = waypoint[2]
        yaw_norm = rm.normalizeAngle(yaw)
        yaw_wrt_basex = yaw_norm  # positive since z goes up / negative if pitch axis goes down in cam frame
        v = mag * angle_to_unit_vector( yaw_wrt_basex )
        
        #raise NotImplementedError
        #v = mag * angle_to_unit_vector(waypoint[2])
    # bearing.append(v)
    # bearing = np.array(bearing)
    return v #out_bearing #

#def gen_bearings_from_waypoints(
    #waypoints: np.ndarray,
    #mag=0.2,
#) -> np.ndarray:
    #"""Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    #bearing = []
    #print(waypoints)
    #print(waypoints.shape)
    #for i in range(0, len(waypoints)):
        #if waypoints.shape[1] > 3:  # label is sin/cos repr
            #v = waypoints[i, 2:]
            #normalize v
            #v = v / np.linalg.norm(v)
            #v = v * mag
        #else:  # label is radians repr
            #v = mag * angle_to_unit_vector(waypoints[i, 2])
        #bearing.append(v)
    #bearing = np.array(bearing)
    #return bearing
    
def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        cx: principal point x coordinate
        cy: principal point y coordinate
    Returns:
        camera matrix
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
