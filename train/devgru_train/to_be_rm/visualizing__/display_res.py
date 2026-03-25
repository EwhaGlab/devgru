import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List
import wandb
import yaml
import rigid_motion as rm
import torch
import torch.nn as nn
import math

from depth_nav_train.visualizing.visualize_utils import (
    to_numpy,
    numpy_to_depth,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)

with open("/home/hankm/python_ws/viznav/depth-nav/train/depth_nav_train/data/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

def plot_trajs_and_points_on_depth(
    ax: plt.Axes,
    depth: np.ndarray,
    #dataset_name: str,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
):
    """
    Plot trajectories and points on an image. If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
    Args:
        ax: matplotlib axis
        img: image to plot
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
    """
    assert len(list_trajs) <= len(traj_colors), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    #assert (
        #dataset_name in data_config
    #), f"Dataset {dataset_name} not found in data/data_config.yaml"
    dataset_name = 'thud'
    ax.imshow(depth)
    if (
        "camera_metrics" in data_config[dataset_name]
        and "camera_height" in data_config[dataset_name]["camera_metrics"]
        and "camera_matrix" in data_config[dataset_name]["camera_metrics"]
        and "dist_coeffs" in data_config[dataset_name]["camera_metrics"]
    ):
        camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
        camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]

        fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
        fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
        cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
        cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
        camera_matrix = gen_camera_matrix(fx, fy, cx, cy)

        k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
        k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
        p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
        p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
        k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
        dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

        for i, traj in enumerate(list_trajs):
            xy_coords = traj[:, :2]  # (horizon, 2)
            traj_pixels = get_pos_pixels(
                xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
            )
            if len(traj_pixels.shape) == 2:
                ax.plot(
                    traj_pixels[:250, 0],
                    traj_pixels[:250, 1],
                    color=traj_colors[i],
                    lw=2.5,
                )

        for i, point in enumerate(list_points):
            if len(point.shape) == 1:
                # add a dimension to the front of point
                point = point[None, :2]
            else:
                point = point[:, :2]
            pt_pixels = get_pos_pixels(
                point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=True
            )
            ax.plot(
                pt_pixels[:250, 0],
                pt_pixels[:250, 1],
                color=point_colors[i],
                marker="o",
                markersize=10.0,
            )
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))



def plot_heading_and_points(
        fig_title: str,
        ax: plt.Axes,
        list_waypoints: list,
        list_points: list,
        waypoint_colors: list = [CYAN, MAGENTA],
        point_colors: list = [RED, GREEN],
        waypoint_labels: Optional[list] = ["pred", "GT"],
        point_labels: Optional[list] = ["robot", "goal"],
        waypoint_alphas: Optional[list] = None,
        point_alphas: Optional[list] = None,
        quiver_freq: int = 1,
        default_coloring: bool = True,
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
    assert (
            len(list_waypoints) <= len(waypoint_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
            waypoint_labels is None or len(list_waypoints) == len(waypoint_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    waypoint_pred = list_waypoints[0]  # 2
    waypoint_gt = list_waypoints[1]

    if waypoint_labels is None:
        ax.plot(
            waypoint_pred[0],
            waypoint_pred[1],
            color=waypoint_colors[0], # pred is CYAN
            alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            marker="s",
        )
        ax.plot(
            waypoint_gt[0],
            waypoint_gt[1],
            color=waypoint_colors[1], # GT is MAGENTA
            alpha=waypoint_alphas[1] if waypoint_alphas is not None else 1.0,
            marker="o",
        )
    else:
        ax.plot(
            waypoint_pred[0],
            waypoint_pred[1],  # traj[:, 1],
            color=waypoint_colors[0],
            label=waypoint_labels[0],
            alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            marker="o",
        )
        ax.plot(
            waypoint_gt[0],
            waypoint_gt[1],  # traj[:, 1],
            color=waypoint_colors[1],
            label=waypoint_labels[1],
            alpha=waypoint_alphas[1] if waypoint_alphas is not None else 1.0,
            marker="o",
        )
        # if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
        if len(waypoint_pred) == 4:
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
            bearing_pred    = gen_bearing(waypoint_pred)  # draw heading dir
            ax.quiver(
                waypoint_pred[0],  # x
                waypoint_pred[1],  # z
                bearing_pred[0],
                bearing_pred[1],
                color=waypoint_colors[0] * 0.5,
                scale=1.0,
            )
            bearing_gt      = gen_bearing(waypoint_gt)
            ax.quiver(
                waypoint_gt[0],  # x
                waypoint_gt[1],  # z
                bearing_gt[0],
                bearing_gt[1],
                color=waypoint_colors[1] * 0.5,
                scale=1.0,
            )
            
            goal  = list_points[1]
            bearing_goal      = gen_bearing(goal)
            ax.quiver(
                goal[0],  # x
                goal[1],  # y
                bearing_goal[0],
                bearing_goal[1],
                color=point_colors[1] * 0.5,
                scale=1.0,
            )
            
    #for i, pt in enumerate(list_points):
    start = list_points[0]
    goal  = list_points[1]
    if point_labels is None:
        ax.plot(
            start[0],
            start[1],
            color=point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0
        )
        ax.plot(
            goal[0],
            goal[1],
            color=point_colors[1],
            alpha=point_alphas[1] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0
        )
    else:
        ax.plot(
            start[0],
            start[1],
            color=point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="s",
            markersize=9.0,
            label=point_labels[0],
        )
        ax.plot(
            goal[0],
            goal[1],
            color=point_colors[1],
            alpha=point_alphas[1] if point_alphas is not None else 1.0,
            marker="*",
            markersize=7.0,
            label=point_labels[1],
        )

    # put the legend below the plot
    if waypoint_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.25), loc="upper left", ncol=2)
    #min_x = min( min( abs(goal[0]), abs(  waypoint_gt[0] ) ), waypoint_pred[0] ) - 0.2
    max_x = max( max( abs(goal[0]), abs(  waypoint_gt[0] ) ), waypoint_pred[0] ) 
    max_x += 0.2 * max_x
    max_x = max(max_x, 1)
    
    #min_z = min( min( abs(goal[1]), abs(  waypoint_gt[1] ) ), waypoint_pred[1] ) - 0.1
    max_z = max( max( abs(goal[1]), abs(  waypoint_gt[1] ) ), waypoint_pred[1] ) + 0.1
    max_z += 0.2 * max_z
    max_z = max(max_z, 1)
    
    ax.set_aspect("equal", "box")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Z-axis')
    ax.set_xlim([-max_x, max_x ])
    ax.set_ylim([-max_z, max_z ])
    ax.text(2,-1.2, f'goal:             (%f, %f)'%(goal[0], goal[1]), fontsize = 12 )
    ax.text(2,-1.4, f'xy waypoint gt:   (%f, %f)'%(waypoint_gt[0], waypoint_gt[1]), fontsize = 12 )
    ax.text(2,-1.6, f'xy waypoint pred: (%f, %f)'%(waypoint_pred[0], waypoint_pred[1]), fontsize = 12 )


def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])


def gen_bearing(
    waypoint: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
#    print(waypoints.shape)  (2,)
    if len(waypoint) == 4:  # label is qs qz  represent
        qs = waypoint[2]
        qy = waypoint[3]
        quat = np.array([qs, 0.0, qy, 0.0])
        htm = rm.quat_to_htm(quat)
        x, y, z, rol, pit, yaw = rm.htm_to_xyzrpy(htm)
        assert( abs(yaw) % 3.1415 < 0.015 and abs(rol) % 3.1415 < 0.015), f"rol is {rol} and yaw is {yaw} while they must be zeros"
        pit_wrt_camz = -pit + math.pi/2.0  # negative b/c pitch axis goes down in cam frame
        v = mag * angle_to_unit_vector(pit_wrt_camz) # Only 2D cam motion !!
        # v = waypoint[2:]
        # # normalize v
        # v = v / np.linalg.norm(v)
        # v = v * mag
    else:  # label is radians repr
        raise NotImplementedError
        #v = mag * angle_to_unit_vector(waypoint[2])
    # bearing.append(v)
    # bearing = np.array(bearing)
    return v #bearing



def compare_pred_to_label(
    fig_title: str,
    obs_depth, #obs_img,
    goal_depth, #goal_img,
    #dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoint: np.ndarray,
    label_waypoint: np.ndarray,
   # save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization.

    Args:
        obs_img: image of the observation
        goal_img: image of the goal
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        goal_pos: goal position in the image
        pred_waypoints: predicted waypoints in the image
        label_waypoints: label waypoints in the image
        save_path: path to save the figure
        display: whether to display the figure
    """

    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0, 0, 0])
    fig.suptitle(fig_title, fontsize = 12)

    if len(pred_waypoint.shape) > 2:
        list_waypoints = [*pred_waypoint, label_waypoint]
    else:
        list_waypoints = [pred_waypoint, label_waypoint]
    #plot_trajs_and_points(
    plot_heading_and_points(
        fig_title,
        ax[0],
        list_waypoints,  # scale up for vizualization
        [start_pos, goal_pos],
        waypoint_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    #plot_trajs_and_points_on_image(
    plot_trajs_and_points_on_depth(
        ax[1],
        obs_depth, #obs_img,
        #dataset_name = 'thud',
        list_waypoints,  # scale up for vizualization
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    ax[2].imshow(goal_depth)#goal_img)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    #if save_path is not None:
        #fig.savefig(
            #save_path,
            #bbox_inches="tight",
        #)

    #if not display:
        #plt.close(fig)
    fig.show()
    plt.pause(0.001)
    input("Press Enter to continue")
    plt.close(fig)

def visualize_action_pred(
    batch_data_info: str,
    batch_obs_depths: np.ndarray,
    batch_goal_depth: np.ndarray,
    #dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_pred_waypoint: np.ndarray,
    batch_label_waypoint: np.ndarray,
    #eval_type: str,
    normalized: bool,
    #save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = False,
    #display: bool = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization. This visualization is for the last batch in the dataset.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
        dataset_names: indices corresponding to the dataset name
        batch_goals (np.ndarray): batch of goal positions [batch_size, 2]
        batch_pred_waypoint (np.ndarray): batch of predicted waypoints [batch_size, horizon, 4] or [batch_size, horizon, 2] or [batch_size, num_trajs_sampled horizon, {2 or 4}]
        batch_label_waypoint (np.ndarray): batch of label waypoints [batch_size, T, 4] or [batch_size, horizon, 2]
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        normalized (bool): whether the waypoints are normalized
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
    """
    #visualize_path = None
    #if save_folder is not None:
        #visualize_path = os.path.join(
            #save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        #)

    #if not os.path.exists(visualize_path):
        #os.makedirs(visualize_path)

    assert (
        len(batch_obs_depths)
        == len(batch_goal_depth)
        == len(batch_goals)
        == len(batch_pred_waypoint)
        == len(batch_label_waypoint)
    )

    #dataset_names = list(data_config.keys())
    #dataset_names.sort()

    batch_size = batch_obs_depths.shape[0]
    wandb_list = []
    for i in range(0,100): #range(min(batch_size, num_images_preds)):
        fig_title = f'(%d)|  %s'%(i, batch_data_info[i])
        obs_depth = numpy_to_depth(batch_obs_depths[i])
        goal_depth = numpy_to_depth(batch_goal_depth[i])
        #dataset_name = dataset_names[int(dataset_indices[i])]
        goal_pos = batch_goals[i]
        pred_waypoint = batch_pred_waypoint[i]
        label_waypoint = batch_label_waypoint[i]
        
        ################################################################################
        # TODO: We need to come up with a better approaches for normalization !!!
        # if normalized:
        #     pred_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
        #     label_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
        #     goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

        #save_path = None
        #if visualize_path is not None:
            #save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        #compare_waypoints_pred_to_label(
        compare_pred_to_label(
            fig_title,
            obs_depth,
            goal_depth,
            #dataset_name,
            goal_pos,
            pred_waypoint,
            label_waypoint,
            #save_path,
            display,
        )

        
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    #if use_wandb:
        #wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)
        


def main():
    
    base_path = '/home/hankm/matlab_ws/DepthNav/log'
    epoch  = 80
    i   = 80
    
    log_pkl_file = f'{base_path}/data%03d.pkl'%epoch
    
    with open(log_pkl_file, 'rb') as fp:
        data = pickle.load(fp)
    
    batch_data_info   = data['data_info']
    batch_obs_depths  = data['obs_depths'].squeeze()
    batch_goal_depth  = data['goal_depth'].squeeze()
    batch_dist_pred   = data['dist_pred'].squeeze()
    batch_dist_label  = data['dist_label'].squeeze()
    batch_goals    = data['goal_pos'].squeeze()
    batch_pred_waypoint = data['action_pred'].squeeze()
    batch_label_waypoint= data['action_label'].squeeze()
    
    
    visualize_action_pred(
    batch_data_info  = batch_data_info,
    batch_obs_depths = batch_obs_depths,
    batch_goal_depth = batch_goal_depth,
    #dataset_indices  = 
    batch_goals =  batch_goals,
    batch_pred_waypoint= batch_pred_waypoint,
    batch_label_waypoint= batch_label_waypoint,
    #eval_type: str,
    normalized = False,
    #save_folder: str,
    epoch = epoch,
    num_images_preds = 8,
    use_wandb = False,
    #display: bool = False,.
    )
    
    

if __name__ == "__main__":
    
    
    main(config)
