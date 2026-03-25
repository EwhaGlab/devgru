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
from matplotlib.gridspec import GridSpec

from depth_nav_train.visualizing.visualize_utils import (
    to_numpy,
    numpy_to_img,
    numpy_to_depth,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)

X_VIZ_START_OFFSET = 0.6    # offset to xdir of base_link pose. We add this num to present the start pt in the image (hard coded for isaac sim)

# load data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

with open(os.path.join(os.path.dirname(__file__), "../../config/depth_nav.yaml"), "r") as f:
    nav_config = yaml.safe_load(f)


def visualize_action_and_dist_pred(
    batch_data_info: tuple,
    np_batch_obs_images: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
    np_batch_goal_image: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
    np_batch_obs_depths: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
    np_batch_goal_depth: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_pred_waypoint: np.ndarray,    # action_pred ( base_link on xy plane )
    batch_label_waypoint: np.ndarray,   # action_label
  #  batch_dist_pred: np.ndarray,
  #  batch_dist_label: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization. This visualization is for the last batch in the dataset.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]   (height, width) = (160, 120)
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
    print(save_folder)
    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    assert (
        len(np_batch_obs_depths)
        == len(np_batch_goal_depth)
        == len(np_batch_obs_images)
        == len(np_batch_goal_image)
        == len(batch_goals)
        == len(batch_pred_waypoint)
        == len(batch_label_waypoint)
    )

    dataset_names = list(data_config.keys())
    dataset_names.sort()

    batch_size = np_batch_obs_images.shape[0] #batch_obs_depths.shape[0]
    wandb_list = []
    text_color = "black"

    for i in range(min(batch_size, num_images_preds)):
        viz_obs_depth  = numpy_to_depth(np_batch_obs_depths[i])
        viz_goal_depth = numpy_to_depth(np_batch_goal_depth[i])
        viz_obs_image = numpy_to_img(np_batch_obs_images[i])       # image resize back to (640, 480)
        viz_goal_image = numpy_to_img(np_batch_goal_image[i])
        dataset_name = dataset_names[int(dataset_indices[i])]
        goal_pos = batch_goals[i]
        pred_waypoint = batch_pred_waypoint[i]
        label_waypoint = batch_label_waypoint[i]
  #      dist_pred = batch_dist_pred[i]
  #      dist_label= batch_dist_label[i]
        info = batch_data_info[i].split(' ')

        # data_info = f'{self.dataset_name} {f_curr} {curr_time} {f_goal} {goal_time} {goal_time - curr_time}'
        fig_title = f"dataset_name: %s \n %s %s \n %s %s \n frame diff: %d \nrel goal: (%0.3f, %0.3f) \n"%\
                                (info[0], info[1], info[2], info[3], info[4], int(info[4]) - int(info[2]), goal_pos[0], goal_pos[1])

        ################################################################################
        # TODO: We need to come up with a better approaches for normalization !!!
        # if normalized:
        #     pred_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
        #     label_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
        #     goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        #compare_waypoints_pred_to_label(
        compare_pred_to_label(
            fig_title,
            viz_obs_depth,
            viz_goal_depth,
            viz_obs_image,   # (640, 480)
            viz_goal_image,  # (640, 480)
            dataset_name,
            goal_pos,
            pred_waypoint,
            label_waypoint,
      #      dist_pred,
      #      dist_label,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)

# def visualize_action_pred(
#     batch_data_info: tuple,
#     np_batch_obs_images: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
#     np_batch_goal_image: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
#     np_batch_obs_depths: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
#     np_batch_goal_depth: np.ndarray,           # img size : VISUALIZATION_IMAGE_SIZE = (160, 120)
#     dataset_indices: np.ndarray,
#     batch_goals: np.ndarray,
#     batch_pred_waypoint: np.ndarray,    # action_pred ( base_link on xy plane )
#     batch_label_waypoint: np.ndarray,   # action_label
#     eval_type: str,
#     normalized: bool,
#     save_folder: str,
#     epoch: int,
#     num_images_preds: int = 8,
#     use_wandb: bool = True,
#     display: bool = False,
# ):
#     """
#     Compare predicted path with the gt path of waypoints using egocentric visualization. This visualization is for the last batch in the dataset.
#
#     Args:
#         batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]   (height, width) = (160, 120)
#         batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
#         dataset_names: indices corresponding to the dataset name
#         batch_goals (np.ndarray): batch of goal positions [batch_size, 2]
#         batch_pred_waypoint (np.ndarray): batch of predicted waypoints [batch_size, horizon, 4] or [batch_size, horizon, 2] or [batch_size, num_trajs_sampled horizon, {2 or 4}]
#         batch_label_waypoint (np.ndarray): batch of label waypoints [batch_size, T, 4] or [batch_size, horizon, 2]
#         eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
#         normalized (bool): whether the waypoints are normalized
#         save_folder (str): folder to save the images. If None, will not save the images
#         epoch (int): current epoch number
#         num_images_preds (int): number of images to visualize
#         use_wandb (bool): whether to use wandb to log the images
#         display (bool): whether to display the images
#     """
#     print(save_folder)
#     visualize_path = None
#     if save_folder is not None:
#         visualize_path = os.path.join(
#             save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
#         )
#
#     if not os.path.exists(visualize_path):
#         os.makedirs(visualize_path)
#
#     assert (
#         len(np_batch_obs_depths)
#         == len(np_batch_goal_depth)
#         == len(np_batch_obs_images)
#         == len(np_batch_goal_image)
#         == len(batch_goals)
#         == len(batch_pred_waypoint)
#         == len(batch_label_waypoint)
#     )
#
#     dataset_names = list(data_config.keys())
#     dataset_names.sort()
#
#     batch_size = np_batch_obs_images.shape[0] #batch_obs_depths.shape[0]
#     wandb_list = []
#     for i in range(min(batch_size, num_images_preds)):
#         viz_obs_depth  = numpy_to_depth(np_batch_obs_depths[i])
#         viz_goal_depth = numpy_to_depth(np_batch_goal_depth[i])
#         viz_obs_image = numpy_to_img(np_batch_obs_images[i])       # image resize back to (640, 480)
#         viz_goal_image = numpy_to_img(np_batch_goal_image[i])
#         dataset_name = dataset_names[int(dataset_indices[i])]
#         goal_pos = batch_goals[i]
#         pred_waypoint = batch_pred_waypoint[i]
#         label_waypoint = batch_label_waypoint[i]
#         info = batch_data_info[i].split(' ')
#         fig_title = f"%s %s \n %s %s \n frame diff %s \nrel goal: (%0.3f, %0.3f)"%( info[0], info[1], info[2], info[3], info[4], goal_pos[0], goal_pos[1] )
#
#         ################################################################################
#         # TODO: We need to come up with a better approaches for normalization !!!
#         # if normalized:
#         #     pred_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
#         #     label_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
#         #     goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]
#
#         save_path = None
#         if visualize_path is not None:
#             save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")
#
#         #compare_waypoints_pred_to_label(
#         compare_pred_to_label(
#             fig_title,
#             viz_obs_depth,
#             viz_goal_depth,
#             viz_obs_image,   # (640, 480)
#             viz_goal_image,  # (640, 480)
#             dataset_name,
#             goal_pos,
#             pred_waypoint,
#             label_waypoint,
#             save_path,
#             display,
#         )
#         if use_wandb:
#             wandb_list.append(wandb.Image(save_path))
#     if use_wandb:
#         wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)

# def visualize_traj_pred(
#     batch_obs_images: np.ndarray,
#     batch_goal_images: np.ndarray,
#     dataset_indices: np.ndarray,
#     batch_goals: np.ndarray,
#     batch_pred_waypoints: np.ndarray,
#     batch_label_waypoints: np.ndarray,
#     eval_type: str,
#     normalized: bool,
#     save_folder: str,
#     epoch: int,
#     num_images_preds: int = 8,
#     use_wandb: bool = True,
#     display: bool = False,
# ):
#     """
#     Compare predicted path with the gt path of waypoints using egocentric visualization. This visualization is for the last batch in the dataset.
#
#     Args:
#         batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
#         batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
#         dataset_names: indices corresponding to the dataset name
#         batch_goals (np.ndarray): batch of goal positions [batch_size, 2]
#         batch_pred_waypoints (np.ndarray): batch of predicted waypoints [batch_size, horizon, 4] or [batch_size, horizon, 2] or [batch_size, num_trajs_sampled horizon, {2 or 4}]
#         batch_label_waypoints (np.ndarray): batch of label waypoints [batch_size, T, 4] or [batch_size, horizon, 2]
#         eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
#         normalized (bool): whether the waypoints are normalized
#         save_folder (str): folder to save the images. If None, will not save the images
#         epoch (int): current epoch number
#         num_images_preds (int): number of images to visualize
#         use_wandb (bool): whether to use wandb to log the images
#         display (bool): whether to display the images
#     """
#     visualize_path = None
#     if save_folder is not None:
#         visualize_path = os.path.join(
#             save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
#         )
#
#     if not os.path.exists(visualize_path):
#         os.makedirs(visualize_path)
#
#     assert (
#         len(batch_obs_images)
#         == len(batch_goal_images)
#         == len(batch_goals)
#         == len(batch_pred_waypoints)
#         == len(batch_label_waypoints)
#     )
#
#     dataset_names = list(data_config.keys())
#     dataset_names.sort()
#
#     batch_size = batch_obs_images.shape[0]
#     wandb_list = []
#     for i in range(min(batch_size, num_images_preds)):
#         obs_image = numpy_to_depth(batch_obs_images[i])
#         goal_image = numpy_to_depth(batch_goal_images[i])
#         dataset_name = dataset_names[int(dataset_indices[i])]
#         goal_pos = batch_goals[i]
#         pred_waypoints = batch_pred_waypoints[i]
#         label_waypoints = batch_label_waypoints[i]
#
#         if normalized:
#             pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
#             label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
#             goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]
#
#         save_path = None
#         if visualize_path is not None:
#             save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")
#
#         compare_waypoints_pred_to_label(
#             #obs_depth,
#             #goal_depth,
#             obs_image,
#             goal_image,
#             dataset_name,
#             goal_pos,
#             pred_waypoints,
#             label_waypoints,
#             save_path,
#             display,
#         )
#         if use_wandb:
#             wandb_list.append(wandb.Image(save_path))
#     if use_wandb:
#         wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)


def compare_pred_to_label(
    fig_title: str,
    viz_obs_depth, #obs_img,
    viz_goal_depth, #goal_img,
    viz_obs_img,
    viz_goal_img,
    dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoint: np.ndarray,
    label_waypoint: np.ndarray,
 #   dist_pred: np.ndarray,
 #   dist_label: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
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
    #fig, ax = plt.subplots(1, 3)
    fig = plt.figure(layout = "constrained")
    fig.suptitle(fig_title, fontsize = 12)

    gs = GridSpec(2, 3, figure = fig)
    ax0 = fig.add_subplot(gs[:,0])
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])

    # We consider cam orig as the start_pos for the visualization
    bHc = rm.xyzrpy_to_htm(data_config[dataset_name]["camera_matrics"]["cam_wrt_base"])
    x_offset = bHc[0, 3] + X_VIZ_START_OFFSET # +0.6 projects uv at the bottom of the (isaac sim)imgs
    y_offset = bHc[1, 3]

    start_pos = np.array([0, 0, 0.0, 0.0])
    start_pos_w_offset = np.array([x_offset, y_offset, 0.0, 0.0])  # start cam pose
    goal_pos_w_offset  = goal_pos.copy()
    goal_pos_w_offset[0] += x_offset
    goal_pos_w_offset[1] += y_offset

    # text_color = "black"
    # dist_comp_txt = ""
    # if (dist_label.ndim > 0):  # xy dist and q dist
    #     if (abs(dist_pred[0] - dist_label[0]) > 0.3 or abs(dist_pred[1] - dist_label[1]) > 0.1):
    #         text_color = "red"
    #         dist_comp_txt = (f"pred dist (xy, theta): {dist_pred[0]} {dist_pred[1]}\n label dist (xy, theta): {dist_label[0]} {dist_label[1]}")
    # else:  # xy only
    #     if abs(dist_pred - dist_label) > 0.3:
    #         text_color = "red"
    #         dist_comp_txt = (f"pred dist: {dist_pred}\nlabel dist: {dist_label}")
    #
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax01.text(0.0, 1.15, dist_comp_txt, transform=ax01.transAxes, fontsize=14,
    #          horizontalalignment='left', verticalalignment='bottom', bbox=props, color=text_color)


    if len(pred_waypoint.shape) > 2:
        raise Exception(" Must be only single way point \n")
        #list_waypoints = [*pred_waypoint, label_waypoint]
    else:
        list_waypoints = [pred_waypoint, label_waypoint]
        np_pred_waypoint_w_offset = pred_waypoint.copy()
        np_label_waypoint_w_offset = label_waypoint.copy()
        np_pred_waypoint_w_offset[0] += x_offset
        np_pred_waypoint_w_offset[1] += y_offset
        np_label_waypoint_w_offset[0] += x_offset
        np_label_waypoint_w_offset[1] += y_offset
        #list_waypoints_w_offset = [pred_waypoint_w_offset, label_waypoint_w_offset]

    #plot_trajs_and_points(
    #plot_heading_and_points(
    plot_cam_heading_and_points(
        ax0, #ax[0][0],
        list_waypoints,  # scale up for vizualization
        [start_pos, goal_pos],      # wrt base
        waypoint_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )

    if dataset_name == 'former' or dataset_name == 'isaac_sim':
       plot_projected_trajs_and_points_on_image(
            ax01, #ax[0][1],
            viz_obs_img,    # (640, 480)
            #viz_obs_depth,
            dataset_name,
            np_pred_waypoint_w_offset,
            np_label_waypoint_w_offset,
            [start_pos_w_offset, goal_pos_w_offset],
            waypts_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
       )
    elif dataset_name == 'thud':
        raise Exception("Not Implemented for 6d pose vec ")
       # plot_trajs_and_points_on_image(
       #      ax[1],
       #      viz_obs_img,
       #      #obs_depth,
       #      dataset_name,
       #      list_waypoints,  # scale up for vizualization
       #      [start_pos_, goal_pos_],
       #      traj_colors=[CYAN, MAGENTA],
       #      point_colors=[GREEN, RED],
       # )
    else:
        raise Exception("Unknown dataset_name <%s>"%dataset_name)

    ax02.imshow(viz_goal_img)
    ax11.imshow(viz_obs_depth,  cmap='gray', vmin=0, vmax=255)
    ax12.imshow(viz_goal_depth, cmap='gray', vmin=0, vmax=255)

    fig.set_size_inches(18.5, 10.5)
    ax0.set_title(f"Action Prediction")
    ax01.set_title(f"Observation")
    ax02.set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    if not display:
        plt.close(fig)

# def compare_waypoints_pred_to_label(
#     obs_img,
#     goal_img,
#     dataset_name: str,
#     goal_pos: np.ndarray,
#     pred_waypoints: np.ndarray,
#     label_waypoints: np.ndarray,
#     save_path: Optional[str] = None,
#     display: Optional[bool] = False,
# ):
#     """
#     Compare predicted path with the gt path of waypoints using egocentric visualization.
#
#     Args:
#         obs_img: image of the observation
#         goal_img: image of the goal
#         dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
#         goal_pos: goal position in the image
#         pred_waypoints: predicted waypoints in the image
#         label_waypoints: label waypoints in the image
#         save_path: path to save the figure
#         display: whether to display the figure
#     """
#
#     fig, ax = plt.subplots(1, 3)
#     start_pos = np.array([0, 0])
#     print("pred wpts", pred_waypoints)
#     print("label wpts:", label_waypoints)
#
#     if len(pred_waypoints.shape) > 2:
#         trajs = [*pred_waypoints, label_waypoints]
#     else:
#         trajs = [pred_waypoints, label_waypoints]
#     plot_heading_and_points(
#         ax[0],
#         trajs,
#         [start_pos, goal_pos],
#         traj_colors=[CYAN, MAGENTA],
#         point_colors=[GREEN, RED],
#     )
#     plot_trajs_and_points_on_image(
#         ax[1],
#         obs_img,
#         dataset_name,
#         trajs,
#         [start_pos, goal_pos],
#         traj_colors=[CYAN, MAGENTA],
#         point_colors=[GREEN, RED],
#     )
#     ax[2].imshow(goal_img)
#
#     fig.set_size_inches(18.5, 10.5)
#     ax[0].set_title(f"Action Prediction")
#     ax[1].set_title(f"Observation")
#     ax[2].set_title(f"Goal")
#
#     if save_path is not None:
#         fig.savefig(
#             save_path,
#             bbox_inches="tight",
#         )
#
#     if not display:
#         plt.close(fig)


def plot_projected_trajs_and_points_on_image(
    ax: plt.Axes,  # ax01
    viz_obs_img: np.ndarray,
    #viz_depth: np.ndarray,
    dataset_name: str,
    pred_waypts: np.ndarray,
    label_waypts: np.ndarray,
    list_points: list,      # [ start_pos,      goal_pos]  wrt robot base frame
    waypts_colors: list = [CYAN, MAGENTA],
    point_colors: list = [GREEN, RED],
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

    ax.imshow(viz_obs_img)  # img size : (640, 480)
    if (
        "camera_matrics" in data_config[dataset_name]
        and "cam_wrt_base" in data_config[dataset_name]["camera_matrics"]       # extrinsic params
        and "camera_matrix" in data_config[dataset_name]["camera_matrics"]      # intrinsic params
        and "dist_coeffs" in data_config[dataset_name]["camera_matrics"]        # distortion coeff
    ):
        img_width = data_config[dataset_name]['img_width']
        img_height= data_config[dataset_name]['img_height']
        u_scale = img_width / VIZ_IMAGE_SIZE[0]  # 640
        v_scale = img_height/ VIZ_IMAGE_SIZE[1]  # 480

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

        if len(label_waypts.shape) == 1: # is 1 if we care about only one waypoint. i.e, the point btwn start_pos and goal_pos
            # add a dimension to the front of point
            label_waypts = label_waypts[None, ...]

        xy   = label_waypts[:, :2].transpose()    # (2,) 2 x N
        N    = xy.shape[1]
        xy_b = xy.copy()
        z_h  = np.ones([2, N])
        z_h[0] = z_h[0] * 0 #z_offset
        xyz_h_b = np.concatenate( [xy_b, z_h], axis=0 )         # xyz wrt b
        xyz_c = np.matmul( cHb[:3,...], xyz_h_b ).squeeze()     # xyz wrt c
        #assert( not( xyz_c[0] == 0 and xyz_c[0] == 0 and xyz_c[0] == 0) )
        uv_gt = rm.pinhole_projection(K, xyz_c)

        # We assume distortion free cam for now...
        ax.plot(
            uv_pred[0,:] / u_scale,  # u
            uv_pred[1,:] / v_scale,  # v
            color=waypts_colors[0],
            marker='D',
            markersize=8.0,
            #lw = 2.5
        )

        ax.plot(
            uv_gt[0,:] / u_scale,  # u
            uv_gt[1,:] / v_scale,  # v
            color=waypts_colors[1],
            marker='s',
            markerfacecolor='none',
            markersize=8.0,
            #lw = 2.5
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
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))

def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    dataset_name: str,
    list_trajs: list,       # [pred_waypoint,   gt_waypoint]
    list_points: list,      # [start_pos,       goal_pos]
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [GREEN, RED],
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
    assert (
        dataset_name in data_config
    ), f"Dataset {dataset_name} not found in data/data_config.yaml"

    ax.imshow(img)
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

def plot_trajs_and_points_on_depth(
    ax: plt.Axes,
    depth: np.ndarray,
    dataset_name: str,
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
    assert (
        dataset_name in data_config
    ), f"Dataset {dataset_name} not found in data/data_config.yaml"

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


def plot_trajs_and_points(
        ax: plt.Axes,
        list_trajs: list,
        list_points: list,
        traj_colors: list = [CYAN, MAGENTA],
        point_colors: list = [RED, GREEN],
        traj_labels: Optional[list] = ["prediction", "ground truth"],
        point_labels: Optional[list] = ["robot", "goal"],
        traj_alphas: Optional[list] = None,
        point_alphas: Optional[list] = None,
        quiver_freq: int = 1,
        default_coloring: bool = True,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        traj_labels: list of labels for trajectories
        point_labels: list of labels for points
        traj_alphas: list of alphas for trajectories
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    assert (
            len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
            traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")

def plot_cam_heading_and_points(
        ax: plt.Axes,
        list_waypoints: list,       # wpts
        list_points: list,          # start & goal pose
        waypoint_colors: list = [CYAN, MAGENTA],
        point_colors: list = [GREEN, RED],
        waypoint_labels: Optional[list] = ["prediction", "ground truth"],
        point_labels: Optional[list] = ["robot", "goal"],
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
    assert (
            len(list_waypoints) <= len(waypoint_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
            waypoint_labels is None or len(list_waypoints) == len(waypoint_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    #bHc = rm.xyzrpy_to_htm(data_config[dataset_name]["camera_matrics"]["cam_wrt_base"])
    #cHb = np.linalg.inv(bHc)

    waypoint_pred = list_waypoints[0]       # x, y
    waypoint_gt   = list_waypoints[1]       # x, y

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

    if waypoint_labels is None:
        ax.plot(
            -waypoint_pred[:, 1],  # y  (negative y b/c y points to the right (see the base_link conf above)
            waypoint_pred[:, 0],   # x
            color=waypoint_colors[0],
            alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            marker="D",
        )
        ax.plot(
            -waypoint_gt[:, 1],        # y
            waypoint_gt[:, 0],         # x
            color=waypoint_colors[1],   # magenta
            alpha=waypoint_alphas[1] if waypoint_alphas is not None else 1.0,
            marker="s",
            markerfacecolor='none',
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
        ax.plot(
            -waypoint_gt[:, 1],
            waypoint_gt[:, 0],  # traj[:, 1],
            color=waypoint_colors[1],
            label=waypoint_labels[1],   # magenta
            alpha=waypoint_alphas[1] if waypoint_alphas is not None else 1.0,
            marker="s",
            markerfacecolor='none',
        )
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
            bearing_pred    = gen_bearing(waypoint_pred)  # draw heading dir
            ax.quiver(
                -waypoint_pred[:,1],  # y
                waypoint_pred[:,0],  # x
                bearing_pred[:,1],
                bearing_pred[:,0],
                color=waypoint_colors[0] * 0.5,
                scale=1.0,
            )
            bearing_gt      = gen_bearing(waypoint_gt)
            ax.quiver(
                -waypoint_gt[:,1],  # y
                waypoint_gt[:,0],  # x
                bearing_gt[1],
                bearing_gt[0],
                color=waypoint_colors[1] * 0.5,
                scale=1.0,
            )

            goal  = list_points[1]
            bearing_goal      = gen_bearing(goal)
            ax.quiver(
                -goal[1],  # y
                goal[0],  # x
                bearing_goal[1],
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
    if waypoint_labels is not None or point_labels is not None:
        ax.legend(loc='lower right', shadow=True, fontsize='medium')
        #ax.legend(bbox_to_anchor=(0.55, 1.15), loc="lower left", ncol=2) # put upper left corner of the leg box at the coord specified by bbox_to_anchor

    max_x = max(max(abs(goal[0]), max(abs(waypoint_gt[..., 0]))), max(waypoint_pred[..., 0]))
    max_x += 0.2 * max_x
    #max_z = max(max_z, 1)

    # min_z = min( min( abs(goal[1]), abs(  waypoint_gt[1] ) ), waypoint_pred[1] ) - 0.1
    max_y = max(max(abs(goal[1]), max(abs(waypoint_gt[..., 1]))), max(waypoint_pred[..., 1])) + 0.1
    max_y += 0.2 * max_y
    #max_x = max(max_x, 1)

    ax.set_aspect("equal", "box")
    ax.set_xlabel('Y-axis')
    ax.set_ylabel('X-axis')

    plot_xlim = plot_zlim = 0
    if nav_config['normalize'] is True:
        plot_xlim = 1.       # xy swapped
        plot_ylim = 1.5
    else:
        plot_xlim = 3.       # xy swapped for drawing
        plot_ylim = 5. #3.4

    ax.set_xlim([-plot_xlim, plot_xlim])
    ax.set_ylim([-plot_ylim + 0.5, plot_ylim])

    textstr0 = '\n'.join((
        r'goal(xy) = (%.2f, %.2f)'% (goal[0], goal[1]), # x y
        r'curr=(%.2f, %.2f)'% (start[0], start[1]), ) )
    waypts_combined = np.concatenate( (waypoint_pred,waypoint_gt), axis = 1 )
    textstr1 = 'xp, yp, xg, yg\n'
    for idx, row in enumerate(waypts_combined):
        (xp,yp,xg,yg) = row
        tmp = f'%.2f, %.2f, %.2f, %.2f\n'%(xp,yp,xg,yg)
        textstr1 +=tmp

    textstr = '\n'.join( (textstr0, textstr1) )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.0, 1.1, textstr, transform=ax.transAxes, fontsize=14,
             horizontalalignment='left', verticalalignment='bottom', bbox=props)


def plot_heading_and_points(
        ax: plt.Axes,
        list_waypoints: list,
        list_points: list,
        waypoint_colors: list = [CYAN, MAGENTA],
        point_colors: list = [GREEN, RED],
        waypoint_labels: Optional[list] = ["prediction", "ground truth"],
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
            waypoint_pred[0], # y
            waypoint_pred[1], # x
            color=waypoint_colors[0],
            alpha=waypoint_alphas[0] if waypoint_alphas is not None else 1.0,
            marker="o",
        )
        ax.plot(
            waypoint_gt[0],
            waypoint_gt[1],
            color=waypoint_colors[1],
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
                waypoint_pred[0],  # y
                waypoint_pred[1],  # x
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
                goal[1],  # z
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
            start[0], # start cam pos wrt base_link
            start[1],
            color=point_colors[0],
            alpha=point_alphas[0] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0
        )
        ax.plot(
            goal[0], # goal cam pos wrt base_link
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
            marker="o",
            markersize=7.0,
            label=point_labels[0],
        )
        ax.plot(
            goal[0],
            goal[1],
            color=point_colors[1],
            alpha=point_alphas[1] if point_alphas is not None else 1.0,
            marker="o",
            markersize=7.0,
            label=point_labels[1],
        )

    # put the legend below the plot
    if waypoint_labels is not None or point_labels is not None:
        #ax.legend(loc='lower right', shadow=True, fontsize='medium')
        ax.legend(bbox_to_anchor=(1.44, -0.10), loc="lower right", ncol=2)
    max_x = max(max(abs(goal[0]), max(abs(waypoint_gt[...,0]))), max(waypoint_pred[...,0]))
    max_x += 0.2 * max_x
    max_x = max(max_x, 1)

    # min_z = min( min( abs(goal[1]), abs(  waypoint_gt[1] ) ), waypoint_pred[1] ) - 0.1
    max_y = max(max(abs(goal[1]), max(abs(waypoint_gt[...,1]))), max(waypoint_pred[...,1])) + 0.1
    max_y += 0.2 * max_y
    max_y = max(max_y, 1)

    ax.set_aspect("equal", "box")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim([-max_x, max_x])
    ax.set_ylim([-max_y, max_y])

    textstr = '\n'.join((
        r'goal=(%.2f, %.2f)'% (goal[0],goal[1]),
        r'label=(%.2f, %.2f)'% (waypoint_gt[0],  waypoint_gt[1]),
        r'pred=(%.2f, %.2f)' % (waypoint_pred[0],waypoint_pred[1])))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])

def gen_bearing(
    waypoint: np.ndarray,
    mag=0.1,
) -> np.ndarray:
    """Generate bearings from a point, (x, y, qs, qz)."""
    bearing = []
#    print(waypoints.shape)  (2,)
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
        raise NotImplementedError
        #v = mag * angle_to_unit_vector(waypoint[2])
    # bearing.append(v)
    # bearing = np.array(bearing)
    return v #bearing

def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    print(waypoints)
    print(waypoints.shape)
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels


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
