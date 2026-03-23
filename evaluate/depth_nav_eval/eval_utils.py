import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.append(BASE_DIR)

from utils.visualizing.action_utils import visualize_action_and_dist_pred, plot_trajs_and_points #, visualize_traj_pred
#from utils.visualizing.distance_utils import visualize_dist_pred
from utils.visualizing.visualize_utils import to_numpy, from_numpy
from depth_nav_eval.logger import Logger
from data.data_utils import VISUALIZATION_IMAGE_SIZE

#from data.data_utils import _denormalize_pose
from data.data_utils import _denormalize_subgoal, _denormalize_waypoints

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import scipy

# LOAD DATA CONFIG
data_config_path = BASE_DIR + "/config/data_config.yaml"
config_path = BASE_DIR + "/config/depth_nav.yaml"

with open(data_config_path, "r") as f:
    data_config = yaml.safe_load(f)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def _quat_dist( q1: torch.Tensor,
                q2: torch.Tensor):

    assert( len(q1.shape) == len(q2.shape) == 2 ) # q1.shape = q2 = [B, 4]
    (B, d) = q1.shape

    assert(q1.clone().norm(dim=1).sum().int() == B), f" sum of q1 norm is { q1.clone().norm(dim=1).sum() }"
    assert(q2.clone().norm(dim=1).sum().int() == B), f" sum of q2 norm is { q2.clone().norm(dim=1).sum() }"

    q2_conj = -q2.clone()
    q2_conj[:, 0] = q2[:, 0].clone()
    d1 = torch.linalg.vector_norm(q1 - q2, dim=1)
    d2 = torch.linalg.vector_norm(q1 - q2_conj, dim=1)

    return torch.min(d1, d2)

def _quat_ang_dist(q1: torch.Tensor, q2: torch.Tensor):
    """
    Unsigned angular distance between quaternions (radians).
    Inputs: (...,4) = [qw,qx,qy,qz] or (...,2) = [qw,qz]
    Returns:
      sin2_half: sin^2(theta/2) in [0,1], shape q1.shape[:-1]
      angle    : theta in [0, pi],        shape q1.shape[:-1]
    """
    assert q1.shape == q2.shape and q1.shape[-1] in (2, 4), f"{q1.shape} vs {q2.shape}"

    # Normalize along last dim (robust even if inputs are near-unit)
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-12)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-12)

    # Dot product along last dim; use |dot| so q ≡ -q
    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)  # |cos(theta/2)|

    sin2_half = 1.0 - dot * dot                          # in [0,1]
    angle = 2.0 * torch.acos(dot)                        # in [0, pi]

    return sin2_half, angle

# Train utils for ViNT and GNM
def _compute_losses(
    pose_diff_label: torch.Tensor,  # (B, 4)
    action_label: torch.Tensor,
    pose_diff_pred: torch.Tensor,   # (B, 4)
    action_pred: torch.Tensor,
    alpha: float,       #   weight between action and dist loss
    beta: float,        #   weight between orient loss and waypt loss
    learn_angle: bool,
    action_mask: torch.Tensor = None, # 1 if the pred action (wpt) is bounded btwn 0 ~ 10
    use_time_weight: bool = True,
    coll_logit: torch.Tensor = None, # (B,)
    coll_label: torch.Tensor = None,
    lambda_cls: float = 0.5
):
    """
    Compute losses for distance and action prediction.
    """

    #pose_diff_loss = F.mse_loss(pose_diff_pred.squeeze(-1), pose_diff_label.float())  # [B] sized vector: distance to goal

    # position loss
    #pos_loss = F.smooth_l1_loss(pose_diff_pred[:, :2], pose_diff_label[:, :2], reduction='none').sum(dim=-1)  # (B,)
    dx_l = F.smooth_l1_loss(pose_diff_pred[:, 0], pose_diff_label[:, 0], reduction='mean')
    dy_l = F.smooth_l1_loss(pose_diff_pred[:, 1], pose_diff_label[:, 1], reduction='mean')
    pos_loss = dx_l + dy_l
    # ang loss
    p_q = F.normalize(pose_diff_pred[:, 2:], dim=-1, eps=1e-8)  # (B,2)
    g_q = F.normalize(pose_diff_label[:, 2:], dim=-1, eps=1e-8)
    dot = (p_q * g_q).sum(dim=-1).abs().clamp(0, 1)  # |cos(dθ/2)|
    yaw_loss = 1.0 - dot ** 2  # smooth ≈ sin^2(dθ/2)

    if learn_angle:
        pose_diff_loss = pos_loss.mean() + yaw_loss.mean()
    else:
        pose_diff_loss = pos_loss.mean()

    len_traj_pred = config['len_traj_pred']
    use_time_weight = config['use_time_weight']

    if use_time_weight == True:
        pred_weights = torch.linspace(1.0, len_traj_pred, len_traj_pred, device=action_pred.device)
        pred_weights = (pred_weights / pred_weights.mean()).view(1, len_traj_pred, 1)
    else:
        pred_weights = torch.ones([1, len_traj_pred], device=action_pred.device)
        pred_weights = pred_weights.view(1, len_traj_pred, 1)

    def action_reduce(unreduced_loss: torch.Tensor):
        # takes [B, X, Y, theta], then repeats mean() of the last dim until the shape becomes [B]
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"  # action_label.shape
    action_mse = F.mse_loss(action_pred[..., :2], action_label[..., :2], reduction="none")  # [B, len_traj_pred, 2], 2 b/c qz and qw only

    weighted_action_loss = action_mse * pred_weights
    action_loss = action_reduce( weighted_action_loss ) # action waypt loss

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[..., :2], action_label[..., :2], dim=-1
    ))

    collision_loss = torch.tensor(0.0, device=action_pred.device)
    collision_acc = torch.tensor(0.0, device=action_pred.device)
    if (coll_logit is not None) and (coll_label is not None):
        z = coll_logit.view(-1)  # logits
        y = coll_label.to(z.dtype).view(-1)  # labels in {0,1}

        # handle class imbalance a bit (1 line). Safe if no positives in batch.
        pos = y.sum()
        neg = (1.0 - y).sum()
        pos_weight = (neg / pos.clamp_min(1.0))
        pos_weight = pos_weight.to(device=z.device, dtype=z.dtype)

        collision_loss = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)
        # with torch.no_grad():
        #     pred = (z > 0).to(y.dtype)
        #     collision_acc = (pred == y).float().mean()

    results = {
        "pose_diff_loss": pose_diff_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        'dx pred error:': dx_l,
        'dy_pred error:': dy_l,
        'collision_loss': collision_loss,
        #"multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }
    device = action_pred.device
    #TODO: Find a better way to compute angle diff btwn two quaternions
    if learn_angle:
        nb = action_pred.shape[0]
        qw_pred = action_pred[..., 2].clone()
        qz_pred = action_pred[..., 3].clone()
        quat_pred = torch.zeros([nb, len_traj_pred, 4]).to(device)

        quat_pred[..., 0] = qw_pred
        quat_pred[..., 3] = qz_pred
        quat_pred = torch.nn.functional.normalize(quat_pred, dim=2)

        qw_label = action_label[..., 2].clone()  # [x y qw qz]
        qz_label = action_label[..., 3].clone()
        quat_label = torch.zeros([nb, len_traj_pred, 4]).to(device)
        quat_label[..., 0] = qw_label
        quat_label[..., 3] = qz_label

        q_dist, q_ang = _quat_ang_dist(quat_pred, quat_label)  # q_dist (B, len_pred),
        #orient_mse = F.mse_loss(quat_label, quat_pred, reduction="none")
        p = F.normalize(action_pred[..., 2:4], dim=-1, eps=1e-8)
        g = F.normalize(action_label[..., 2:4], dim=-1, eps=1e-8)
        dot = (p * g).sum(dim=-1).abs().clamp(0, 1)  # |cos(dθ/2)|
        wp_yaw_loss_per = 1.0 - dot ** 2
        weighted_orient_loss = wp_yaw_loss_per * pred_weights.squeeze(-1)
        orient_loss = action_reduce( weighted_orient_loss )

        action_orient_quat_similarity = ( (1-q_dist) * action_mask[...,None].repeat(1, len_traj_pred)).mean() / (action_mask.mean() + 1e-2)
        results["action_orient_quat_sim"] = action_orient_quat_similarity

        wp, zp = p_q.unbind(dim=-1) # each (B,)
        wg, zg = g_q.unbind(dim=-1) # each (B,)
        w_rel = wp * wg + zp * zg
        z_rel = zp * wg - wp * zg
        theta_err = 2.0 * torch.atan2(z_rel, w_rel)
        # wrap to (-pi, pi]
        theta_err = (theta_err + torch.pi) % (2 * torch.pi) - torch.pi

        results['angle_pred_error'] = theta_err

        tot_action_loss = (beta * orient_loss + (1 - beta) * action_loss)
        total_loss = alpha * tot_action_loss + (1.0 - alpha) * pose_diff_loss + lambda_cls * collision_loss #dist_loss
    else:
        total_loss = alpha * action_loss + (1.0 - alpha) * pose_diff_loss + lambda_cls * collision_loss #dist_loss
    results["total_loss"] = total_loss
    return results

def _log_data(
    data_info,
    i,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    ts_obs_images_vz,     # NOT (85, 64) !!! the img is rescaled to VISUALIZATION_IMAGE_SIZE = (160, 120)
    ts_obs_depths_vz,     # the img is rescaled to VISUALIZATION_IMAGE_SIZE = (160, 120)
    ts_goal_image_vz,     # the img is rescaled to VISUALIZATION_IMAGE_SIZE = (160, 120)
    ts_goal_depth_vz,     # the img is rescaled to VISUALIZATION_IMAGE_SIZE = (160, 120)
    ts_action_pred,
    ts_action_label,
    ts_pose_diff_pred,      # [dy, dy, qw, qz]
    ts_pose_diff_label,     # [dy, dy, qw, qz]
    ts_goal_pos,
    ts_collision_pred,
    ts_collision_label,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
    save_data_path = None,
):
    """
    Log data to wandb and print to console.
    """
    # data_log = {}
    # for key, logger in loggers.items():
    #     if use_latest:
    #         data_log[logger.full_name()] = logger.latest()
    #         if i % print_log_freq == 0 and print_log_freq != 0:
    #             print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
    #     else:
    #         data_log[logger.full_name()] = logger.average()
    #         if i % print_log_freq == 0 and print_log_freq != 0:
    #             print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    # if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
    #     wandb.log(data_log, commit=wandb_increment_step)

    # print("obs depth shape: ", obs_depths.shape)
    # save data

    # For the visualization, we change the shape of obs_images [B,C,H,W] --> [B,H,W,C]
    tp_data_info  = data_info
    np_obs_images = to_numpy(ts_obs_images_vz).transpose(0, 2, 3, 1) # enforce [B, H, W, C]
    np_obs_depths = to_numpy(ts_obs_depths_vz)
    np_goal_image = to_numpy(ts_goal_image_vz).transpose(0, 2, 3, 1) # enforce [B, H, W, C]
    np_goal_depth = to_numpy(ts_goal_depth_vz)
    np_pose_diff_pred  = to_numpy(ts_pose_diff_pred)
    np_pose_diff_label = to_numpy(ts_pose_diff_label)
    np_goal_pos   = to_numpy(ts_goal_pos)
    np_action_pred= to_numpy(ts_action_pred)
    np_action_label=to_numpy(ts_action_label)
    np_collision_pred=to_numpy(ts_collision_pred)
    np_collision_label=to_numpy(ts_collision_label)

    dataset_name = []
    if "isaac" in data_info[0]:
        dataset_name = 'isaac_sim'
    elif "former" in data_info[0]:
        dataset_name = 'former'
    else:
        print("unknown dataset %s name "%(dataset_name))
        raise NotImplementedError

    if config['normalize'] == True:     # denormalize pose to display them
        np_action_pred = _denormalize_waypoints( np_action_pred, waypoint_spacing=config['datasets'][dataset_name]['waypoint_spacing'], dataset_name = dataset_name)
        np_action_label = _denormalize_waypoints( np_action_label, waypoint_spacing=config['datasets'][dataset_name]['waypoint_spacing'], dataset_name = dataset_name)
        np_goal_pos = _denormalize_subgoal( np_goal_pos, max_frame_dist=config['distance']['max_frame_dist'], dataset_name = dataset_name)

    # TODO: Need to Fix the logging/ debugging functions to below
    if image_log_freq != 0 and i % image_log_freq == 0:

        print(project_folder)
        visualize_action_and_dist_pred(
            tp_data_info,
            to_numpy(ts_obs_images_vz),
            to_numpy(ts_goal_image_vz),
            to_numpy(ts_obs_depths_vz),
            to_numpy(ts_goal_depth_vz),
            to_numpy(dataset_index),
            np_goal_pos,
            np_action_pred,
            np_action_label,
            np_pose_diff_pred,
            np_pose_diff_label,
            np_collision_pred,
            np_collision_label,
            mode,
            normalized,
            save_folder=project_folder,
            epoch=-1,
            num_images_preds=32,
            use_wandb=False,
        )

def evaluate(
    eval_type: str,
    goal_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    alpha: float = 0.9,
    beta:  float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("pose_diff_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    action_orient_quat_sim_logger = Logger("action_orient_quat_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "action_orient_quat_sim": action_orient_quat_sim_logger,
        #"multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orient_quat_sim_logger = Logger("action_orien_quat_sim", eval_type)
        loggers["action_orient_quat_sim"] = action_orient_quat_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    #viz_obs_image = None

    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} ",
        )
        for i, data in enumerate(tqdm_iter):
            (
                ts_obs_images,
                ts_obs_depths,
                ts_goal_image,
                ts_goal_depth,
                ts_action_label,
                ts_context_action,
                ts_goal_pose,
                ts_pose_diff_label,
                ts_dataset_index,
                ts_action_mask,
                ts_collision_label,  # true (bool) if collision
                str_data_info,
            ) = data
            # for str in str_data_info:
            #     print('%s' % str)
            # print("\n")

            # obs_image
        # TODO: check if dim=0 is correct for torch.split() and torch.cat() below
            assert (ts_obs_depths.max() <= 1)
            assert (ts_goal_image.shape[1] == 3), f" rgb img shape is {ts_goal_image.shape}"
        # ts_obs_images.shape  is [B, Contxt*C, H, W],  ex) [256, 18, 64, 85]
            tuple_ts_obs_images = torch.split(ts_obs_images, 3, dim=1)  # tuple  of  obs image imgs (B, CxC, H, W)
            ts_curr_obs_image_vz   = TF.resize(tuple_ts_obs_images[-1], VISUALIZATION_IMAGE_SIZE)  # current obs img (160, 120)
            ls_ts_obs_images = [transform(ts_obs_image).to(device) for ts_obs_image in tuple_ts_obs_images]
            ts_obs_images = torch.cat(ls_ts_obs_images, dim=1)

        # obs_depth
            tuple_obs_depths = torch.split(ts_obs_depths, 1, dim=1)  # convert tensor to tuple
            ts_curr_obs_depth_vz = TF.resize(tuple_obs_depths[-1], VISUALIZATION_IMAGE_SIZE) # current obs
            ts_obs_depths = ts_obs_depths.to(device)
        # goal_image
            assert (ts_goal_image.max() <= 1)
            ts_goal_image_vz = TF.resize(ts_goal_image, VISUALIZATION_IMAGE_SIZE)
            ts_goal_image = transform(ts_goal_image).to(device)  # transform() does img normalization (data preprocessing)

        # goal_depth
            ts_goal_depth_vz = TF.resize(ts_goal_depth, VISUALIZATION_IMAGE_SIZE)
            ts_goal_depth = ts_goal_depth.to(device)
        # model step
            ts_context_action = ts_context_action.to(device)
            ts_goal_pose = ts_goal_pose.to(device)

            #######################################################################################################
            # model step FF
            if config['model_type'] == 'image-rnn':
                if goal_type == "rgb":
                    model_outputs = model(ts_obs_images, ts_goal_image)
                elif goal_type == "depth":
                    model_outputs = model(ts_obs_depths, ts_goal_depth)
                else:
                    print("goal type %s is not supported" % goal_type)
                    raise NotImplementedError
            elif config['model_type'] == 'image_pose_rnn':
                if goal_type == "rgb":
                    model_outputs = model(ts_obs_images, ts_goal_image, ts_context_action, ts_goal_pose)
                elif goal_type == "depth":
                    model_outputs = model(ts_obs_depths, ts_goal_depth, ts_context_action, ts_goal_pose)
                else:
                    print("goal type %s is not supported" % goal_type)
                    raise NotImplementedError
            elif config['model_type'] == 'image_pg_rnn':
                if goal_type == "rgb":
                    model_outputs = model(ts_obs_images, ts_goal_image, ts_goal_pose)
                elif goal_type == "depth":
                    model_outputs = model(ts_obs_depths, ts_goal_depth, ts_goal_pose)
                else:
                    print("goal type %s is not supported" % goal_type)
                    raise NotImplementedError
            else:
                print("unknown NN model")
                raise NotImplementedError

            ts_pose_diff_label   = ts_pose_diff_label.to(device)
            ts_action_label = ts_action_label.to(device)
            ts_action_mask  = ts_action_mask.to(device)
            ts_collision_label = ts_collision_label.to(device)

            ts_action_pred, ts_pose_diff_pred, ts_collision_pred = model_outputs
            #ts_action_pred = model_outputs

            losses = _compute_losses(
                pose_diff_label=ts_pose_diff_label,
                action_label=ts_action_label,
                pose_diff_pred=ts_pose_diff_pred,
                action_pred=ts_action_pred,
                alpha=alpha,
                beta= beta,
                learn_angle=learn_angle,
                action_mask=ts_action_mask,
                coll_logit=ts_collision_pred,
                coll_label=ts_collision_label,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    print("\n %s \n" % str_data_info[0])
    print("logging started in eval \n")
    # Log data to wandb/console, with visualizations selected from the last batch
    _log_data(
        data_info=str_data_info,
        i=i,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        ts_obs_images_vz=ts_curr_obs_image_vz,
        ts_obs_depths_vz=ts_curr_obs_depth_vz,
        ts_goal_image_vz=ts_goal_image_vz,
        ts_goal_depth_vz=ts_goal_depth_vz,
        ts_action_pred=ts_action_pred,
        ts_action_label=ts_action_label,
        ts_goal_pos=ts_goal_pose,
        ts_pose_diff_pred=ts_pose_diff_pred,
        ts_pose_diff_label=ts_pose_diff_label,
        ts_collision_pred=ts_collision_pred,
        ts_collision_label=ts_collision_label,
        dataset_index=ts_dataset_index,
        use_wandb=False, # use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
        save_data_path=None, #"/home/hankm/matlab_ws/DepthNav/log"
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()
