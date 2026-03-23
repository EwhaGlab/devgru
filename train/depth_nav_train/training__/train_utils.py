import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

#from depth_nav_train.visualizing.action_utils import visualize_action_and_dist_pred, plot_trajs_and_points #, visualize_traj_pred

from os.path import dirname, abspath
BASE_DIR = os.path.join( dirname(dirname(dirname(dirname(os.path.abspath(__file__))))))

import sys
sys.path.append(BASE_DIR)
from utils.visualizing.action_utils import visualize_action_and_dist_pred, plot_trajs_and_points #, visualize_traj_pred
from utils.visualizing.distance_utils import visualize_dist_pred
from utils.visualizing.visualize_utils import to_numpy, from_numpy
from depth_nav_train.training.logger import Logger
from depth_nav_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from depth_nav_train.data.data_utils import _denormalize_pose

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import rigid_motion as rm
from scipy.io import savemat
import pickle

# LOAD DATA CONFIG
data_config_path = BASE_DIR + "/train/depth_nav_train/data/data_config.yaml"
config_path = BASE_DIR + "/config/depth_nav.yaml"

with open(data_config_path, "r") as f:
    data_config = yaml.safe_load(f)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    #dataset_name = list( config['datasets'].keys() )[0]

# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def _quat_dist( q1: torch.Tensor,
                q2: torch.Tensor):
    # q1 = [B, 4]
    assert( len(q1.shape) == len(q2.shape) == 2 )
    (B, d) = q1.shape
    # q1_ = q1.clone()
    # mdict = {"q1": q1_.detach().cpu().numpy()}
    # savemat("/home/hankm/matlab_ws/DepthNav/q1.mat", mdict)
    assert(q1.clone().norm(dim=1).sum().int() == B), f" sum of q1 norm is { q1.clone().norm(dim=1).sum() }"
    assert(q2.clone().norm(dim=1).sum().int() == B), f" sum of q2 norm is { q2.clone().norm(dim=1).sum() }"

    q2_conj = -q2.clone()
    q2_conj[:, 0] = q2[:, 0].clone()
    d1 = torch.linalg.vector_norm(q1 - q2, dim=1)
    d2 = torch.linalg.vector_norm(q1 - q2_conj, dim=1)

    return torch.min(d1, d2)

# Train utils for ViNT and GNM
def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
  #  dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None, # 1 if the pred action (wpt) is bounded btwn 0 ~ 10
):
    """
    Compute losses for distance and action prediction.
    """

    #dist_label_fake = torch.zeros(dist_label.shape)
    #dist_pred_fake = torch.zeros(dist_pred.shape)
    #dist_loss = F.mse_loss(dist_pred_fake.squeeze(-1), dist_label_fake.float())

    #dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())  # [B] sized vector: distance to goal
    #print("dist_label: ", dist_label, "shape: ", dist_label.shape)

    def action_reduce(unreduced_loss: torch.Tensor):
        # takes [B, X, Y], then repeats mean() of the last dim until the shape becomes [B]
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"  # action_label.shape
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none") )

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[..., :2], action_label[..., :2], dim=-1
    ))

    # print("action_waypts_cos_sim: ", F.cosine_similarity(action_pred[:, :, :2], action_label[:, :, :2], dim=-1), "shape: ", F.cosine_similarity(action_pred[:, :, :2], action_label[:, :, :2], dim=-1).shape )
    # print("action_waypts_cos_sim (action reduced): ", action_waypts_cos_similairity.shape )

    # multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
    #     torch.flatten(action_pred[..., :2], start_dim=1),
    #     torch.flatten(action_label[..., :2], start_dim=1),
    #     dim=-1,
    # ))

    # print("multi_action_waypts_cos_sim : ", F.cosine_similarity(torch.flatten(action_pred[:, :, :2], start_dim=1), torch.flatten(action_label[:, :, :2], start_dim=1), dim=-1) )
    # print("multi_action_waypts_cos_sim shape: ", F.cosine_similarity(torch.flatten(action_pred[:, :, :2], start_dim=1), torch.flatten(action_label[:, :, :2], start_dim=1), dim=-1).shape )
    # print("multi_action_waypts_cos_sim (action reduced) shape: ", multi_action_waypts_cos_sim.shape )

    results = {
        #"dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        #"multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }
    device = action_pred.device
    #TODO: Find a better way to compute angle diff btwn two quaternions
    if learn_angle:
        nb = action_pred.shape[0]
        qw_pred = action_pred[..., 2].clone()
        qz_pred = action_pred[..., 3].clone()
        quat_pred = torch.zeros([nb, 4]).to(device)
        quat_pred[..., 0] = qw_pred
        quat_pred[..., 3] = qz_pred

        qw_label = action_label[..., 2].clone()
        qz_label = action_label[..., 3].clone()
        quat_label = torch.zeros([nb, 4]).to(device)
        quat_label[..., 0] = qw_label
        quat_label[..., 3] = qz_label

        q_dist = _quat_dist(quat_pred, quat_label)
        action_orient_quat_similarity = (q_dist * action_mask).mean() / (action_mask.mean() + 1e-2)
        results["action_orient_quat_sim"] = action_orient_quat_similarity

    #total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    total_loss = action_loss
    results["total_loss"] = total_loss
    #exit()
    return results

def _log_data(
    data_info,
    i,
    epoch,
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
    #ts_dist_pred,
    #ts_dist_label,
    ts_goal_pos,
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
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    # print("obs depth shape: ", obs_depths.shape)
    # save data

    # For the visualization, we change the shape of obs_images [B,C,H,W] --> [B,H,W,C]
    tp_data_info  = data_info
    np_obs_images = to_numpy(ts_obs_images_vz).transpose(0, 2, 3, 1) # enforce [B, H, W, C]
    np_obs_depths = to_numpy(ts_obs_depths_vz)
    np_goal_image = to_numpy(ts_goal_image_vz).transpose(0, 2, 3, 1) # enforce [B, H, W, C]
    np_goal_depth = to_numpy(ts_goal_depth_vz)
 #   np_dist_pred  = to_numpy(ts_dist_pred)
 #   np_dist_label = to_numpy(ts_dist_label)
    np_goal_pos   = to_numpy(ts_goal_pos)
    np_action_pred= to_numpy(ts_action_pred)
    np_action_label=to_numpy(ts_action_label)

    dataset_name = []
    if "isaac" in data_info[0]:
        dataset_name = 'isaac_sim'
    else:
        print("unknown dataset %s name "%(dataset_name))
        raise NotImplementedError

    if config['normalize'] == True:     # denormalize pose to display them
        max_frame_dist = config['distance']['max_frame_dist']
        np_action_pred = _denormalize_pose( np_action_pred, max_frame_dist=max_frame_dist, eta=1.0, dataset_name = dataset_name)
        np_action_label = _denormalize_pose( np_action_label, max_frame_dist=max_frame_dist, eta=1.0, dataset_name = dataset_name)
        np_goal_pos = _denormalize_pose( np_goal_pos, max_frame_dist=max_frame_dist, eta=1.0, dataset_name = dataset_name)

    # if save_data_path is not None:
    #     mdict = {'data_info': data_info, 'obs_images': np_obs_images, 'obs_depths': np_obs_depths, 'goal_images': np_goal_image, 'goal_depth': np_goal_depth,
    #              'dist_pred': np_dist_pred, 'dist_label': np_dist_label, 'goal_pos': np_goal_pos,
    #              'action_pred': np_action_pred, 'action_label': np_action_label}
    #     log_mat_file = f'%s/data%03d.mat'%(save_data_path, epoch)
    #     log_pkl_file = f'%s/data%03d.pkl'%(save_data_path, epoch)
    #     savemat(log_mat_file, mdict)
    #     with open(log_pkl_file, 'wb') as fp:
    #         pickle.dump(mdict, fp)

    # TODO: Need to Fix the logging/ debugging functions to below
    if image_log_freq != 0 and i % image_log_freq == 0:
        # visualize_dist_pred(
        #     np_obs_images,
        #     np_obs_depths,   # 0~1 normalized
        #     np_goal_image,
        #     np_goal_depth,
        #     np_dist_pred,
        #     np_dist_label,
        #     mode,
        #     project_folder,
        #     epoch = epoch,
        #     num_images_preds = num_images_log,
        #     use_wandb=use_wandb,
        # )

        #visualize_traj_pred(
        #visualize_action_pred(
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
         #   np_dist_pred,
         #   np_dist_label,
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )

def train(
    goal_type: str,
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = False,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model.train()   # this line sets the model to train mode
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    # multi_action_waypts_cos_sim_logger = Logger(
    #     "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    # )
    action_orient_quat_sim_logger = Logger(
        "action_orient_quat_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        #"multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "action_orient_quat_sim": action_orient_quat_sim_logger,
        "total_loss": total_loss_logger,
    }

    # if learn_angle:
    #     action_orien_cos_sim_logger = Logger(
    #         "action_orien_cos_sim", "train", window_size=print_log_freq
    #     )
    #     multi_action_orien_cos_sim_logger = Logger(
    #         "multi_action_orien_cos_sim", "train", window_size=print_log_freq
    #     )
    #     loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
    #     loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):        # actual training loop
        (
            ts_obs_images,     # obs_image,
            ts_obs_depths,
            ts_goal_image,     # goal_image,
            ts_goal_depth,
            ts_action_label,   # wrt curr cam
            ts_dist_label,
            ts_goal_pos,       # goal camera pose wrt curr cam
            ts_dataset_index,
            ts_action_mask,
            str_data_info,
        ) = data
# obs_image
        # TODO: check if dim=0 is correct for torch.split() and torch.cat() below
        assert(ts_obs_images.max() <= 1)
        # obs_images.shape  is [B, Contxt*C, H, W],  ex) [256, 18, 64, 85]
        tuple_ts_obs_images = torch.split(ts_obs_images, 3, dim=1)  # tuple  of  obs image imgs (B, CxC, H, W)
        ts_curr_obs_image_vz = TF.resize(tuple_ts_obs_images[-1], VISUALIZATION_IMAGE_SIZE) # current obs img (160, 120)
        ls_ts_obs_images = [transform(ts_obs_image).to(device) for ts_obs_image in tuple_ts_obs_images]
        ts_obs_images = torch.cat(ls_ts_obs_images, dim=1)

# obs_depth
        assert(ts_obs_depths.max() <= 1)
        # obs_depths.shape is [B, Contxt, H, W],   ex) [256, 6, 64, 85]
        tuple_ts_obs_depths = torch.split(ts_obs_depths, 1, dim=1)
        ts_curr_obs_depth_vz = TF.resize(tuple_ts_obs_depths[-1], VISUALIZATION_IMAGE_SIZE) # the last element of the tuple
        ls_ts_obs_depths = torch.cat(tuple_ts_obs_depths, dim=1)
        ts_obs_depths = ls_ts_obs_depths.to(device)

# goal_image
        assert(ts_goal_image.max() <= 1)
        ts_goal_image_vz = TF.resize(ts_goal_image, VISUALIZATION_IMAGE_SIZE)
        ts_goal_image = transform(ts_goal_image).to(device) # transform() does img normalization (data preprocessing)

# goal_depth
        assert(ts_goal_depth.max() <= 1)
        ts_goal_depth_vz = TF.resize(ts_goal_depth, VISUALIZATION_IMAGE_SIZE)
        ts_goal_depth = ts_goal_depth.to(device)

        #######################################################################################################
        # model step FF
        if goal_type == "rgb":
            model_outputs = model(ts_obs_images, ts_goal_image)
        elif goal_type == "depth":
            model_outputs = model(ts_obs_depths, ts_goal_depth)
        else:
            print("goal type %s is not supported" % goal_type)
            raise NotImplementedError

        ts_dist_label = ts_dist_label.to(device)          # geometric distance to goal
        ts_action_label = ts_action_label.to(device)      # [5, 2] (x,y) 5 future steps
        ts_action_mask = ts_action_mask.to(device)

        optimizer.zero_grad()               # re-init the gradient buffers b/c we don't want any grad from previous epoch
        ts_action_pred = model_outputs      # [B, len_traj_pred, num_params]

        losses = _compute_losses(
            dist_label=     ts_dist_label,
            action_label=   ts_action_label,
        #    dist_pred=      ts_dist_pred,
            action_pred=    ts_action_pred,
            alpha=alpha,
            learn_angle=    learn_angle,
            action_mask=    ts_action_mask,
        )

        losses["total_loss"].backward()     # back-propagation
        optimizer.step()                    # update model parameter (weights)  ==> model learns

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            data_info=str_data_info,
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            ts_obs_images_vz=ts_curr_obs_image_vz,     # (B, 3, H, W)
            ts_obs_depths_vz= ts_curr_obs_depth_vz,     # (B, 1, H, W)
            ts_goal_image_vz= ts_goal_image_vz,
            ts_goal_depth_vz= ts_goal_depth_vz,
            ts_action_pred  = ts_action_pred,
            ts_action_label = ts_action_label,
           # ts_dist_pred = ts_dist_pred,
          #  ts_dist_label= ts_dist_label,
            ts_goal_pos= ts_goal_pos,
            dataset_index = ts_dataset_index,
            wandb_log_freq= wandb_log_freq,
            print_log_freq= print_log_freq,
            image_log_freq= image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
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
    epoch: int = 0,
    alpha: float = 0.5,
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
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    action_orient_quat_sim_logger = Logger("action_orient_quat_sim", eval_type)
    #multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
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
        # action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        # multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        # loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        # loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    #viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                ts_obs_images,
                ts_obs_depths,
                ts_goal_image,
                ts_goal_depth,
                ts_action_label,
                ts_dist_label,
                ts_goal_pos,
                ts_dataset_index,
                ts_action_mask,
                str_data_info,
            ) = data
            # obs_image
        # TODO: check if dim=0 is correct for torch.split() and torch.cat() below
            assert (ts_obs_images.max() <= 1)
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

            if goal_type == "rgb":
                model_outputs = model(ts_obs_images, ts_goal_image)
            elif goal_type == "depth":
                model_outputs = model(ts_obs_depths, ts_goal_depth)
            else:
                print("goal type %s is not supported" % goal_type)
                raise NotImplementedError

            ts_dist_label = ts_dist_label.to(device)
            ts_action_label = ts_action_label.to(device)
            ts_action_mask = ts_action_mask.to(device)
          #  ts_dist_pred, ts_action_pred = model_outputs
            ts_action_pred = model_outputs

            losses = _compute_losses(
                dist_label=ts_dist_label,
                action_label=ts_action_label,
           #     dist_pred=ts_dist_pred,
                action_pred=ts_action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=ts_action_mask,
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
        epoch=epoch,
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
        ts_goal_pos=ts_goal_pos,
    #    ts_dist_pred=ts_dist_pred,
   #     ts_dist_label=ts_dist_label,
        dataset_index=ts_dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
        save_data_path=None, #"/home/hankm/matlab_ws/DepthNav/log"
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }


def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)


