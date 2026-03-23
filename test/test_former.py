import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF

"""
IMPORT YOUR MODEL HERE
"""
import sys
sys.path.append('../train')
sys.path.append('../utils')
sys.path.append('../deployment')
from depth_nav_train.models.depth_rnn.depth_rnn import DepthRNN
from visualizing.action_utils import compare_pred_to_label
from visualizing.visualize_utils import to_numpy, from_numpy
from data.data_utils import _denormalize_pose


from visualizing.visualize_utils import (
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
from depth_nav_train.data.depth_nav_dataset import DepthNav_Dataset
from depth_nav_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from utils import load_model

import tqdm
import itertools

MODEL_CONFIG_PATH = "../config/models.yaml"

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def main(config):

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#    transform = ([transforms.Normalize(mean=[1, 1, 1], std=[0, 0, 0]),])
    transform = transforms.Compose(transform) # data preprocessing ( e.g. data augmentation )

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False



    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        data_split_type = "test"
        dataset = DepthNav_Dataset(
            data_folder=data_config["data_folder"],
            data_split_folder=data_config[data_split_type],
            dataset_name=dataset_name,
            image_size=config["image_size"],
            waypoint_spacing=data_config["waypoint_spacing"],   # 1 by default b/c waypoint spacing is not set in the config file
            min_frame_dist=config["distance"]["min_frame_dist"],
            max_frame_dist=config["distance"]["max_frame_dist"],
            negative_mining=data_config["negative_mining"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            context_size=config["context_size"],
            context_type=config["context_type"],
            goal_dist_type=config["goal_dist_type"],
            end_slack=data_config["end_slack"],
            goals_per_obs=data_config["goals_per_obs"],
            normalize=config["normalize"],
            goal_type=config["goal_type"],
        )

        dataset_type = f"{dataset_name}_{data_split_type}"
        if dataset_type not in test_dataloaders:
            test_dataloaders[dataset_type] = {}
        test_dataloaders[dataset_type] = dataset

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    context_size = config["context_size"]
    # load model weights
    ckpth_path = os.path.join(os.path.abspath(os.getcwd())+"/model_weights/latest.pth")

    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

    # load the model
    model = load_model(
        ckpth_path,
        config,
        device,
    )
    model = model.to(device)
    model.eval()

    dataloader = test_dataloaders['former'].dataset
    num_batches = len(dataloader) / 100
    num_batches = max(int(num_batches * 1), 1)

    use_tqdm = True
    eval_type = dataset_type

    visualize_path = os.path.join(os.path.dirname(__file__), "visualize")
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    tidx = 0
    # viz_obs_image = None
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
                ts_dist_label,
                ts_goal_pos,
                ts_dataset_index,
                ts_action_mask,
                str_data_info,
            ) = data
            # obs_image
            # TODO: check if dim=0 is correct for torch.split() and torch.cat() below
            assert (ts_obs_images.max() <= 1)

            ts_obs_images = ts_obs_images[None, ...]
            ts_obs_depths = ts_obs_depths[None, ...]
            ts_goal_image = ts_goal_image[None, ...]
            ts_goal_depth = ts_goal_depth[None, ...]

            # ts_obs_images.shape  is [B, Contxt*C, H, W],  ex) [256, 18, 64, 85]
            tuple_ts_obs_images = torch.split(ts_obs_images, 3, dim=1)  # tuple  of  obs image imgs (B, CxC, H, W)
            ts_curr_obs_image_vz = TF.resize(tuple_ts_obs_images[-1],
                                             VISUALIZATION_IMAGE_SIZE)  # current obs img (160, 120)

            # obs_depth
            tuple_obs_depths = torch.split(ts_obs_depths, 1, dim=1)  # convert tensor to tuple
            ts_curr_obs_depth_vz = TF.resize(tuple_obs_depths[-1], VISUALIZATION_IMAGE_SIZE)  # current obs
            ts_obs_depths = ts_obs_depths.to(device)
            # goal_image

            ts_goal_image_vz = TF.resize(ts_goal_image, VISUALIZATION_IMAGE_SIZE)

            # goal_depth
            ts_goal_depth_vz = TF.resize(ts_goal_depth, VISUALIZATION_IMAGE_SIZE)
            ts_goal_depth = ts_goal_depth.to(device)
            # model step
            model_outputs = model(ts_obs_depths, ts_goal_depth)
            ts_dist_label = ts_dist_label.to(device)
            ts_action_label = ts_action_label.to(device)
            ts_action_mask = ts_action_mask.to(device)
            #  ts_dist_pred, ts_action_pred = model_outputs
            ts_action_pred = model_outputs

            viz_obs_depth = numpy_to_depth(to_numpy(ts_curr_obs_depth_vz).squeeze())
            viz_goal_depth = numpy_to_depth(to_numpy(ts_goal_depth_vz)  .squeeze())
            viz_obs_image = numpy_to_img(to_numpy(ts_curr_obs_image_vz).squeeze())  # image resize back to (640, 480)
            viz_goal_image = numpy_to_img(to_numpy(ts_goal_image_vz).transpose(0, 2, 3, 1).squeeze())
            np_goal_pos = to_numpy(ts_goal_pos)
            np_action_pred = to_numpy(ts_action_pred)
            np_action_label = to_numpy(ts_action_label)
            #      dist_pred = batch_dist_pred[i]
            #      dist_label= batch_dist_label[i]
            info = str_data_info.split(' ')

            # data_info = f'{self.dataset_name} {f_curr} {curr_time} {f_goal} {goal_time} {goal_time - curr_time}'
            fig_title = f"dataset_name: %s \n %s %s \n %s %s \n frame diff: %d \nrel goal: (%0.3f, %0.3f) \n" % \
                        (info[0], info[1], info[2], info[3], info[4], int(info[4]) - int(info[2]), np_goal_pos[0],
                         np_goal_pos[1])

            ################################################################################
            # TODO: We need to come up with a better approaches for normalization !!!
            # if normalized:
            #     pred_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
            #     label_waypoint *= data_config[dataset_name]["metric_waypoint_spacing"]
            #     goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

            if config['normalize'] == True:  # denormalize pose to display them
                max_frame_dist = config['distance']['max_frame_dist']
                np_action_pred = _denormalize_pose(np_action_pred, max_frame_dist=max_frame_dist, eta=1.0,
                                                   dataset_name=dataset_name)
                np_action_label = _denormalize_pose(np_action_label, max_frame_dist=max_frame_dist, eta=1.0,
                                                    dataset_name=dataset_name)
                np_goal_pos = _denormalize_pose(np_goal_pos, max_frame_dist=max_frame_dist, eta=1.0,
                                                dataset_name=dataset_name)

            save_path = os.path.join(visualize_path, f"{str(tidx).zfill(4)}.png")
            compare_pred_to_label(
                fig_title,
                viz_obs_depth,
                viz_goal_depth,
                viz_obs_image,  # (640, 480)
                viz_goal_image,  # (640, 480)
                dataset_name,
                np_goal_pos,
                np_action_pred,
                np_action_label,
                #      dist_pred,
                #      dist_label,
                save_path,
                display = False,
            )
            tidx += 1

    # Log data to wandb/console, with visualizations selected from the last batch

    print("FINISHED Testing")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Depth Navigation ANN")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="../train/config/depth_nav.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("../train/config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)


    print(config)
    main(config)
