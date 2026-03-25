import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

from devgru.eval_utils import evaluate
#from depth_nav_train.training.train_utils import train_nomad, evaluate_nomad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

def evaluate_loop(
    goal_type: str,
    model: nn.Module,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    num_images_log: int = 16,
    alpha: float = 0.9,
    beta: float = 0.5,
    learn_angle: bool = False, #True,
    eval_fraction: float = 0.25,
):
    """
    Train and evaluate the model

    Args:
        model: model to train
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        num_images_log: number of images to log to wandb
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1

    avg_total_test_loss = []
    for dataset_type in test_dataloaders:
        print(f"Start {dataset_type} DepthNav Testing ")
        loader = test_dataloaders[dataset_type]
        test_dist_loss, test_action_loss, total_eval_loss = evaluate(
            eval_type=dataset_type,
            goal_type=goal_type,
            model=model,
            dataloader=loader,
            transform=transform,
            device=device,
            project_folder=project_folder,
            normalized=normalized,
            alpha=alpha,
            beta=beta,
            learn_angle=learn_angle,
            num_images_log=num_images_log,
            eval_fraction=eval_fraction,
        )

        avg_total_test_loss.append(total_eval_loss)
        print("Finished the evaluation for test_datasets \n")


def load_model(
        model_path: str,
        config: dict,
        device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["deployment"]["model_type"]

    if model_type == "devgru":
        if config["goal_type"] == "rgb":
            model = ImagePoseRNN(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                odom_encoding_size=config["odom_encoding_size"],
                num_ch=3,
            )
        else:
            model = ImagePoseRNN(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                odom_encoding_size=config["odom_encoding_size"],
                num_ch=1,
            )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device)

    loaded_model = checkpoint["model"]
    try:
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict, strict=False)
    except AttributeError as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    return model


