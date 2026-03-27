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
from warmup_scheduler import GradualWarmupScheduler

"""
IMPORT YOUR MODEL HERE
"""

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(BASE_DIR)
#print("base dir from train_isaac:", BASE_DIR)

from models.image_rnn.devgru_ap import DevGRU
#from depth_nav_train.data.depth_nav_dataset import
from data.devgru_dataset import DevGRU_Dataset
from devgru_train.train_eval_loop import (
    train_eval_loop,
    load_model,
)

def load_legacy_checkpoint_weights(model, ckpt_path, device="cpu", strict=False):

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        # Some files are already bare state_dicts
        state_dict = ckpt
    else:
        inner = ckpt["model"]
        if isinstance(inner, nn.Module):
            state_dict = inner.state_dict()
        elif isinstance(inner, dict):
            state_dict = inner  # already a state_dict
        else:
            raise TypeError(f"Unsupported type for ckpt['model']: {type(inner)}")

    # Strip 'module.' if it was saved from DP/DDP
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print("[load_state_dict] missing:", missing)
    print("[load_state_dict] unexpected:", unexpected)

    return ckpt  # (optional) you can still inspect epoch, metrics, etc.


def main(config):

    assert config["distance"]["min_frame_dist"] < config["distance"]["max_frame_dist"]
    assert config["action"]["min_frame_dist"] < config["action"]["max_frame_dist"]

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
    transform = ([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    #    transform = ([transforms.Normalize(mean=[1, 1, 1], std=[0, 0, 0]),])
    transform = transforms.Compose(transform)  # data preprocessing ( e.g. data augmentation )

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        print(data_config)
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1
        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = DevGRU_Dataset(
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
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)
    # next(iter(train_loader))  # to access a batch of dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle= False, #True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

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

    if config["model_type"] == "devgru_ap":
        if config["goal_type"] == "rgb":
            model = DevGRU(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                odom_encoding_size=config["odom_encoding_size"],
                final_dim=32,
                num_ch=3,
            )
        else:
            model = DevGRU(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                odom_encoding_size=config["odom_encoding_size"],
                final_dim=32,
                num_ch=1,
            )
    else:
        raise ValueError(f"Model {config['model_type']} not supported")

    if config["clipping"]:
        max_norm = 1.
        print("Clipping gradients to", max_norm)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * max_norm, max_norm
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            cyclic_period = 10
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=cyclic_period // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            plateau_patience = 3
            plateau_factor= 0.5
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=plateau_factor,
                patience=plateau_patience,
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            warmup_epochs = 4
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=warmup_epochs,
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # load model weight

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    print_log_freq = 100
    image_log_freq = 1000
    print("START TRAINING")
    if config["model_type"] == "devgru_ap":
        train_eval_loop(
            goal_type=config["goal_type"],
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            beta =config["beta"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    else:
        raise ValueError(f"Model {config['model_type']} is not supported")

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Depth Navigation ANN")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="../config/devgru.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("../config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)
    #run_name = 'action_pred'
    run_name = "action_pred_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], run_name #config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="hankm", # TODO: change this to your wandb entity
        )
        base_path = os.path.dirname(args.config)
        wandb.save(args.config, base_path=base_path, policy="now")  # save the config file
        wandb.run.name = run_name
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
