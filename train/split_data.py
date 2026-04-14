import argparse
import os
import shutil
import random
import scipy
import numpy as np
import pickle
import glob
import csv
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append('../')
import utils.rigid_motion as rm
import yaml
from os.path import dirname, abspath

BASE_DIR = dirname(dirname(abspath(__file__)))  # proj root dir

# Data split to train & test dataset for the training
# 1. You should have processed the bagfiles using the scripts in navdata_collector/runscript/data_extractor/ folder
# 2. the processed (synced) files should be under /media/results/navdata_collector/processed/~~~~/bag_2025_~~~~/synced/
# 3. Make sure to run this script before processing the two scripts above

def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

def main(config):
    ############################################################
    # gen pkl files from mat file
    ############################################################
    split_config = config["data_split"]
    base_dir = split_config["former"]["data_folder"]

    data_dirs = glob.glob(f"{base_dir}/data*")
    data_dirs.sort()

    print(f"Found {len(data_dirs)} data directories")

    # Generate traj_data.pkl for each data directory
    for data_dir in tqdm(data_dirs, desc="Generating traj_data.pkl", unit="dir"):
        pose_file = f"{data_dir}/pose_context_m.txt"  # x, y, qx, qy, qz, qw

        if not os.path.exists(pose_file):
            print(f"[Warning] Missing pose file: {pose_file}")
            continue

        pose_dat = np.loadtxt(pose_file)
        num_pose = len(pose_dat)

        # Handle case where pose_dat is 1D because file has only one line
        if pose_dat.ndim == 1:
            pose_dat = np.expand_dims(pose_dat, axis=0)
            num_pose = 1

        position = np.zeros((num_pose, 2), dtype="double")
        orientation = np.zeros((num_pose, 1), dtype="double")
        pkl_file = f"{data_dir}/traj_data.pkl"

        for idx in tqdm(
            range(num_pose),
            desc=f"Processing {os.path.basename(data_dir)}",
            unit="pose",
            leave=False
        ):
            xy = pose_dat[idx][0:2].astype(float)
            quat = pose_dat[idx][3:].astype(float)  # quat, [qw, qx, qy, qz]

            htm = rm.quat_to_htm(quat)
            (_, _, _, rol, pit, yaw) = rm.htm_to_xyzrpy(htm)

            position[idx] = xy
            orientation[idx] = yaw

        dict_pose = {
            "position": position,
            "orientation": orientation
        }

        with open(pkl_file, "wb") as f:
            pickle.dump(dict_pose, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Get the names of the folders in the data directory
    folder_names = [
        f for f in data_dirs
        if os.path.isdir(f)
        # and "traj_data.pkl" in os.listdir('%s/synced' % f)
    ]

    # Randomly shuffle the names of the folders
    random.shuffle(folder_names)

    # Split the names of the folders into train and test sets
    split_ratio = split_config["split_ratio"]
    split_index = int(split_ratio * len(folder_names))

    train_folder_names = folder_names[:split_index]
    test_folder_names = folder_names[split_index:]

    print(f"Train folders: {len(train_folder_names)}")
    print(f"Test folders : {len(test_folder_names)}")

    # Create directories for the train and test sets
    train_dir = split_config["former"]["train"]
    test_dir = split_config["former"]["test"]

    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")
            remove_files_in_dir(dir_path)
        else:
            print(f"Creating {dir_path}")
            os.makedirs(dir_path)

    # Write the names of the train and test folders to files
    train_txt = os.path.join(train_dir, "traj_names.txt")
    test_txt = os.path.join(test_dir, "traj_names.txt")

    with open(train_txt, "w") as f:
        for folder_name in tqdm(train_folder_names, desc="Writing train split", unit="file"):
            f.write(folder_name + '\n')

    with open(test_txt, "w") as f:
        for folder_name in tqdm(test_folder_names, desc="Writing test split", unit="file"):
            f.write(folder_name + '\n')


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="../config/devgru.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        devgru_config = yaml.safe_load(f)

    main(devgru_config)
    print("Done")

