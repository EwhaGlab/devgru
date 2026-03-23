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
sys.path.append('../')

import utils.rigid_motion as rm
import yaml

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(abspath(__file__))) # proj root dir

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


def main(args: argparse.Namespace):
    ############################################################
    # gen pkl files from mat file
    ############################################################
    base_dir = args.base_dir
    data_dirs = glob.glob('%s/data*'%base_dir)
    data_dirs.sort()
    
    for data_dir in(data_dirs):
        pose_file = '%s/pose_context_m.txt'%data_dir  # x, y, qx, qy, qz, qw
        pose_dat = np.loadtxt(pose_file)
        num_pose = len(pose_dat)
        position = np.zeros([num_pose, 2], dtype='double')
        orientation = np.zeros([num_pose, 1], dtype='double')
        pkl_file = '%s/traj_data.pkl'%data_dir

        for idx in range(0, num_pose):
            xy = pose_dat[idx][0:2].astype('float')
            quat = pose_dat[idx][3:].astype('float') # quat, [qw, qx, qy, qz]
            htm = rm.quat_to_htm(quat)
            (_, _, _, rol, pit, yaw) = rm.htm_to_xyzrpy(htm)
            position[idx] = xy
            orientation[idx] = yaw
        dict_pose = {'position': position, 'orientation': orientation}
        with open(pkl_file, 'wb') as f:
            pickle.dump(dict_pose, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
    folder_names = [
        f
        for f in data_dirs
        if os.path.isdir(f)
        #and "traj_data.pkl" in os.listdir('%s/synced'%f)
    ]

    # Randomly shuffle the names of the folders
    random.shuffle(folder_names)

    # Split the names of the folders into train and test sets
    split_index = int(args.split * len(folder_names))
    train_folder_names = folder_names[:split_index]
    test_folder_names = folder_names[split_index:]

    # Create directories for the train and test sets
    train_dir = os.path.join(args.data_splits_dir, "train")
    test_dir = os.path.join(args.data_splits_dir, "test")
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")
            remove_files_in_dir(dir_path)
        else:
            print(f"Creating {dir_path}")
            os.makedirs(dir_path)

    # Write the names of the train and test folders to files
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for folder_name in train_folder_names:
            f.write(folder_name+'\n')

    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for folder_name in test_folder_names:
            f.write(folder_name+'\n')

if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir", "-i", help="Base dir of Former_Dataset", required=True
    )
    # parser.add_argument(
    #     "--data-list-file", "-l", help="The file containing the data list", required=True
    # )
    parser.add_argument(
        "--split", "-s", type=float, default=0.8, help="Train/test split (default: 0.8)"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", default="/media/mydata/viznav/data_splits/former_collision", help="Data splits directory"
    )
    args = parser.parse_args()
    main(args)
    print("Done")
