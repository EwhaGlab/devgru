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
#print("%s\n\n"%BASE_DIR)

with open("%s/config/data_config.yaml"%BASE_DIR, "r") as f:
    data_config = yaml.safe_load(f)

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

    #TODO: gen pkl files (call gen_pkl... in data )
    ############################################################
    # gen pkl files from mat file
    ############################################################
    base_dir = args.base_dir
    os.path.dirname(base_dir)
    data_dirs = glob.glob('%s/path_*'%base_dir, recursive=True)

    bHc = rm.xyzrpy_to_htm(data_config['isaac_sim']["camera_matrics"]["cam_wrt_base"])
    cHb = np.linalg.inv(bHc)

    for data_dir in(data_dirs):

        rgb_dir = '%s/rgb'%(data_dir)
        depth_dir = '%s/depth'%(data_dir)

        rgbfiles = glob.glob('%s/rgb*.png'%rgb_dir)
        depthfiles = glob.glob('%s/depth*.png'%depth_dir)
        
        # read pose_csv
        #print( '%s\n %d %d %d'%(data_dir, len(rgbfiles), len(depthfiles), pose.shape[0] ) )
        #assert(len(rgbfiles) == len(depthfiles) == pose.shape[0] )

    # Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
    folder_names = [
        f
        for f in data_dirs
        if os.path.isdir(f)
        and "traj_data.pkl" in os.listdir(f)
    ]
    # Randomly shuffle the names of the folders
    folder_names.sort()
    #random.shuffle(folder_names)
    
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
            f.write(folder_name + "\n")

    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for folder_name in test_folder_names:
            f.write(folder_name + "\n")


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir", "-i", help="Base dir of isaacsim_dataset", required=True
    )
    parser.add_argument(
        "--split", "-s", type=float, default=0.8, help="Train/test split (default: 0.8)"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", default="/media/data/mydata/viznav/data_splits/isaacsim/dataname", help="Data splits directory"
    )
    args = parser.parse_args()
    main(args)
    print("Done")
