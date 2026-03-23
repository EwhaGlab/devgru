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
import yaml

import sys
sys.path.append('../')
import rigid_motion as rm

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(dirname(abspath(__file__)))) # proj root dir
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
    ############################################################
    # gen pkl files from mat file
    ############################################################
    base_dir = args.base_dir
    os.path.dirname(base_dir)
    data_dirs = glob.glob('%s/path_*'%base_dir, recursive=True)

    bHc = rm.xyzrpy_to_htm(data_config['isaac_sim']["camera_matrics"]["cam_wrt_base"])
    cHb = np.linalg.inv(bHc)

    for data_dir in(data_dirs):

        pose_csv = '%s/robot_pose.csv'%(data_dir)        
        out_pkl_file = '%s/traj_data.pkl'%data_dir
        out_mat_file = '%s/traj_data.mat'%data_dir # for debugging
        # read pose_csv
        df = pd.read_csv(pose_csv)
        pose = df.to_numpy() #np.genfromtxt(pose_csv, delimiter=',')
        #print( '%s\n %d %d %d'%(data_dir, len(rgbfiles), len(depthfiles), pose.shape[0] ) )
        
        num_pose = len(pose)
        position = np.zeros([num_pose, 2], dtype='double')
        orientation = np.zeros([num_pose, 1], dtype='double')
        for idx in range(0, num_pose):
            xyz = pose[idx][2:5].astype('float')
            q = pose[idx][5:].astype('float') # qx qy qz qw
            quat = [q[3], q[0], q[1], q[2]] # qw qx qy qz
            wHc = rm.quat_to_htm( quat )
            wHc[0,3] = xyz[0]
            wHc[1,3] = xyz[1]
            wHc[2,3] = xyz[2]
            wHb = np.matmul(wHc,  cHb)

            [xb, yb, zb, rol_b, pit_b, yaw_b] = rm.htm_to_xyzrpy(wHb)
            position[idx] = [xb, yb]       # base_link xy
            orientation[idx] = yaw_b    # base_link theta

        dict_pose = {'position': position, 'orientation': orientation}
        with open(out_pkl_file, 'wb') as f:
            pickle.dump(dict_pose, f, protocol=pickle.HIGHEST_PROTOCOL)

        scipy.io.savemat(out_mat_file, mdict=dict_pose)

if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir", "-i", help="Base dir of isaacsim_dataset", required=True
    )

    args = parser.parse_args()
    main(args)
    print("Done")
