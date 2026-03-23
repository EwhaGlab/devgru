import argparse
import os
import shutil
import random
import scipy
import numpy as np
import pickle

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
    is_real = int(args.is_real)
    print("is real: %d"%is_real)
    base_dir = args.base_dir
    if is_real :
        data_dir = '%s/Real_Scenes'%base_dir
        data_type = 'real'
    else:
        data_dir = '%s/Synthetic_Scenes'%base_dir
        data_type = 'synth'
    
    data_list_file = args.data_list_file
    thud_data_list = '%s/../../%s'%(data_dir, data_list_file)
    fid = open(thud_data_list,'r')
    input_dirs = fid.read().splitlines()
    fid.close()
    for input_dir in(input_dirs):
        if is_real :
            pos_dir = '%s/%s/Label/Pose'%(data_dir, input_dir)
        else:
            pos_dir = '%s/%s/Label'%(data_dir, input_dir)

        img_dir = '%s/%s/RGB'%(data_dir, input_dir)
        pkl_file = '%s/../traj_data.pkl'%img_dir
        mat_file = '%s/pos.mat'%pos_dir
        if True: #os.path.isfile( pkl_file ) is False: # no pkl file. Need to create one
            pose_mat = scipy.io.loadmat(mat_file)
            pose_dat = pose_mat['pose'][0]
            num_pose = len(pose_dat)
            position = np.zeros([num_pose, 2], dtype='double')
            yaw = np.zeros([num_pose, 1], dtype='double')
            for idx in range(0, num_pose):
                xyz = pose_dat[idx][0].astype('float')
                theta = pose_dat[idx][1].astype('float')[0][0]
                position[idx] = xyz[0:2].squeeze()
                yaw[idx] = theta
            dict_pose = {'position': position, 'yaw': yaw}
            with open(pkl_file, 'wb') as f:
                pickle.dump(dict_pose, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
    folder_names = [
        f
        for f in input_dirs
        if os.path.isdir(os.path.join(data_dir, f))
        and "traj_data.pkl" in os.listdir(os.path.join(data_dir, f))
    ]
    # Randomly shuffle the names of the folders
    random.shuffle(folder_names)

    # Split the names of the folders into train and test sets
    split_index = int(args.split * len(folder_names))
    train_folder_names = folder_names[:split_index]
    test_folder_names = folder_names[split_index:]

    # Create directories for the train and test sets
    train_dir = os.path.join(args.data_splits_dir, "thud/%s/train"%data_type)
    test_dir = os.path.join(args.data_splits_dir, "thud/%s/test"%data_type)
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
        "--base-dir", "-i", help="Base dir of THUD_Dataset", required=True
    )
    parser.add_argument(
        "--is-real", "-r", help="Name of the dataset", required=True 
    )
    parser.add_argument(
        "--data-list-file", "-l", help="The file containing the data list", required=True
    )
    parser.add_argument(
        "--split", "-s", type=float, default=0.8, help="Train/test split (default: 0.8)"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", default="vint_train/data/data_splits", help="Data splits directory"
    )
    args = parser.parse_args()
    main(args)
    print("Done")
