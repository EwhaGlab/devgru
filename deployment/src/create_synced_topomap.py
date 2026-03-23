import argparse
import os
import time
import numpy as np

import yaml
import shutil
from os.path import dirname, abspath
BASE_DIR = dirname( dirname(dirname(abspath(__file__))) )
import sys
sys.path.append(BASE_DIR)
#print(BASE_DIR)

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
    #global obs_img
#   rospy.init_node("CREATE_TOPOMAP", anonymous=False)

    sync_topomap_dir = args.topomap_dir
    input_dir = args.input_dir
    #sync_topomap_dir = topomap_root_dir
    use_config_file = args.config

    with open("%s/config/defaults.yaml" % BASE_DIR, "r") as f:
        default_config = yaml.safe_load(f)

    with open("%s" % use_config_file, "r") as f:
        user_config = yaml.safe_load(f)

    config = default_config
    config.update(user_config)  
    
    config_deployment = config['deployment']
    sync_data_dir = '%s/synced'%input_dir
    #if not os.path.isdir(sync_topomap_dir):
        #os.makedirs(sync_topomap_dir)
    #else:
        #print(f"{sync_topomap_dir} already exists. Removing previous topomap...")
        #remove_files_in_dir(sync_topomap_dir)

    ts = 10 #config_deployment['subgoal_spacing']
    ws = config_deployment['waypoint_spacing']

    sync_odom_file = '%s/sync_odom.txt' % sync_data_dir
    sync_odom   = np.loadtxt(sync_odom_file)
    sync_tf_m2b      = np.loadtxt('%s/sync_tf_m2b.txt'% sync_data_dir)
    sync_tf_m2o      = np.loadtxt('%s/sync_tf_m2o.txt'% sync_data_dir)

    num_rgb = len( sync_odom ) # == num_odom
    topo_odom = sync_odom[0:num_rgb:ts]
    topo_m2b  = sync_tf_m2b[0:num_rgb:ts]
    topo_m2o  = sync_tf_m2o[0:num_rgb:ts]

    # copy topo rgb and depth image
    # get rgb index
    num_nodes = len( topo_m2b)
    topo_rgb_idx = list( range(0, num_rgb, ts) )
    rgb_filenames = [f"rgb{idx:05d}.png" for idx in topo_rgb_idx]
    depth_filenames = [f"depth{idx:05d}.png" for idx in topo_rgb_idx]

    for fname in rgb_filenames:
        src = os.path.join(sync_data_dir, fname)
        dst = os.path.join(sync_topomap_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            #print(f"Copied {fname}")
        else:
            print(f"\033[91mFile not found: {fname}\033[0m")

    #print(f"Finished copying rgb files to %s"%topomap_dir )
    for fname in depth_filenames:
        src = os.path.join(sync_data_dir, fname)
        dst = os.path.join(sync_topomap_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"\033[91mFile not found: {fname}\033[0m")

    #print(f"Finished copying depth files to %s" % topomap_dir)

    topo_odom_file = '%s/topo_odom.txt' % sync_topomap_dir
    topo_tf_m2b_file = '%s/topo_tf_m2b.txt' % sync_topomap_dir
    topo_tf_m2o_file = '%s/topo_tf_m2o.txt' % sync_topomap_dir
    np.savetxt(topo_odom_file, topo_odom)
    np.savetxt(topo_tf_m2b_file, topo_m2b)
    np.savetxt(topo_tf_m2o_file, topo_m2o)

    print(f'Finished creating %d topo nodes' % num_nodes )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Navigation ANN")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="%s/config/depth_nav.yaml"%BASE_DIR,
        type=str,
        help="Path to the config file in train_config folder",
    )
    parser.add_argument(
        "--topomap-dir", "-o", help="topomap dir to process", required=True
    )

    parser.add_argument(
        "--input-dir", "-i", help="input metadata dir containing a synced folder", required=True
    )

    args = parser.parse_args()

    print("base: %s" %args.config)

    args = parser.parse_args()
    main(args)
