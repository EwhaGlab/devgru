# DevGRU

DevGRU is a deep learning-based topological navigation framework designed for long-horizon navigation without requiring global localization or a dense obstacle map.

Given a pre-built topological map, the robot sequentially visits intermediate sub-goals to reach the final destination. DevGRU leverages both depth observations and point-goal information for navigation.

## Dependencies

DevGRU is designed to operate within a ROS-based Conda Env:

- Ubuntu 20.04  
- [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)  
- [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main)   

---

## Packages to install

To run DevGRU, two main components must be installed:

- **DevGRU (this repository)**: training and deployment framework  
- **NavDataCollector** ([link](#)): dataset collection and deployment support  

---
## Overview

This repository provides code for training and deploying **DevGRU** for topological navigation using depth and sub-goal information.

- `./train/`: training code for DevGRU models  
- `./train/devgru_train/models/`: model definitions for the Action Predictor (AP), Collision Predictor (CP), and related baselines  
- `./deployment/src/`: deployment scripts for running DevGRU on a robot platform in real-world environments  

<hr style="height:4px; background-color:black; border:none;">

## 1. How to Setup

### (1) Setup NavDataCollector

This package is used for dataset collection and real-world deployment. Install it using the following commands:

```bash
cd ~/catkin_ws/src
git clone https://github.com/han-kyung-min/navdata_collector.git
cd ~/catkin_ws
catkin_make install
```
---
<a id="setup-devgru"></a>
### (2) To Setup DevGRU project and Training Env

<!--#### ROS1 with Ubuntu 20.04 -->

Make sure to install [ROS-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) and [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) prior to the following procedures.  

- Clone DevGRU project

`git clone https://github.com/han-kyung-min/devgru`  
`cd devgru`  
`git checkout master`

- Set up the conda environment:

`conda env create -f train/train_environment.yml`

<!-- Install DevGRU train packages
`pip install -e train/`-->

To activate env  
`conda activate devgru_train`

---
### (3) To Set up Deployment Env

- Install Conda for managing environments. We recommend you [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

Make conda env with environment.yml 

`conda env create -f deployment/deployment_environment.yml`  

To activate env  
`conda activate devgru_deployment`

<!-- Install the vint_train packages (run this inside the vint_release/ directory):  
`pip install -e train/` -->

<hr style="height:4px; background-color:black; border:none;">

<a id="howto-train"></a>
## 2. How to Train Models

>Make sure to install the [DevGRU project](#setup-devgru) before proceeding with the steps below.


### (1) Split Dataset into Training and Test Sets

Open the `devgru.yaml` file located at: /devgru/config/

Modify the configuration to correctly set the dataset paths in the following section:  
devgru['data_split']['former']  

Ensure that the following fields are properly specified:

>i. `data_folder`  
>ii. `train`  
>iii. `test` 

Run the following commands to split the real-world dataset collected from the FORMER robot.

```
cd $PROJECT_DIR/train
conda activate devgru_train
python split_data.py
```

Here, `$PROJECT_DIR` refers to the root directory of the DevGRU project (e.g., `/home/$USER/python_ws/viznav/devgru`).

---
### (2) To train Action Prector (AP)
<!-- Open devgru.yaml under $PROJECT_DIR/devgru/  
#Set `batch_size=256`    -->

Open the `devgru.yaml` file located at: $PROJECT_DIR/config/  
Modify the configuration to correctly set the dataset paths in the following section:  devgru['datasets']['former']  

Ensure that the following fields are properly specified:

>i. `data_folder`  
>ii. `train`  
>iii. `test` 

Run the following commands to start training:  
```
cd $PROJECT_DIR/train 
conda activate devgru_train
python train_action_predictor.py
```
---
### (3) To train Collision Predictor (CP)

Open the `devgru.yaml` file located at: $PROJECT_DIR/config/  
Modify the configuration to correctly set the dataset paths in the following section:  
devgru['datasets']['former']

Ensure that the following fields are properly specified:

>i. `data_folder`  
>ii. `train`  
>iii. `test` 

Run the following commands to start training:  
```
cd $PROJECT_DIR/train 
conda activate devgru_train
python train_collision_predictor.py
```

The trained models and their logs are saved under $PROJECT_DIR/train/logs using the time and date as the ID of the training session.


<hr style="height:4px; background-color:black; border:none;">

## 3. How to Conduct Autonomous Navigation

Make sure that both the [DevGRU project](#setup-devgru) and [NavDataCollector](https://github.com/han-kyung-min/navdata_collector) are installed before proceeding with the steps below.

Although users can train models by following the [training step](#howto-train)  
we also provide pre-trained models available at [THIS LINK](#)

The trained models should be copied to $PROJECT_DIR/deployment/model_weights

### (1) Generate a Topological Map

<a id="init-odom"></a>
#### (a) Initialize the Robot

Move the robot to the starting position and reset the odometry.

> For Former robot users, run in the robot's terminal:
> ```bash
> sudo service former stop
> sudo service former start
>```
> Alternatively, rebooting the robot will also reset the odometry.

#### (b) Record bag file while driving the robot manually.

Next, modifiy the navdata_collector.yaml located under ~/catkin_ws/src/navdata_collector/params/ to specify the path for saving bag files.

<!--This configuration file contains parameters required for `run_manual_data_collection.py`.-->  

In particular, update the following field:
navdata_collector['navdata_collector']['bagfile_root_path'] to your desired directory for storing bag files.

<!--After updating the configuration, compile the `navdata_collector` package:

```bash
cd ~/catkin_ws
catkin_make install
```
-->

Then, Follow the proceedure below:  
> (i) Run the recording script  
>``cd $PROJECT_DIR/deployment/src``  
>``./record_topomap_bag.sh``  
> (ii) Press 'y' and hit 'Enter' to start recording a bagfile.  
> (iii) Move the robot to explore the environment and record the topological map data.  
> (iv) Once the robot reaches the desired destination, press 'q' and hit 'Enter' to stop recording.

This command launches `run_manual_data_collection.py` installed under `~/catkin_ws/install/lib`.  
The process records a ROS bag file containing relevant messages and also saves SLAM map files (e.g., pose graph and associated data) upon completion.

---
### (2) Extract bag and copy the extracted data to TOPOMAP_DIR

Similar to the previous step, modify `navdata_collector.yaml` to set the correct paths for extracting bag file data.
Specifically, update the following fields:

navdata_collector['navdata_extractor']['inpath']  
navdata_collector['navdata_extractor']['outpath']  

- `$INPATH`: Directory generated from the previous data collection step (typically named in the format `YYYY-MM-DD-HH-MM`).    
  This directory usually contains subfolders such as `bag_YYYY-MM-DD-HH-MM-SS`.  
- `$OUTPATH`: Directory where the extracted data will be saved.  
  This must be a **different directory** from `$INPATH`.

After configuring the paths, execute the following command to generate the topological map from the recorded bagfile:
<!-- Make sure that ROS_MASTER_URI is set to http://127.0.0.1:11311 before running the following steps-->

```
source set_ros_local.sh
cd ~/catkin_ws/src/navdata_collector/run_script/data_extractor
python generate_topomap_from_bag.py
```

The first command sets ROS_MASTER_URI to 127.0.0.1, disconnecting the session from the robot’s ROS master.
The following procedure extracts and synchronizes the metadata from the bag files recorded in the previous step.  
The extracted data is then copied to `$TOPOMAP_DIR`, where `slam_poses.txt` and the final topological map are generated.
    
---
### (3) Run Autonomous Navigation


After completing Steps 1 and 2, you are ready to start autonomous navigation.  
Open the set_ros_former.sh script located in ~/catkin_ws/src/navdata_collector/run_script/data_extractor and modify the IP addresses as needed.

Then, do the following procedure:  

(a) Place the robot at the starting position of the topological map,  
(b) Ensure that the [robot’s odometry is initialized](#init-odom)  
(c) Execute the navigation process using the following commands: 
 
>```
>cd ~/catkin_ws/src/navdata_collector/run_script/data_extractor
>source set_ros_former.sh
>cd $PROJECT_DIR/deployment/src
>python run_navigate_viz.py
>```
>The first command configures ROS_MASTER_URI and ROS_HOSTNAME.   
The remaining commands execute the navigation procedure.  
>If you want to record navigation logs with SLAM (not required for navigation, but useful for quantitative analysis),  
>use the following command instead of `run_navigate_viz.py`.
>```
>cd $PROJECT_DIR/deployment/src  
>./run_navigate_viz_stack.py
>```

---
## 4. [How to Collection Collision Dataset](deployment/HOW_TO_COLLECT_COLLISION_DATA.md)

---
## Acknowledgment

This codebase builds upon the implementation of [ViNT](https://general-navigation-models.github.io/vint/) (Vision-based Navigation Transformer).  

We thank the authors for making their code publicly available.

Parts of the data processing pipeline and source code are adapted from the ViNT project, with modifications to support the DevGRU framework.

---
