# DevGRU: Depth-guided Visual Navigation using a Collision-aware Recurrent Model

**Contributors**: Kyung Min Han, Young J. Kim  

[Project Page](#) | [Paper](#) | [Pre-Trained Models](#) | [Dataset](#)

---

DevGRU is a deep learning-based topological navigation framework designed for robust long-horizon navigation without requiring global localization or a dense obstacle map. 

Given a pre-built topological map, DevGRU sequentially visits intermediate sub-goals to reach the final destination. 

Unlike prior goal-conditioned navigation models that directly predict actions from observations alone, DevGRU explicitly leverages both depth observations and point-goal information for navigation, making it a depth- and point-goal-conditioned navigation system.

---

Deploying DevGRU on real-robot requires two SW packages:

1. **DevGRU** repo (the current repo)  
  
2. [**NavDataCollector**](#) repo 
   

## Acknowledgment

This codebase builds upon the implementation of [ViNT](https://github.com/...) (Vision-based Navigation Transformer).  

We thank the authors for making their code publicly available.

Parts of the data processing pipeline and source code are adapted from the ViNT project, with modifications to support the DevGRU framework.

---

## Overview

This repository contains code for training and deploying **DevGRU** for topological navigation using depth and sub-goal information.

- `./train/`  
  training code for DevGRU models.

- `./train/devgru_train/models/`  
  contains model definitions for the Action Predictor (AP), Collision Predictor (CP), and related baselines.

- `./deployment/src/`  
  deployment scripts for running DevGRU on a robot platform in real environments.

<!-- - `./deployment/src/record_bag.sh`  
  script to collect demonstration trajectories for building topological maps.

- `./deployment/src/create_topomap.sh`  
  script to convert a recorded trajectory into a topological map representation. -->

---
## 1. How to Setup

### (1) Setup NavDataCollector

The main purpose of this package is for collecting dataset and deploying the trained model.  
Make sure to have them installed by executing the commands below.

`cd ~/catin_ws/src`  
`git clone https://github.com/han-kyung-min/navdata_collector.git`  
`cd ~/catkin_ws`  
`catkin_make install`  

### (2) How to Setup DevGRU project and Training Env

#### ROS1 with Ubuntu 20.04 

Make sure to install [ROS-noetic](#) and [Conda](#) prior to the following procedures.  

- Clone DevGRU project

`git clone https://github.com/han-kyung-min/devgru`  
`git checkout master`

- Set up the conda environment:

`conda env create -f train/train_environment.yml`

- Install DevGRU train packages

`pip install -e train/`

`conda activate devgru_train`

### (3) How to Set up Deployment Env

Conda

- Install anaconda/miniconda/etc. for managing environments. We recommend you [Miniconda](#)

Make conda env with environment.yml (run this inside the vint_release/ directory)

`Conda env create -f deployment/deployment_environment.yaml`  

Source env  
`conda activate devgru_deployment`

<!-- (Recommended) add to ~/.bashrc:
echo “conda activate vint_deployment” >> ~/.bashrc  -->

<!-- Install the vint_train packages (run this inside the vint_release/ directory):  
`pip install -e train/` -->


## 2. [How to Train Models](#) 

<!-- ### Download dataset
The DataSet used to train DevGRU is avaliable at this [DataSet Link](#) 

### Train

Run the following procedure to train Action Predictor (AP) model -->


## 3. [How to conduct the Autonomous Navigation]()

<!-- 
We encourage you to download our [PreTrained-Models](#). Place them under `devgru/deployment/model_weights/`.
You can also train the model again. If you want to do so please refer to [How to train models](#).  -->



---

