
# How to do the navigation using the trained model.

## 1. To generate topomap.
### (1) Record bag file while driving the robot manually.

>First, modifiy the navdata_collector.yaml located under ~/catkin_ws/src/navdata_collector/params/ to set the path to save the bagfile.
>This config file contains several parameters needed for run_manual_data_collection.py 
>In the yaml file, make sure to modify navdata_collector['navdata_collector']['bagfile_root_path'] according to the path to save bagfiles.
>Then, compile navdata_collector pkg.
```
cd ~/catkin_ws
catkin_make install
```

>Then, the the following commands to record bagfile for a topology map.

```
cd $PROJECT_DIR/deployment/src
./record_topomap_bag.sh
```
>$PROJECT_DIR refers to the root dir of DevGRU project.
>This command starts run_manual_data_collection.py installed in the ~/catkin_ws/install/lib.
>The process records a bag file containing the ros msgs. This program also saves SLAM map files (posegraph and data) at the end of the process.

## 2. Extract and copy the extracted data to TOPOMAP_DIR

>Similar to the previous step, modify navdata_collector.yaml to set correct $INPATH and $OUTPATH to extract the bagfile data.
>Specifically, modify navdata_collector['navdata_extractor']['inpath'] and navdata_collector['navdata_extractor']['outpath']
>$INPATH could be the folder generated from the previous data collection step whose name has YYYY-MM-DD-HH-MM. The '$INPATH normally contains child bagfile folders such as bag_YYYY-MM-DD-HH-MM-SS. $OUTPATH must be a physically different folder from $OUTPATH.

>execute the following command to generate the topomap where $EXTRACTED_DATA_DIR is must be identical to $OUTPATH specified above. $TOPOMAP_DIR the final destination of topology map.

```
cd $PROJECT_DIR/deployment/src
./generate_topomap.sh <EXTRACTED_DATA_DIR> <TOPOMAP_DIR>
```

>This process extracts and syncs the metadata stored in the bag file recorded in the previous step.
>Then, copies them to $TOPOMAP_DIR, followed by generating slam_poses.txt and the final topopmap

    
## 3. Experiment the autonomous navigation

>If you have completed the step 1 and 2, you are ready to start autonomous navigation.
>Run the following command to exectute the navigation process.
```
cd $PROJECT_DIR/deployment/src
./navigate_viz.sh
```

# How to collect collision dataset.

>This process might not applicable to other users because the procedure is not very generic and it is robot specific. 
>The final train model provided in this research should have trained for the collision dataset.
>Nevertheless, the procedure is written as below for the sake of reproducibility of this research.
>You have to have a topology map ready a prior to collect collision dataset. Thus, make sure to complete the step 1 and 2 mentioned above before proceeding this step.

## 1. Collision data bagging

>The main purpose of this procedure is collecting collision event during a navigation process. 
>To start the process, run the following commands

```
cd $PROJECT_DIR/deployment/src
./navigate_w_colldata_bagging.sh <TOPOMAP_DIR>
```

>This script runs (1) navigate_w_colldata_bagging.py, (2) run_colldata_collection.py, and (3) pd_controller.py, in turn.
>navigate_w_colldata_bagging.py loads the pretrained nav model to predict output waypoints, given input depth images and previous poses.
>run_colldata_collection.py creates a bagfile and collect critical ros msgs.
>pd_controller.py controls the robot based on the predicted waypoint outputs generated from the pretrained network model.


                         _______________________________
        ________________/                               \________________
       /   L1     L2   /     _________       _________   \   R2     R1   \
      /_______________/     |         |     |         |    \______________\
     /   ___      ___   \   |  TOUCH  |     |  TOUCH  |    /   ___  ___   \
    |   (___)    (___)   |  |   PAD   |     |   PAD   |   |   (___)(___)   |
    |                     |  |_________|     |_________|   |               |
    |   D-PAD                                  Buttons    |   □    O    X  |
    |  [↑]                                               |                 |
    | [←][→]         SHARE      ( PS )      OPTIONS      |   △            |
    |  [↓]           ___          ○            ___       |  ___    ___    |
    |               (LS)                     (RS)        | (___)  (___)   |
     \                                                                    /
      \____________________ _________________________ ___________________/
                           \_________________________/

>Push L1 button (deadman switch) with appropriate joystick control (LS and RS sticks) to avoid the robot collision from obstacles. 
>This L1 button pushing event is supposed to be recorded in the bagfile, and this msg is used for creating collision dataset, later.

## 2. Collision data generation


```
cd ~/catkin_ws/src/navdata_collector/run_script/data_extractor/potmap2d
```

>Then, execute script_colldata_collection.m in MATLAB window. 
>This matlab script should be changed to python code in the future..










