# How to train DevGRU models

Make sure to download install [DevGRU project](#) prior to proceeding the process below.

## 1. Split dataset into train and test dataset

Open the devgru.yaml located in under /devgru/config/  
Modify the devgru.yaml file. Make sure to correctly set paths: (1) data_folder, (2) train, and (3) test dirs located in devgru['data_split']['former'] section.

Executing the following cmds to split the real world dataset collected from FORMER robot.  

```
cd $PROJECT_DIR/train
conda activate devgru_train
python split_data.py
```

Here, $PROJECT_DIR refers to where the devgru project is located e.g) /home/$USER$python_ws/devgru

## 2. Train the model

### To train Action Prector (AP)
<!-- Open devgru.yaml under $PROJECT_DIR/devgru/  
#Set `batch_size=256`    -->

Execute the following to begin the training session
```
cd $PROJECT_DIR/train 
conda activate devgru_train
python train_action_predictor.py
```

### To train Collision Predictor (CP)

```
cd $PROJECT_DIR/train 
conda activate devgru_train
python train_collision_predictor.py
```

The trained models and their logs are saved under $PROJECT_DIR/train/logs using the time and date as the ID of the training session.


<!-- ## How to collect real-world dataset
>The training process should be reproducible with the provided training dataset [][]. It is not recommended to collect new dataset unless the user knows exactly what he/she is supposed to do. 
>Neverthess, how to collect training dataset is recorded in this page for the sake of reproduciblility of the research from the scratch.

## 1. How to collect nav dataset
```
cd $catkin_ws/src/navdata_collector/run_script/data_collector
conda activate navdata
python run_manual_data_collection.py
```

## 2. How to collect collision dataset -->

<!-- >The collision data set is provided in [Link][], but how to collect new collision dataset is recorded in this [Link][]. 
The purpose of this manual is to increase the reproduciblility of this research. 
Thus, you are not recommended to collect new collision dataset unless you want to do this research again from the scratch. -->




