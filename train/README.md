# How to train the main navigation model

## 1. Split dataset into train and test dataset

>Executing data_split_former.py splits the real world dataset collected from FORMER robot.  
```
cd $PROJECT_DIR/train
conda activate devgru_train
python data_split_former -s 0.8 -i <IN_DATA_PATH> -o <OUT_DATA_PATH>
```
>$IN_DATA_PATH should contain the training dataset extracted. Although, we provide the full training dataset [LINK][], one can  
refer to [][] to collect a new dataset if necessary.  
>$OUT_DATA_PATH saves 

## 2. Train the model

### (1) Modify config file

>Modify depth_nav.yaml file located under $PROJECT/config/ folder. Make sure to correctly set paths: (1) data_folder, (2) train, and (3) test dirs located in depth_nav['datasets']['former'] section.
These folders are associated with the <IN_DATA_PATH> and <OUT_DATA_PATH> specified in the previous step.
>Lastly, set batch_size = 256.
>Then, execute the following command
```
cd $PROJECT_DIR/train
conda activate devgru_train
python train_former.py
```

# How to collect real-world dataset
>The training process should be reproducible with the provided training dataset [][]. It is not recommended to collect new dataset unless the user knows exactly what he/she is supposed to do. 
>Neverthess, how to collect training dataset is recorded in this page for the sake of reproduciblility of the research from the scratch.

## 1. How to collect nav dataset
```
cd $catkin_ws/src/navdata_collector/run_script/data_collector
conda activate navdata
python run_manual_data_collection.py

```

## 2. How to collect collision dataset

>The collision data set is provided in [Link][], but how to collect new collision dataset is recorded in this [Link][]. 
The purpose of this manual is to increase the reproduciblility of this research. 
Thus, you are not recommended to collect new collision dataset unless you want to do this research again from the scratch.

## 3. How to train collision enhanced model

### (1) Split dataset into train and test dataset

### (2) Execute 

```
cd $PROJECT_DIR/train
conda activate devgru_train
python train_collision_avoidance.py
```



