
# Test the Trained `model_weights` on a New Collision Dataset

## 1. Set Model Weights
1. Open `depth_nav.yaml`
2. Modify the following fields to point to the model weights you want to test:
   - `depth_nav['deployment']['nav_ckpth_name']`
   - `depth_nav['deployment']['col_ckpth_name']`

---

## 2. Set the Dataset Directory
1. Open `test_colldata.py`
2. Set `base_data_folder` to the directory containing the collision dataset  
   Example:
   
   /media/data/mydata/former_datasets/colldata/colldata-all/bag_****
   
## 3. Run `test_colldata.py`
```
cd /home/hankm/python_ws/viznav/depth-nav/test
python test_colldata.py
```
