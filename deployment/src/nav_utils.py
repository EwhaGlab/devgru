
import os
import sys
import io
import matplotlib.pyplot as plt

# ROS
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional
from cv_bridge import CvBridge
bridge = CvBridge()
# models

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.append(BASE_DIR)
#from models.image_rnn.image_rnn import ImageRNN
#from models.image_rnn.image_pose_rnn import ImagePoseRNN
from models.image_rnn.devgru_ap import DevGRU
from models.coll_pred.devgru_cp import DepthCollision
from models.coll_pred.depth_sg_coll import DepthSGCollision


MAX_DEPTH = 65535
def load_model(
    model_path: str,
    model_type: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    #model_type = config["deployment"]["model_type"]
    print("%s"%model_type)

    if model_type == "devgru_cp":
        if config["goal_type"] == "rgb":
            model = DepthCollision(
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                num_ch=3,
            )
        else:
            model = DepthCollision(
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                num_ch=1,
            )
    elif model_type == "depth_sg_coll":
        if config["goal_type"] == "rgb":
            model = DepthSGCollision(
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                num_ch=3,
                pretrained=True,
                dropout_p=0.2,
                freeze_backbone_bn=False
            )
        else:
            model = DepthSGCollision(
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                num_ch=1,
                pretrained=True,
                dropout_p=0.2,
                freeze_backbone_bn=False
            )
    elif model_type == "devgru_cp":
        if config["goal_type"] == "rgb":
            model = DevGRU(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                odom_encoding_size=config["odom_encoding_size"],
                final_dim=32, #64, #
                num_ch=3,
            )
        else:
            model = DevGRU(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                odom_encoding_size=config["odom_encoding_size"],
                final_dim=32, #64,
                num_ch=1,
            )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    loaded_model = checkpoint["model"]
    try:
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict, strict=False)
    except AttributeError as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image

def msg_to_pil_depth(msg: Image) -> PILImage.Image:
    np_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # dtype = uint16
    # Convert numpy array to PIL Image
    pil_depth = PILImage.fromarray(np_depth, mode='I;16')
    return pil_depth

def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)  
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def resize_and_normalized_depths(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:

    max_pix_val = np.max( np.array(pil_imgs) )
    assert max_pix_val > 1 and max_pix_val < MAX_DEPTH, f"max of raw depth img: {max_pix_val}"

    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    pil_depths = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_depth = pil_img.resize(image_size)
        pil_depths.append(pil_depth)
        #transf_img = transform_type(pil_img)
        #transf_img = torch.unsqueeze(transf_img, 0)
        #transf_imgs.append(transf_img)
    return torch.cat(pil_depths, dim=1)

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def msg_to_caminfo(msg) -> CameraInfo:
    cam_info = CameraInfo
    cam_info.header = msg.header
    cam_info.height = msg.height
    cam_info.width  = msg.width
    cam_info.K = msg.K
    cam_info.D = msg.D
    cam_info.R = msg.R
    cam_info.P = msg.P
    cam_info.binning_x = msg.binning_x
    cam_info.binning_y = msg.binning_y
    cam_info.roi = msg.roi
    K = np.asarray(cam_info.K).reshape(3,3)
    D = np.asarray(cam_info.D)
    R = np.asarray(cam_info.R).reshape(3,3)
    P = np.asarray(cam_info.P).reshape(3,4)
    return cam_info, K, D, R, P
    
def caminfo_msg_to_numpy(cam_info):
    K = np.asarray(cam_info.K)
    D = np.asarray(cam_info.D)
    R = np.asarray(cam_info.R)
    
def set_tform(data) -> TransformStamped:
    tform = TransformStamped
    tform.header = data.header
    tform.child_frame_id = data.child_frame_id
    tform.transform = data.transform
    return tform
    
