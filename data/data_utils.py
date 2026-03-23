import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union
import yaml

from os.path import dirname, abspath
BASE_DIR = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

import utils.rigid_motion as rm
VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training
MAX_DEPTH = 65535.

# LOAD DATA CONFIG
with open(BASE_DIR+"/config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

def get_data_path(f: str, time: int, data_type: str = "rgb", file_ext: str = "png") -> (str, str):
    data_ext = {
        "jpg": ".jpg",
        "png": ".png"
        # add more data types here
    }
    f_tokens = f.split('/')
    if any("thud" in s for s in f_tokens):  # THUD dataset
        timepadded = '%06d'%time
        if (data_type == 'rgb'):
            outdatapath   = os.path.join(f, f"RGB/frame-{str(timepadded)}.color{data_ext[file_ext]}")
        else:
            outdatapath = os.path.join(f, f"RGB/frame-{str(timepadded)}.color{data_ext[file_ext]}")
    elif any("isaacsim" in s for s in f_tokens): # isaac_sim dataset
        timepadded = '%05d' % time
        if (data_type == "rgb"):
            outdatapath = os.path.join(f, f"rgb/rgb_{str(timepadded)}{data_ext[file_ext]}")
        elif (data_type == "depth"):
            outdatapath = os.path.join(f, f"depth/depth_{str(timepadded)}{data_ext[file_ext]}")
    elif any( ("navdata_collector" or "former_all" in s) for s in f_tokens): # former dataset
        timepadded = '%05d' % time
        if (data_type == "rgb"):
            outdatapath = os.path.join(f, f"rgb{str(timepadded)}{data_ext[file_ext]}")
        elif (data_type == "depth"):
            outdatapath = os.path.join(f, f"depth{str(timepadded)}{data_ext[file_ext]}")
        else:
            raise Exception("Unknown data type RGB or Depth ? ")
    else:
        raise Exception("Unknown dataset \n")
    return outdatapath

def get_colldata_path(f: str, time: int, data_type: str = "rgb", file_ext: str = "png") -> (str, str):
    data_ext = {
        "jpg": ".jpg",
        "png": ".png"
        # add more data types here
    }
    f_tokens = f.split('/')
    if any( ("colldata" or "colldata_all" in s) for s in f_tokens): # colldataset
        if time < 0:  # read goal
            if (data_type == "rgb"):
                outdatapath = os.path.join(f, f"rgb_sg{data_ext[file_ext]}")
            elif (data_type == "depth"):
                outdatapath = os.path.join(f, f"depth_sg{data_ext[file_ext]}")
            else:
                raise Exception("Unknown data type RGB or Depth ? ")
        else:
            timepadded = '%05d' % time
            if (data_type == "rgb"):
                outdatapath = os.path.join(f, f"rgb{str(timepadded)}{data_ext[file_ext]}")
            elif (data_type == "depth"):
                outdatapath = os.path.join(f, f"depth{str(timepadded)}{data_ext[file_ext]}")
            else:
                raise Exception("Unknown data type RGB or Depth ? ")
    else:
        raise Exception("Unknown dataset \n")
    return outdatapath

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = torch.from_numpy(np.asarray(viz_img, dtype=np.float32))
    #viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    # resize_img = TF.to_tensor(img)  #<--- cannot run this code line b/c the reason below
    # TF.to_tensor works for uint8 images !! I had to convert PIL to np array 1st, followed by converting to tensor
    if (img.mode == 'RGBA'):    # we don't need rgba img
        img = img.convert('RGB')
    elif (img.mode == 'RGB'):
        img = img
    elif (img.mode == 'I;16'):  # depth img case
        img = Image.fromarray( np.array(img).astype('float') / MAX_DEPTH)
    else:
        print("Uknown input image type to process.. \n")
        raise NotImplementedError

    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)


def _interp_pose(xyth0, xyth_e, t):
    (x0, y0, th_0) = xyth0
    (xe, ye, th_e) = xyth_e
    htm_0 = rm.xyzrpy_to_htm( [x0, y0, 0, 0, 0, th_0] )
    htm_e = rm.xyzrpy_to_htm( [xe, ye, 0, 0, 0, th_e] )
    q0 = rm.htm_to_quat( htm_0 )
    qe = rm.htm_to_quat( htm_e )
    # print("q0", q0)
    # print("qe", qe)
    qp = rm.slerp( q0, qe, t )
    htm_p = rm.quat_to_htm( qp )
    (_, _, _, _, _, yaw_p) = rm.htm_to_xyzrpy( htm_p )
    # interpolate xy coord
    yp = y0 + (ye - y0) * t
    xp = x0 + (xe - x0) * t
    return [xp, yp, yaw_p]

# def _normalize_waypoints(pose, waypoint_spacing: int, dataset_name = 'former'):
#     if pose.shape[-1] != 4:
#         raise AssertionError("pose last dim must be 4: [x, y, qw, qz]")
#
#     gamma = 1.2
#     (T, ndim) = pose.shape
#
#     idx = np.arange(1, T + 1, dtype=pose.dtype).reshape(T, 1)   # (T,1)
#     meters_per_wp = []
#     if dataset_name == 'isaac_sim':
#         isaac_config = data_config['isaac_sim']
#         meters_per_wp = isaac_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = isaac_config['gamma']
#     elif dataset_name == 'former':
#         former_config = data_config['former']
#         meters_per_wp = former_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = former_config['gamma']
#
#     dmax_xy = gamma * idx * meters_per_wp  # (T,1)
#
#     pose_norm = np.array(pose, copy=True)
#     pose_norm[:, 0:2] = pose_norm[:, 0:2] / dmax_xy
#     return pose_norm
#
# def _denormalize_waypoints(pose_norm: np.ndarray, waypoint_spacing: int, dataset_name = 'former'):
#     out = np.array(pose_norm, copy=True)
#     meters_per_wp = []
#     gamma = 1.2
#     if dataset_name == 'isaac_sim':
#         isaac_config = data_config['isaac_sim']
#         meters_per_wp = isaac_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = isaac_config['gamma']
#     elif dataset_name == 'former':
#         former_config = data_config['former']
#         meters_per_wp = former_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = former_config['gamma']
#
#     (B, len_traj, dim) = pose_norm.shape
#     assert dim == 4
#     idx_ = np.arange(1, len_traj + 1, dtype=np.float32).reshape(1, len_traj, 1)
#     idx = np.broadcast_to(idx_, (B, len_traj, 2))
#
#     dmax_xy = gamma * idx * meters_per_wp  # (T,1)
#     if out.ndim == 1:      # (4,)
#         out[0:2] = out[0:2] * float(dmax_xy)
#     else:                  # (T,4) or (B,T,4)
#         out[..., 0:2] = out[..., 0:2] * dmax_xy
#     return out
#
#
# def _normalize_context_poses(pose, waypoint_spacing: int, dataset_name = 'former'):
#     """
#     normalize context_poses
#     Args:
#         In pose, P0 corresponds to the farthest, P4 corresponds to the right before the curr pose
#         That is, the order or pose is, P0-5, P0-4, P0-3, P0-2, P0-1
#     Returns:
#         torch.Tensor: normalized pose
#     """
#     if pose.shape[-1] != 4:
#         raise AssertionError("pose last dim must be 4: [x, y, qw, qz]")
#
#     gamma = 1.2
#     T = pose.shape[0]
#     idx = np.arange(T, 0, -1, dtype=float).reshape(T, 1)
#
#     meters_per_wp = []
#     if dataset_name == 'isaac_sim':
#         isaac_config = data_config['isaac_sim']
#         meters_per_wp = isaac_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = isaac_config['gamma']
#     elif dataset_name == 'former':
#         former_config = data_config['former']
#         meters_per_wp = former_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = former_config['gamma']
#
#     dmax_xy = gamma * idx * meters_per_wp  # (T,1)
#
#     pose_norm = np.array(pose, copy=True)
#     pose_norm[:, 0:2] = pose_norm[:, 0:2] / dmax_xy
#     return pose_norm
#
# def _denormalize_context_poses(pose_norm: np.ndarray, waypoint_spacing: int, dataset_name = 'former'):
#     """
#     denormalize context_poses
#     Args:
#         In pose_norm, P0 corresponds to the farthest, P4 corresponds to the right before the curr pose
#     Returns:
#         torch.Tensor: denormalized pose
#     """
#
#     out = np.array(pose_norm, copy=True)
#     meters_per_wp = []
#     gamma = 1.2
#     if dataset_name == 'isaac_sim':
#         isaac_config = data_config['isaac_sim']
#         meters_per_wp = isaac_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = isaac_config['gamma']
#     elif dataset_name == 'former':
#         former_config = data_config['former']
#         meters_per_wp = former_config['metric_frame_spacing'] * waypoint_spacing
#         gamma = former_config['gamma']
#
#     (B, len_traj, dim) = pose_norm.shape
#     idx_ = np.arange(len_traj, 0, -1, dtype=np.float32).reshape(1, len_traj, 1)
#     idx = np.broadcast_to(idx_, (B, len_traj, 2))
#
#     dmax_xy = gamma * idx * meters_per_wp  # (T,1)
#     if out.ndim == 1:      # (4,)
#         out[0:2] = out[0:2] * float(dmax_xy)
#     else:                  # (T,4) or (B,T,4)
#         out[..., 0:2] = out[..., 0:2] * dmax_xy
#     return out


def _normalize_subgoal(sg_pose, max_frame_dist: int= 24, dataset_name = 'former'):
    # if sg_pose.shape[-1] != 4:
    #     raise AssertionError("subgoal last dim must be 4: [x, y, qw, qz]")

    scale = None
    gamma = 1.2
    if dataset_name == 'isaac_sim':
        isaac_config = data_config['isaac_sim']
        scale = isaac_config['metric_frame_spacing'] * max_frame_dist
        gamma = isaac_config['gamma']
    elif dataset_name == 'former':
        former_config = data_config['former']
        scale = former_config['metric_frame_spacing'] * max_frame_dist
        gamma = former_config['gamma']

    goal_scale = float(gamma * scale)
    sg_pose_norm = sg_pose.copy()
    sg_pose_norm[..., :2] = sg_pose_norm[..., :2] / goal_scale
    return sg_pose_norm

def _denormalize_subgoal(sg_pose_norm, max_frame_dist: int = 24, dataset_name = 'former'):
    out = np.array(sg_pose_norm, copy=True)
    scale = None
    gamma = 1.2
    if dataset_name == 'isaac_sim':
        isaac_config = data_config['isaac_sim']
        scale = isaac_config['metric_frame_spacing'] * max_frame_dist
        gamma = isaac_config['gamma']
    elif dataset_name == 'former':
        former_config = data_config['former']
        scale = former_config['metric_frame_spacing'] * max_frame_dist
        gamma = former_config['gamma']

    goal_scale = float(gamma * scale)
    out[..., 0:2] = out[..., 0:2] * float(goal_scale)
    return out

def _normalize_pose( pose, waypoint_spacing: int, dataset_name = 'isaac_sim'  ):

    out = np.array(pose, copy=True)
    meters_per_wp = []
    gamma = 1.2
    if dataset_name == 'isaac_sim':
        isaac_config = data_config['isaac_sim']
        meters_per_wp = isaac_config['metric_frame_spacing'] * waypoint_spacing
        gamma = isaac_config['gamma']
    elif dataset_name == 'former':
        former_config = data_config['former']
        meters_per_wp = former_config['metric_frame_spacing'] * waypoint_spacing
        gamma = former_config['gamma']

    dmax_xy = gamma * meters_per_wp  # (T,1)
    if out.ndim == 1:      # (4,)
        out[0:2] = out[0:2] / float(dmax_xy)
    else:                  # (T,4) or (B,T,4)
        out[..., 0:2] = out[..., 0:2] / dmax_xy
    return out

    return pose_norm
#
def _denormalize_pose( pose_norm, waypoint_spacing: int, dataset_name = 'isaac_sim'):

    out = pose_norm.copy() #np.array(pose_norm, copy=True)
    meters_per_wp = []
    gamma = 1.2
    if dataset_name == 'isaac_sim':
        isaac_config = data_config['isaac_sim']
        meters_per_wp = isaac_config['metric_frame_spacing'] * waypoint_spacing
        gamma = isaac_config['gamma']
    elif dataset_name == 'former':
        former_config = data_config['former']
        meters_per_wp = former_config['metric_frame_spacing'] * waypoint_spacing
        gamma = former_config['gamma']

    dmax_xy = gamma * meters_per_wp  # (T,1)

    if out.ndim == 1:
        out[0:2] *= float(dmax_xy)
    else:
        out[..., 0:2] *= float(dmax_xy)
    return out


# def _get_rel_pose_se2(xyt0, xytp, xyte, data_name, bHc: np.array = np.eye(4) ):
#
#     x0, y0, th0 = xyt0
#     xp, yp, th_p = xytp
#     xe, ye, th_e = xyte
#     cHb = np.linalg.inv(bHc)
#     num_wpts = len(xp)
#     rel_xp = np.zeros([num_wpts])
#     rel_yp = np.zeros([num_wpts])
#     rel_theta_p = np.zeros([num_wpts])
#     if data_name == 'thud':
#         raise NotImplementedError
#
#         wHc0_left = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
#         wHc0 = rm.Left2Right(wHc0_left)
#         c0Hw = np.linalg.inv(wHc0)
#         b0Hw = bHc * c0Hw
#
#         wHcp_left = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
#         wHcp = rm.Left2Right(wHcp_left)
#         wHbp = wHcp * cHb
#
#         wHce_left = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
#         wHce = rm.Left2Right(wHce_left)
#         wHbe = wHce * cHb
#
#         # prediction w.r.t b0
#         b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
#         xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
#
#         rel_xp = xb_p
#         rel_yp = yb_p  # base motion is on xy plane
#         rel_theta_p = rm.normalizeAngle(yaw_p) # -pi ~ pi
#         # TODO: need to check if using only qs qz is sufficient.
#
#         # goal_pos w.r.t b0
#         b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
#         xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
#         quat_g = rm.htm_to_quat(b0Hbe)
#         assert (np.linalg.norm(quat_g) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
#         rel_xg = xb_g
#         rel_yg = yb_g  # base_link motion is on xy plane !!
#         rel_theta_g = rm.normalizeAngle(yaw_g) # -pi ~ pi
#
#     elif data_name == 'isaac_sim' or data_name == 'former':
#         wHb0 = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
#         b0Hw = np.linalg.inv(wHb0)
#         wHbe = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
#
#         for idx in range(0, num_wpts):
#             wHbp = rm.xyzrpy_to_htm(np.array([xp[idx], yp[idx], 0, 0, 0, th_p[idx]]))
#             b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
#             xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
#             rel_xp[idx] = xb_p
#             rel_yp[idx] = yb_p
#             # prediction w.r.t c0
#             rel_theta_p[idx] = rm.normalizeAngle(yaw_p) # enforce -pi ~ pi
#
#         # goal_pos w.r.t b0
#         b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
#         xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
#         rel_xg = xb_g
#         rel_yg = yb_g  # base_link motion is on xy plane !!
#         rel_theta_g = rm.normalizeAngle(yaw_g) # -pi ~ pi
#
#     return rel_xp, rel_yp, rel_theta_p, rel_xg, rel_yg, rel_theta_g


def _get_rel_pose_se2(xyt0: np.ndarray,
                      xytp: np.ndarray,  #  3 x N
                      data_name,
                      bHc: np.array = np.eye(4) ):
##################################3
#xyt0 : base (origin) coord
# xytp : target coord
    if ( len(xytp.shape) == 1 ):
        xytp = xytp[..., None]  # (3, context-1) or (3, 1)

    x0, y0, th0 = xyt0
    cHb = np.linalg.inv(bHc)

    (_, num_pts) = xytp.shape
    rel_xp = np.zeros([num_pts])
    rel_yp = np.zeros([num_pts])
    rel_theta_p = np.zeros([num_pts])
    if data_name == 'thud':
        raise NotImplementedError

        # wHc0_left = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        # wHc0 = rm.Left2Right(wHc0_left)
        # c0Hw = np.linalg.inv(wHc0)
        # b0Hw = bHc * c0Hw
        #
        # wHcp_left = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
        # wHcp = rm.Left2Right(wHcp_left)
        # wHbp = wHcp * cHb
        #
        # wHce_left = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
        # wHce = rm.Left2Right(wHce_left)
        # wHbe = wHce * cHb
        #
        # # prediction w.r.t b0
        # b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
        # xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
        #
        # rel_xp = xb_p
        # rel_yp = yb_p  # base motion is on xy plane
        # rel_theta_p = rm.normalizeAngle(yaw_p) # -pi ~ pi
        # # TODO: need to check if using only qs qz is sufficient.
        #
        # # goal_pos w.r.t b0
        # b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
        # xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
        # quat_g = rm.htm_to_quat(b0Hbe)
        # assert (np.linalg.norm(quat_g) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        # rel_xg = xb_g
        # rel_yg = yb_g  # base_link motion is on xy plane !!
        # rel_theta_g = rm.normalizeAngle(yaw_g) # -pi ~ pi

    elif data_name == 'isaac_sim' or data_name == 'former':
        wHb0 = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        b0Hw = np.linalg.inv(wHb0)

        for idx in range(0, num_pts):
            wHbp = rm.xyzrpy_to_htm(np.array([xytp[0,idx], xytp[1,idx], 0, 0, 0, xytp[2,idx]]))
            b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
            xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
            rel_xp[idx] = xb_p
            rel_yp[idx] = yb_p
            # prediction w.r.t c0
            rel_theta_p[idx] = rm.normalizeAngle(yaw_p) # enforce -pi ~ pi
    if num_pts == 1:
        return rel_xp.item(), rel_yp.item(), rel_theta_p.item()
    else:
        return rel_xp, rel_yp, rel_theta_p

def _get_rel_robot_poses(xyt0, xytp, xyte, data_name, bHc: np.array = np.eye(4) ):

    x0, y0, th0 = xyt0
    xp, yp, th_p = xytp
    xe, ye, th_e = xyte
    cHb = np.linalg.inv(bHc)

    if data_name == 'thud':
        wHc0_left = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        wHc0 = rm.Left2Right(wHc0_left)
        c0Hw = np.linalg.inv(wHc0)
        b0Hw = bHc * c0Hw

        wHcp_left = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
        wHcp = rm.Left2Right(wHcp_left)
        wHbp = wHcp * cHb

        wHce_left = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
        wHce = rm.Left2Right(wHce_left)
        wHbe = wHce * cHb

        # prediction w.r.t b0
        b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
        xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
        quat_p = rm.htm_to_quat(b0Hbp)
        assert (np.linalg.norm(quat_p) == 1), (f"action quat norm is {np.linalg.norm(quat_p)} while it should be 1")
        rel_xp = xb_p
        rel_yp = yb_p  # base motion is on xy plane
        rel_pred_qs = quat_p[0].copy()  # qs
        rel_pred_qz = quat_p[2].copy()  # qz yaw of the robot
        # TODO: need to check if using only qs qz is sufficient.

        # goal_pos w.r.t b0
        b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
        xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
        quat_g = rm.htm_to_quat(b0Hbe)
        assert (np.linalg.norm(quat_g) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        rel_xg = xb_g
        rel_yg = yb_g  # base_link motion is on xy plane !!
        rel_goal_qs = quat_g[0].copy()
        rel_goal_qz = quat_g[3].copy()

    elif data_name == 'isaac_sim' or data_name == 'former':
        wHb0 = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        b0Hw = np.linalg.inv(wHb0)

        wHbp = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
        wHbe = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))

        # prediction w.r.t c0
        b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
        xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
        quat_p = rm.htm_to_quat(b0Hbp)
        assert ( round(np.linalg.norm(quat_p), 4) == 1), (f"action quat norm is {np.linalg.norm(quat_p)} while it should be 1")
        rel_xp = xb_p
        rel_yp = yb_p  # base_link motion is on xy plane
        rel_pred_qs = quat_p[0].copy()  # qs (qw)
        rel_pred_qz = quat_p[3].copy()  # qy yaw of robot corresponds to  the pitch motion of the cam

        # goal_pos w.r.t b0
        b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
        xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
        quat_g = rm.htm_to_quat(b0Hbe)
        assert ( round(np.linalg.norm(quat_g), 4) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        rel_xg = xb_g
        rel_yg = yb_g  # base_link motion is on xy plane !!
        rel_goal_qs = quat_g[0].copy()
        rel_goal_qz = quat_g[3].copy()

    return rel_xp, rel_yp, rel_pred_qs, rel_pred_qz, rel_xg, rel_yg, rel_goal_qs, rel_goal_qz


def _get_rel_robot_poses(xyt0, xytp, xyte, data_name, bHc: np.array = np.eye(4) ):

    x0, y0, th0 = xyt0
    xp, yp, th_p = xytp
    xe, ye, th_e = xyte
    cHb = np.linalg.inv(bHc)

    if data_name == 'thud':
        wHc0_left = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        wHc0 = rm.Left2Right(wHc0_left)
        c0Hw = np.linalg.inv(wHc0)
        b0Hw = bHc * c0Hw

        wHcp_left = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
        wHcp = rm.Left2Right(wHcp_left)
        wHbp = wHcp * cHb

        wHce_left = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
        wHce = rm.Left2Right(wHce_left)
        wHbe = wHce * cHb

        # prediction w.r.t b0
        b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
        xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
        quat_p = rm.htm_to_quat(b0Hbp)
        assert (np.linalg.norm(quat_p) == 1), (f"action quat norm is {np.linalg.norm(quat_p)} while it should be 1")
        rel_xp = xb_p
        rel_yp = yb_p  # base motion is on xy plane
        rel_pred_qs = quat_p[0].copy()  # qs
        rel_pred_qz = quat_p[2].copy()  # qz yaw of the robot
        # TODO: need to check if using only qs qz is sufficient.

        # goal_pos w.r.t b0
        b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
        xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
        quat_g = rm.htm_to_quat(b0Hbe)
        assert (np.linalg.norm(quat_g) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        rel_xg = xb_g
        rel_yg = yb_g  # base_link motion is on xy plane !!
        rel_goal_qs = quat_g[0].copy()
        rel_goal_qz = quat_g[3].copy()

    elif data_name == 'isaac_sim' or data_name == 'former':
        wHb0 = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        b0Hw = np.linalg.inv(wHb0)

        wHbp = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
        wHbe = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))

        # prediction w.r.t c0
        b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
        xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
        quat_p = rm.htm_to_quat(b0Hbp)
        assert ( round(np.linalg.norm(quat_p), 4) == 1), (f"action quat norm is {np.linalg.norm(quat_p)} while it should be 1")
        rel_xp = xb_p
        rel_yp = yb_p  # base_link motion is on xy plane
        rel_pred_qs = quat_p[0].copy()  # qs (qw)
        rel_pred_qz = quat_p[3].copy()  # qy yaw of robot corresponds to  the pitch motion of the cam

        # goal_pos w.r.t b0
        b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
        xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
        quat_g = rm.htm_to_quat(b0Hbe)
        assert ( round(np.linalg.norm(quat_g), 4) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        rel_xg = xb_g
        rel_yg = yb_g  # base_link motion is on xy plane !!
        rel_goal_qs = quat_g[0].copy()
        rel_goal_qz = quat_g[3].copy()

    return rel_xp, rel_yp, rel_pred_qs, rel_pred_qz, rel_xg, rel_yg, rel_goal_qs, rel_goal_qz

def _get_rel_cam_poses(self, xyt0, xytp, xyte, data_name, rHc: np.array = np.eye(4) ):
    x0, y0, th0 = xyt0
    xp, yp, th_p = xytp
    xe, ye, th_e = xyte

    if data_name == "thud":
        wHc0_left = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        wHc0 = rm.Left2Right(wHc0_left)
        c0Hw = np.linalg.inv(wHc0)

        wHcp_left = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
        wHcp = rm.Left2Right(wHcp_left)

        wHce_left = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
        wHce = rm.Left2Right(wHce_left)

        # prediction w.r.t c0
        c0Hcp = np.matmul(c0Hw, wHcp)  # relative pred pose htm
        xc_p, yc_p, zc_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(c0Hcp)  #
        quat_p = rm.htm_to_quat(c0Hcp)
        assert (np.linalg.norm(quat_p) == 1), (f"action quat norm is {np.linalg.norm(quat_p)} while it should be 1")
        rel_xp = xc_p
        rel_zp = zc_p  # cam motion is on xz plane
        rel_pred_qs = quat_p[0].copy()  # qs
        rel_pred_qy = quat_p[2].copy()  # qy yaw of robot corresponds to  the pitch motion of the cam

        # goal_pos w.r.t c0
        c0Hce = np.matmul(c0Hw, wHce)  # relative goal pose htm
        xc_g, yc_g, zc_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(c0Hce)  #
        quat_g = rm.htm_to_quat(c0Hce)
        assert (np.linalg.norm(quat_g) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        rel_xg = xc_g
        rel_zg = zc_g  # cam motion is on xz plane !!
        rel_goal_qs = quat_g[0].copy()
        rel_goal_qy = quat_g[2].copy()

    elif data_name == "former":
        wHr0 = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        wHc0 = np.matmul( wHr0, rHc )
        c0Hw = np.linalg.inv(wHc0)

        wHrp = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
        wHcp = np.matmul( wHrp, rHc )

        wHre = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
        wHce = np.matmul( wHre, rHc)

        # prediction w.r.t c0
        c0Hcp = np.matmul(c0Hw, wHcp)  # relative pred pose htm
        xc_p, yc_p, zc_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(c0Hcp)  #
        quat_p = rm.htm_to_quat(c0Hcp)
        assert ( round(np.linalg.norm(quat_p), 4) == 1), (f"action quat norm is {np.linalg.norm(quat_p)} while it should be 1")
        rel_xp = xc_p
        rel_zp = zc_p  # cam motion is on xz plane
        rel_pred_qs = quat_p[0].copy()  # qs (qw)
        rel_pred_qy = quat_p[2].copy()  # qy yaw of robot corresponds to  the pitch motion of the cam

        # goal_pos w.r.t c0
        c0Hce = np.matmul(c0Hw, wHce)  # relative goal pose htm
        xc_g, yc_g, zc_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(c0Hce)  #
        quat_g = rm.htm_to_quat(c0Hce)
        assert ( round(np.linalg.norm(quat_g), 4) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        rel_xg = xc_g
        rel_zg = zc_g  # cam motion is on xz plane !!
        rel_goal_qs = quat_g[0].copy()
        rel_goal_qy = quat_g[2].copy()

    return rel_xp, rel_zp, rel_pred_qs, rel_pred_qy, rel_xg, rel_zg, rel_goal_qs, rel_goal_qy


# wHc0_left = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
# wHc0 = rm.Left2Right(wHc0_left)
# c0Hw = np.linalg.inv(wHc0)
#
# wHcp_left = rm.xyzrpy_to_htm(np.array([xp, yp, 0, 0, 0, th_p]))
# wHcp = rm.Left2Right(wHcp_left)
#
# wHce_left = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))
# wHce = rm.Left2Right(wHce_left)
#
# # prediction w.r.t c0
# c0Hcp = np.matmul(c0Hw, wHcp)  # relative pred pose htm
# xc_p, yc_p, zc_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(c0Hcp)  #
# quat_p = rm.htm_to_quat(c0Hcp)
# assert (np.linalg.norm(quat_p) == 1), (f"action quat norm is {np.linalg.norm(quat_p)} while it should be 1")
# rel_xp = xc_p
# rel_zp = zc_p  # cam motion is on xz plane
# rel_pred_qs = quat_p[0].copy()  # qs
# rel_pred_qy = quat_p[2].copy()  # qy yaw of robot corresponds to  the pitch motion of the cam
#
# # goal_pos w.r.t c0
# c0Hce = np.matmul(c0Hw, wHce)  # relative goal pose htm
# xc_g, yc_g, zc_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(c0Hce)  #
# quat_g = rm.htm_to_quat(c0Hce)
# assert (np.linalg.norm(quat_g) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
# rel_xg = xc_g
# rel_zg = zc_g  # cam motion is on xz plane !!
# rel_goal_qs = quat_g[0].copy()
# rel_goal_qy = quat_g[2].copy()
