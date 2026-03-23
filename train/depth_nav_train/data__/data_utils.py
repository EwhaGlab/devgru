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

import sys
sys.path.insert(0,'../..')
import rigid_motion as rm


VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "./data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

# def get_data_path_thud(data_folder: str, f: str, time: int, data_type: str = "image"):
#     data_ext = {
#         "image": ".png",
#         # add more data types here
#     }
#     timepadded = '%06d'%time
#     return os.path.join(data_folder, f, f"frame-{str(timepadded)}.color{data_ext[data_type]}")

# def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
#     data_ext = {
#         "image": ".jpg",
#         # add more data types here
#     }
#
#     return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")

# def get_data_path(data_folder: str, f: str, time: int, data_type: str = "rgb", file_ext: str = "png") -> (str, str):
#     data_ext = {
#         "jpg": ".jpg",
#         "png": ".png"
#         # add more data types here
#     }
#     if( data_folder.split('/')[-2].find('THUD') == 0 ):  # THUD dataset
#         timepadded = '%06d'%time
#         if (data_type == 'rgb'):
#             outdatapath   = os.path.join(data_folder, f, f"RGB/frame-{str(timepadded)}.color{data_ext[file_ext]}")
#         else:
#             outdatapath = os.path.join(data_folder, f, f"RGB/frame-{str(timepadded)}.color{data_ext[file_ext]}")
#     elif (data_folder.split('/')[3].find('isaac') == 0):  # isaac_sim dataset
#     timepadded = '%05d' % time
#     if (data_type == "rgb"):
#         outdatapath = os.path.join(data_folder, f, f"rgb/rgb_{str(timepadded)}{data_ext[file_ext]}")
#     elif (data_type == "depth"):
#         outdatapath = os.path.join(data_folder, f, f"depth/depth_{str(timepadded)}{data_ext[file_ext]}")
#     else:
#         raise Exception("Unknown data type RGB or Depth ? ")
#     elif ( data_folder.split('/')[-3].find('navdata_collector') == 0 ): # Former dataset
#         timepadded = '%05d' % time
#         if (data_type == "rgb"):
#             outdatapath = os.path.join(data_folder, f, f"rgb{str(timepadded)}{data_ext[file_ext]}")
#         elif (data_type == "depth"):
#             outdatapath = os.path.join(data_folder, f, f"depth{str(timepadded)}{data_ext[file_ext]}")
#         else:
#             raise Exception("Unknown data type RGB or Depth ? ")
#     else:
#         raise Exception("Unknown dataset \n")
#         #rgbdatapath = os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")
#         #depthdatapath = os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")
#     return outdatapath

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
        else:
            raise Exception("Unknown data type RGB or Depth ? ")
    elif any("navdata_collector" in s for s in f_tokens): # Former dataset
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

    #resize_img = TF.to_tensor(img)  #<--- cannot run this code b/c the reason below
    # TF.to_tensor works for uint8 images !! I had to convert PIL to np array 1st, followed by converting to tensor
    if (img.mode == 'RGBA'):    # we don't need rgba img
        img = img.convert('RGB')

    Q = np.asarray(img, dtype=np.float32)
    if len(Q.shape) == 3:       # if 3ch rgb image
        Q = Q.transpose(2,0,1)  # HxWxC --> CxHxW
        resize_img = torch.from_numpy(Q)
    else:
        Q = Q[np.newaxis, ...]
    resize_img = torch.from_numpy(Q)
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

# def normalize_goal_pos( ):
#     max_vx_dist = self.data_config['max_vx_pf'] * self.max_frame_dist  # per frame dist * max num frames
#     min_vx_dist = self.data_config['min_vx_pf'] * self.max_frame_dist
#     max_vy_dist = self.data_config['max_vy_pf'] * self.max_frame_dist
#     min_vy_dist = self.data_config['min_vy_pf'] * self.max_frame_dist
#     vx_p_norm = rel_xp / (max_vx_dist / 2)
#     vy_p_norm = rel_xp / (max_vy_dist / 2)
#     theta_p_norm = (theta_p - min_ang_pf * self.max_frame_dist / 2) / (
#                 max_ang_pf * self.max_frame_dist / 2 - min_ang_pf * self.max_frame_dist / 2)
#     vx_g_norm = rel_xg / max_vx_dist
#     vy_g_norm = rel_yg / max_vy_dist
#     theta_g_norm = (theta_g - min_ang_pf * self.max_frame_dist) / (
#                 max_ang_pf * self.max_frame_dist - min_ang_pf * self.max_frame_dist)

def _normalize_pose( pose, max_frame_dist: int, eta = 1.0, dataset_name = 'isaac_sim'  ):
    pose_norm = pose.copy()
    if len(pose_norm.shape) > 1:
        assert(pose_norm.shape[1] == 3) # x y theta
    if dataset_name == 'isaac_sim':
        isaac_config = data_config['isaac_sim']
        max_x_dist = isaac_config['max_vx_pf'] * max_frame_dist  # per frame dist * max num frames
        min_x_dist = isaac_config['min_vx_pf'] * max_frame_dist
        max_y_dist = isaac_config['max_vy_pf'] * max_frame_dist
        min_y_dist = isaac_config['min_vy_pf'] * max_frame_dist
        min_ang_dist = isaac_config['min_vang_pf'] * max_frame_dist
        max_ang_dist = isaac_config['max_vang_pf'] * max_frame_dist

        pose_norm[..., 0] = (pose_norm[..., 0] - min_x_dist * eta) / (max_x_dist * eta - min_x_dist * eta)
        pose_norm[..., 1] = (pose_norm[..., 1] - min_y_dist * eta) / (max_y_dist * eta - min_y_dist * eta)

        if pose_norm.shape[-1] == 3: # we have theta
           pose_norm[..., 2] = (pose_norm[..., 2] - min_ang_dist * eta) / (max_ang_dist * eta - min_ang_dist * eta)
    else:
        raise NotImplementedError
    return pose_norm

def _denormalize_pose( np_pose_norm, max_frame_dist: int, eta = 1.0, dataset_name = 'isaac_sim'):

    np_pose = np_pose_norm.copy()
    if dataset_name == 'isaac_sim':
        isaac_config = data_config[dataset_name]
        max_x_dist = isaac_config['max_vx_pf'] * max_frame_dist  # per frame dist * max num frames
        min_x_dist = isaac_config['min_vx_pf'] * max_frame_dist
        max_y_dist = isaac_config['max_vy_pf'] * max_frame_dist
        min_y_dist = isaac_config['min_vy_pf'] * max_frame_dist

        np_pose[..., 0] = np_pose[..., 0] * (max_x_dist * eta - min_x_dist * eta) + min_x_dist * eta
        np_pose[..., 1] = np_pose[..., 1] * (max_y_dist * eta - min_y_dist * eta) + min_y_dist * eta

        if np_pose.shape[-1] == 3:  # if we have angle there
           max_vang = isaac_config['max_vang_pf'] * max_frame_dist
           min_vang = isaac_config['min_vang_pf'] * max_frame_dist
           np_pose[..., 2] = np_pose[..., 2] * (max_vang * eta - min_vang * eta) + min_vang * eta
    else:
        print("%s is unknown datasetname"%(dataset_name))
        raise NotImplementedError
    return np_pose


def _get_rel_pose_se2(xyt0, xytp, xyte, data_name, bHc: np.array = np.eye(4) ):

    x0, y0, th0 = xyt0
    xp, yp, th_p = xytp
    xe, ye, th_e = xyte
    cHb = np.linalg.inv(bHc)
    num_wpts = len(xp)
    rel_xp = np.zeros([num_wpts])
    rel_yp = np.zeros([num_wpts])
    rel_theta_p = np.zeros([num_wpts])
    if data_name == 'thud':
        raise NotImplementedError

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

        rel_xp = xb_p
        rel_yp = yb_p  # base motion is on xy plane
        rel_theta_p = rm.normalizeAngle(yaw_p) # -pi ~ pi
        # TODO: need to check if using only qs qz is sufficient.

        # goal_pos w.r.t b0
        b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
        xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
        quat_g = rm.htm_to_quat(b0Hbe)
        assert (np.linalg.norm(quat_g) == 1), (f"goal quat norm is {np.linalg.norm(quat_g)} while it should be 1")
        rel_xg = xb_g
        rel_yg = yb_g  # base_link motion is on xy plane !!
        rel_theta_g = rm.normalizeAngle(yaw_g) # -pi ~ pi

    elif data_name == 'isaac_sim' or data_name == 'former':
        wHb0 = rm.xyzrpy_to_htm(np.array([x0, y0, 0, 0, 0, th0]))
        b0Hw = np.linalg.inv(wHb0)
        wHbe = rm.xyzrpy_to_htm(np.array([xe, ye, 0, 0, 0, th_e]))

        for idx in range(0, num_wpts):
            wHbp = rm.xyzrpy_to_htm(np.array([xp[idx], yp[idx], 0, 0, 0, th_p[idx]]))
            b0Hbp = np.matmul(b0Hw, wHbp)  # relative pred pose htm
            xb_p, yb_p, zb_p, rol_p, pit_p, yaw_p = rm.htm_to_xyzrpy(b0Hbp)  #
            rel_xp[idx] = xb_p
            rel_yp[idx] = yb_p
            # prediction w.r.t c0
            rel_theta_p[idx] = rm.normalizeAngle(yaw_p) # enforce -pi ~ pi

        # goal_pos w.r.t b0
        b0Hbe = np.matmul(b0Hw, wHbe)  # relative goal pose htm
        xb_g, yb_g, zb_g, rol_g, pit_g, yaw_g = rm.htm_to_xyzrpy(b0Hbe)  #
        rel_xg = xb_g
        rel_yg = yb_g  # base_link motion is on xy plane !!
        rel_theta_g = rm.normalizeAngle(yaw_g) # -pi ~ pi

    return rel_xp, rel_yp, rel_theta_p, rel_xg, rel_yg, rel_theta_g

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
