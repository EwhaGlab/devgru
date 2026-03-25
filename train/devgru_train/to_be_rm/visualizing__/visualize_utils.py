import numpy as np
from PIL import Image
import torch

VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])
MAXDEPTH = 65535.0  # max value of uint16

def numpy_to_img(arr: np.ndarray) -> Image:
    img_u8 = np.uint8(255 * arr)
    (d1, d2, d3) = img_u8.shape
    if d1 == 3:
        img = Image.fromarray(np.transpose(img_u8, (1, 2, 0)))
    elif d3 == 3:
        img = Image.fromarray(img_u8)
    else:
        raise Exception("weird img shape: %d %d %d"%(d1, d2, d3))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img

def numpy_to_depth(arr: np.ndarray) -> Image:
    # depth is written in millimeters (/1000 gives meters)
    # rescale 0~65535 --> 0~255
    # given 0~5, return 5th img (curr obs)
    depth_map = (arr.astype(np.float32) * 255.0).astype(np.uint8)
    img = Image.fromarray(depth_map[-1].squeeze())
    img = img.resize(VIZ_IMAGE_SIZE)
    return img

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()
