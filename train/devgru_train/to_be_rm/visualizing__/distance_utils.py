import os
import wandb
import numpy as np
from typing import List, Optional, Tuple
from depth_nav_train.visualizing.visualize_utils import numpy_to_depth, numpy_to_img
import matplotlib.pyplot as plt
import cv2

def visualize_dist_pred(
    batch_np_obs_rgbs: np.ndarray,
    batch_np_obs_depths: np.ndarray,
    batch_np_goal_rgb: np.ndarray,
    batch_np_goal_depth: np.ndarray,
    batch_np_dist_preds: np.ndarray,
    batch_np_dist_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
    xy_dist_error_threshold: float = 0.033, #hkm  #3.0,
    quat_dist_error_threshold: float = 0.1,
):
    """
    Visualize the distance classification predictions and labels for an observation-goal image pair.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
        batch_dist_preds (np.ndarray): batch of distance predictions [batch_size]
        batch_dist_labels (np.ndarray): batch of distance labels [batch_size]
        eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        save_folder (str): folder to save the images. If None, will not save the images
        display (bool): whether to display the images
        rounding (int): number of decimal places to round the distance predictions and labels
        dist_error_threshold (float): distance error threshold for classifying the distance prediction as correct or incorrect (only used for visualization purposes)
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "dist_classification",
    )
    f"batch depth shape: {batch_np_obs_depths.shape}, batch goal shape: {batch_np_goal_depth.shape}, " \
    f"batch dist pred shape: {batch_np_dist_preds.shape}, batch_dist_label shape {batch_np_dist_labels.shape} "

    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert len(batch_np_obs_depths) == len(batch_np_goal_depth)
    assert len(batch_np_goal_depth) == len(batch_np_dist_preds)
    assert len(batch_np_dist_preds) == len(batch_np_dist_labels)
    batch_size = batch_np_obs_depths.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        np_dist_pred = np.round(batch_np_dist_preds[i], rounding)     #
        np_dist_label = np.round(batch_np_dist_labels[i], rounding)
        viz_obs_rgb    = numpy_to_img(batch_np_obs_rgbs[i])
        viz_obs_depths = numpy_to_depth(batch_np_obs_depths[i])
        viz_goal_rgb   = numpy_to_img(batch_np_goal_rgb[i])
        viz_goal_depth = numpy_to_depth(batch_np_goal_depth[i])

#        print("dist label b4 rounding ", batch_dist_labels[i])
#        print("dist label af rounding ", dist_label)

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")
        text_color = "black"

        if ( np_dist_label.ndim > 0): # xy dist and q dist
            if( abs(np_dist_pred[0] - np_dist_label[0]) > xy_dist_error_threshold or abs(np_dist_pred[1] - np_dist_label[1]) > quat_dist_error_threshold  ):
                text_color = "red"
        else:   # xy only
            if abs(np_dist_pred - np_dist_label) > xy_dist_error_threshold:
                text_color = "red"

        display_distance_pred(
            [viz_obs_rgb, viz_goal_rgb],
            [viz_obs_depths, viz_goal_depth],
            ["Observation", "Goal"],
            np_dist_pred,
            np_dist_label,
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_dist_prediction": wandb_list}, commit=False)


def visualize_dist_pairwise_pred(
    batch_obs_images: np.ndarray,
    batch_close_images: np.ndarray,
    batch_far_images: np.ndarray,
    batch_close_preds: np.ndarray,
    batch_far_preds: np.ndarray,
    batch_close_labels: np.ndarray,
    batch_far_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
):
    """
    Visualize the distance classification predictions and labels for an observation-goal image pair.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_close_images (np.ndarray): batch of close goal images [batch_size, height, width, channels]
        batch_far_images (np.ndarray): batch of far goal images [batch_size, height, width, channels]
        batch_close_preds (np.ndarray): batch of close predictions [batch_size]
        batch_far_preds (np.ndarray): batch of far predictions [batch_size]
        batch_close_labels (np.ndarray): batch of close labels [batch_size]
        batch_far_labels (np.ndarray): batch of far labels [batch_size]
        eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
        rounding (int): number of decimal places to round the distance predictions and labels
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "pairwise_dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert (
        len(batch_obs_images)
        == len(batch_close_images)
        == len(batch_far_images)
        == len(batch_close_preds)
        == len(batch_far_preds)
        == len(batch_close_labels)
        == len(batch_far_labels)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        close_dist_pred = np.round(batch_close_preds[i], rounding)
        far_dist_pred = np.round(batch_far_preds[i], rounding)
        close_dist_label = np.round(batch_close_labels[i], rounding)
        far_dist_label = np.round(batch_far_labels[i], rounding)
        obs_image = numpy_to_depth(batch_obs_images[i])
        close_image = numpy_to_depth(batch_close_images[i])
        far_image = numpy_to_depth(batch_far_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")

        if close_dist_pred < far_dist_pred:
            text_color = "black"
        else:
            text_color = "red"

        display_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_pred = {close_dist_pred}, far_pred = {far_dist_pred}",
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_pairwise_classification": wandb_list}, commit=False)


def display_distance_pred(
    imgs: list,     # PILImage
    depths: list,   # PILImage
    titles: list,
    dist_pred: float,
    dist_label: float,
    text_color: str = "black",
    save_path: Optional[str] = None,
    display: bool = False,
):
    plt.figure()
    fig, ax = plt.subplots(2, len(imgs))

    plt.suptitle(f"prediction: {dist_pred}\nlabel: {dist_label}", color=text_color)

    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    for axis, img, depth, title in zip(ax, imgs, depths, titles):
        #image is class  'PIL.Image.Image'
        # np.savetxt("/home/hankm/Desktop/r.txt", img[..., 0], fmt='%.18e', delimiter=' ', newline='\n')
        # np.savetxt("/home/hankm/Desktop/g.txt", img[..., 1], fmt='%.18e', delimiter=' ', newline='\n')
        # np.savetxt("/home/hankm/Desktop/b.txt", img[..., 2], fmt='%.18e', delimiter=' ', newline='\n')
        #cv2.imwrite("/home/hankm/Desktop/tmp_u8.png", img * 255)
        #assert(np.max(img) <= 1)
        # img[..., 0] = img[..., 0].astype('float') * 0.229 + 0.485
        # img[..., 1] = img[..., 1].astype('float') * 0.224 + 0.456
        # img[..., 2] = img[..., 2].astype('float') * 0.225 + 0.406
        axis[0].imshow(img)
        axis[0].set_title(title)
        axis[0].xaxis.set_visible(False)
        axis[0].yaxis.set_visible(False)
        axis[1].imshow(depth)
        axis[1].set_title(title)
        axis[1].xaxis.set_visible(False)
        axis[1].yaxis.set_visible(False)

    # make the plot large
    fig.set_size_inches((18.5 / 3) * len(imgs), 10.5)

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )
    if not display:
        plt.close(fig)
