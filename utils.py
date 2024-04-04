from typing import Dict
import torch
import numpy as np
import skimage
from sklearn.metrics.cluster import adjusted_rand_score

def run_object_discovery_evaluation(
    batch_size: int, smooth_labels: bool, pred_labels: np.ndarray, gt_labels: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Run object discovery evaluation and calculate metrics.

    Args:
        opt (DictConfig): Configuration options.
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).
        gt_labels (Dict[str, torch.Tensor]): Ground truth labels, each of shape (b, h, w).

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    pred_labels = resize_pred_labels(
        smooth_labels, pred_labels, gt_labels["pixelwise_instance_labels"]
    )
    gt_labels = tensor_dict_to_numpy(gt_labels, dtype=np.int32)

    metrics = {
        "ARI": calc_ari_score(
            batch_size, gt_labels["pixelwise_instance_labels"], pred_labels.copy()
        ),
        "MBO_i": calc_mean_best_overlap(
            batch_size, gt_labels["pixelwise_instance_labels"], pred_labels.copy()
        ),
    }

    if "pixelwise_class_labels" in gt_labels:
        metrics["MBO_c"] = calc_mean_best_overlap(
            batch_size, gt_labels["pixelwise_class_labels"], pred_labels,
        )
    return metrics

def tensor_dict_to_numpy(
    tensor_dict: Dict[str, torch.Tensor], dtype=np.float32
) -> Dict[str, np.ndarray]:
    """
    Convert a dictionary of PyTorch tensors into a dictionary of NumPy arrays.

    Args:
        tensor_dict (Dict[str, torch.Tensor]): A dictionary of PyTorch tensors.
        dtype (Type[np.ndarray], optional): Data type for the resulting NumPy arrays. Default is np.float32.

    Returns:
        Dict[str, np.ndarray]: A dictionary of NumPy arrays.
    """
    for key in tensor_dict:
        tensor_dict[key] = tensor_dict[key].detach().cpu().numpy().astype(dtype)
    return tensor_dict

def resize_pred_labels(
    smooth_labels: bool, pred_labels: np.ndarray, gt_labels: np.ndarray
) -> np.ndarray:
    """
    Resize predicted labels to match the shape of ground truth labels and optionally smooth them.

    Args:
        opt (DictConfig): Configuration options.
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).
        gt_labels (np.ndarray): Ground truth labels, shape (b, h, w).

    Returns:
        np.ndarray: Resized and optionally smoothed predicted labels.
    """
    if pred_labels.shape == gt_labels.shape:
        return pred_labels

    # Resize pred_labels to shape of gt_labels.
    resized_pred_labels = (
        torch.nn.functional.interpolate(
            torch.Tensor(pred_labels)[:, None],
            size=(gt_labels.shape[1], gt_labels.shape[2],),
            mode="nearest",
        )[:, 0]
        .numpy()
        .astype(np.uint8)
    )

    # Smooth out predicted labels by applying mode filter.
    if smooth_labels:
        disk_size = gt_labels.shape[1] // pred_labels.shape[1]
        for i in range(resized_pred_labels.shape[0]):
            resized_pred_labels[i] = skimage.filters.rank.modal(
                skimage.util.img_as_ubyte(resized_pred_labels[i]),
                skimage.morphology.disk(disk_size),
            )

    return resized_pred_labels

def calc_ari_score(
    batch_size: int, gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Calculate Adjusted Rand Index (ARI) score for object discovery evaluation.

    Args:
        opt (DictConfig): Configuration options.
        gt_labels (np.ndarray): Ground truth labels, shape ((b, h, w)).
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).

    Returns:
        float: ARI score.
    """
    ari = 0
    for idx in range(batch_size):
        # Remove "ignore" (-1) and background (0) gt_labels.
        area_to_eval = np.where(gt_labels[idx] > 0)

        ari += adjusted_rand_score(
            gt_labels[idx][area_to_eval], pred_labels[idx][area_to_eval]
        )
    return ari / batch_size


def compute_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) matrix between ground truth and predicted labels.

    Args:
        gt_labels (np.ndarray): Ground truth labels, shape (m, h, w).
        pred_labels (np.ndarray): Predicted labels, shape (o, h, w).

    Returns:
        np.ndarray: IoU matrix, shape (m, o).
    """
    intersection = np.logical_and(
        gt_labels[:, None, :, :], pred_labels[None, :, :, :]
    ).sum(axis=(2, 3))
    union = np.logical_or(gt_labels[:, None, :, :], pred_labels[None, :, :, :]).sum(
        axis=(2, 3)
    )
    return intersection / (union + 1e-9)


def mean_best_overlap_single_sample(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Compute the Mean Best Overlap (MBO) for a single sample between ground truth and predicted labels.

    Args:
        gt_labels (np.ndarray): Ground truth labels, shape (h, w).
        pred_labels (np.ndarray): Predicted labels, shape (h, w).

    Returns:
        float: MBO score for the sample.
    """
    unique_gt_labels = np.unique(gt_labels)
    # Remove "ignore" (-1) label.
    unique_gt_labels = unique_gt_labels[unique_gt_labels != -1]

    # Mask areas with "ignore" gt_labels in pred_labels.
    pred_labels[np.where(gt_labels < 0)] = -1

    # Ignore background (0) gt_labels.
    unique_gt_labels = unique_gt_labels[unique_gt_labels != 0]

    if len(unique_gt_labels) == 0:
        return -1  # If no gt_labels left, skip this element.

    unique_pred_labels = np.unique(pred_labels)

    # Remove "ignore" (-1) label.
    unique_pred_labels = unique_pred_labels[unique_pred_labels != -1]

    gt_masks = np.equal(gt_labels[None, :, :], unique_gt_labels[:, None, None])
    pred_masks = np.equal(pred_labels[None, :, :], unique_pred_labels[:, None, None])

    iou_matrix = compute_iou_matrix(gt_masks, pred_masks)
    best_iou = np.max(iou_matrix, axis=1)
    return np.mean(best_iou)


def calc_mean_best_overlap(
    batch_size: int, gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Calculate the Mean Best Overlap (MBO) for a batch of ground truth and predicted labels.

    Args:
        opt (DictConfig): Configuration options.
        gt_labels (np.ndarray): Ground truth labels, shape (b, h, w).
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).

    Returns:
        float: MBO score for the batch.
    """
    mean_best_overlap = np.array(
        [
            mean_best_overlap_single_sample(gt_labels[b_idx], pred_labels[b_idx])
            for b_idx in range(batch_size)
        ]
    )

    if np.any(mean_best_overlap != -1):
        return np.mean(mean_best_overlap[mean_best_overlap != -1])
    else:
        return 0.0
