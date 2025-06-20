import numpy as np
import json
import torch.nn.functional as F
import torch
from monai.transforms import ScaleIntensity
from monai.metrics.regression import SSIMMetric


def to_serializable(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj




def summarize_errors(errors, save=False, save_path=None):
    """
    Summarize reconstruction errors from diffusion models.
    Args:
        errors: np.ndarray of shape [N, H, W] - [number of samples, pixel_heigh, pixel_weight]

    Returns:
        statistical summary include: min, max, average, (25%, 50%, 75%, 95%) quantile.
    """

    if not isinstance(errors, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if errors.ndim != 3:
        raise ValueError("Input must be a 3D array with shape [N, H, W].")
    
    # assert save and save_path is not None, "save_path is required when save is True "

    flat_errors = errors.flatten()

    summary = dict()
    # Global summary
    global_sum = {
        "min": float(np.min(flat_errors)),
        "max": float(np.max(flat_errors)),
        "mean": float(np.mean(flat_errors)),
        "std": float(np.std(flat_errors)),
        "25%": float(np.percentile(flat_errors, 25)),
        "50%": float(np.percentile(flat_errors, 50)),
        "75%": float(np.percentile(flat_errors, 75)),
        "95%": float(np.percentile(flat_errors, 95)),
    }

    # Image-wise summary (over H×W per image)
    img_wise = {
        "min": np.min(errors, axis=(1, 2)),
        "max": np.max(errors, axis=(1, 2)),
        "mean": np.mean(errors, axis=(1, 2)),
        "std": np.std(errors, axis=(1, 2)),
        "90%": np.percentile(errors, 90, axis=(1, 2)),
        "95%": np.percentile(errors, 95, axis=(1, 2)),
    }

    # Pixel-wise summary (over N samples for each pixel)
    pixel_wise = {
        "min": np.min(errors, axis=0),
        "max": np.max(errors, axis=0),
        "mean": np.mean(errors, axis=0),
        "std": np.std(errors, axis=0),
        "90%": np.percentile(errors, 90, axis=0),
        "95%": np.percentile(errors, 95, axis=0),
    }

    summary["global"] = {k: to_serializable(v) for k, v in global_sum.items()}
    summary["img_wise"] = {k: to_serializable(v) for k, v in img_wise.items()}
    summary["pixel_wise"] = {k: to_serializable(v) for k, v in pixel_wise.items()}

    if save: 
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=4)

    return summary


def percnorm(arr, lperc=1, uperc=99):
    '''
    Remove outlier intensities from a brain component,
    similar to Tukey's fences method.
    '''
    upperbound = np.percentile(arr, uperc)
    lowerbound = np.percentile(arr, lperc)
    arr[arr > upperbound] = upperbound
    arr[arr < lowerbound] = lowerbound
    return arr


def error_filter(arr, value=None):
    '''
    Filter out error based on a value. Value is default to 95th percentile of arr. 
    '''
    if value is None:
        value = np.percentile(arr, 95)
    mask = arr < value
    new_arr = np.where(mask, 0, arr)
    return new_arr


def mse(x, x_recons):
    """
    Calculate the MSE errors between original images and reconstruction
    """
    se = (x - x_recons) ** 2             # [150, 10, 64, 64]
    mse_per_img = se.mean(dim=(2, 3))           # [150, 10]
    mse_per_img = mse_per_img.view(-1)     # [1500]

    mean_mse = mse_per_img.mean()
    sd_mse = mse_per_img.std()
    return mean_mse, sd_mse, mse_per_img


def mae(x, x_recons):
    """
    Calculate the MAE between original images and reconstruction
    """

    mae_per_img = (x - x_recons).abs().mean(dim=(2, 3))  # [B, T]
    mae_per_img = mae_per_img.view(-1)  # [B x T]

    mean_mae = mae_per_img.mean()
    sd_mae = mae_per_img.std()
    return mean_mae, sd_mae, mae_per_img



def ssim(x, x_recons):
    """
    Compute SSIM between x and x_recons
    Apply ScaleIntensity to [0, 1] to x_recons before calculating. 
    """

    ssim_metric = SSIMMetric(
        spatial_dims=2,     # Use 2 for 2D images, 3 for 3D
        data_range=1.0, # Max value of input images (e.g. 1.0 if normalized to [0, 1])
        k1=0.01,
        k2=0.03,
        win_size=16,
    )

    return ssim_metric(x, x_recons).cpu().numpy()
