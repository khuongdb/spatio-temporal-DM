import numpy as np
import json
import torch.nn.functional as F
import torch
import torch.nn as nn
from monai.transforms import ScaleIntensity
from monai.metrics.regression import SSIMMetric, PSNRMetric
import scipy


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

def psnr(x, x_recons):
    """
    Compute PSNR metric between x_origin and x_recons
    max_val = 1.0
    """
    psnr = PSNRMetric(max_val=1.0, 
                      reduction="mean")
    return psnr(x, x_recons)

def get_eval_dictionary():
    _eval = {
        'IDs': [],
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        'Segmentation': [],
        'reconstructionTimes': [],
        'latentSpace': [],
        'Age': [],
        'AgeGroup': [],
        'l1reconstructionErrors': [],
        'l1recoErrorAll': [],
        'l1recoErrorUnhealthy': [],
        'l1recoErrorHealthy': [],
        'l2recoErrorAll': [],
        'l2recoErrorUnhealthy': [],
        'l2recoErrorHealthy': [],
        'l1reconstructionErrorMean': 0.0,
        'l1reconstructionErrorStd': 0.0,
        'l2reconstructionErrors': [],
        'l2reconstructionErrorMean': 0.0,
        'l2reconstructionErrorStd': 0.0,
        'HausPerVol': [],
        'TPPerVol': [],
        'FPPerVol': [],
        'FNPerVol': [],
        'TNPerVol': [],
        'TPRPerVol': [],
        'FPRPerVol': [],
        'TPTotal': [],
        'FPTotal': [],
        'FNTotal': [],
        'TNTotal': [],
        'TPRTotal': [],
        'FPRTotal': [],

        'PrecisionPerVol': [],
        'RecallPerVol': [],
        'PrecisionPerSlice': [],
        'RecallPerSlice': [],
        'lesionSizePerSlice': [],
        'lesionSizePerVol': [],
        'Dice': [],
        'DiceScorePerSlice': [],
        'DiceScorePerVol': [],
        'BestDicePerVol': [],
        'BestThresholdPerVol': [],
        'AUCPerVol': [],
        'AUPRCPerVol': [],
        'SpecificityPerVol': [],
        'AccuracyPerVol': [],
        'TPgradELBO': [],
        'FPgradELBO': [],
        'FNgradELBO': [],
        'TNgradELBO': [],
        'TPRgradELBO': [],
        'FPRgradELBO': [],
        'DicegradELBO': [],
        'DiceScorePerVolgradELBO': [],
        'BestDicePerVolgradELBO': [],
        'BestThresholdPerVolgradELBO': [],
        'AUCPerVolgradELBO': [],
        'AUPRCPerVolgradELBO': [],
        'KLD_to_learned_prior':[],

        'AUCAnomalyCombPerSlice': [], # PerVol!!! + Confusionmatrix.
        'AUPRCAnomalyCombPerSlice': [],
        'AnomalyScoreCombPerSlice': [],


        'AUCAnomalyKLDPerSlice': [],
        'AUPRCAnomalyKLDPerSlice': [],
        'AnomalyScoreKLDPerSlice': [],


        'AUCAnomalyRecoPerSlice': [],
        'AUPRCAnomalyRecoPerSlice': [],
        'AnomalyScoreRecoPerSlice': [],
        'AnomalyScoreRecoBinPerSlice': [],
        'AnomalyScoreAgePerSlice': [],
        'AUCAnomalyAgePerSlice': [],
        'AUPRCAnomalyAgePerSlice': [],

        'labelPerSlice' : [],
        'labelPerVol' : [],
        'AnomalyScoreCombPerVol' : [],
        'AnomalyScoreCombiPerVol' : [],
        'AnomalyScoreCombMeanPerVol' : [],
        'AnomalyScoreRegPerVol' : [],
        'AnomalyScoreRegMeanPerVol' : [],
        'AnomalyScoreRecoPerVol' : [],
        'AnomalyScoreCombPriorPerVol': [],
        'AnomalyScoreCombiPriorPerVol': [],
        'AnomalyScoreAgePerVol' : [],
        'AnomalyScoreRecoMeanPerVol' : [],
        'DiceScoreKLPerVol': [],
        'DiceScoreKLCombPerVol': [],
        'BestDiceKLCombPerVol': [],
        'BestDiceKLPerVol': [],
        'AUCKLCombPerVol': [],
        'AUPRCKLCombPerVol': [],
        'AUCKLPerVol': [],
        'AUPRCKLPerVol': [],
        'TPKLCombPerVol': [],
        'FPKLCombPerVol': [],
        'TNKLCombPerVol': [],
        'FNKLCombPerVol': [],
        'TPRKLCombPerVol': [],
        'FPRKLCombPerVol': [],
        'TPKLPerVol': [],
        'FPKLPerVol': [],
        'TNKLPerVol': [],
        'FNKLPerVol': [],
        'TPRKLPerVol': [],
        'FPRKLPerVol': [],
    }
    return _eval

def _test_step(self, data_recon, data_orig, data_seg, data_mask, batch_idx, ID, label_vol) :
        
        # calculate the residual image
        if self.cfg.get('residualmode','l1'): # l1 or l2 residual
            diff_volume = torch.abs((data_orig-data_recon))
        else:
            diff_volume = (data_orig-data_recon)**2

       # Calculate Reconstruction errors with respect to anomal/normal regions
        l1err = nn.functional.l1_loss(data_recon.squeeze(),data_orig.squeeze())
        l2err = nn.functional.mse_loss(data_recon.squeeze(),data_orig.squeeze())
        l1err_anomal = nn.functional.l1_loss(data_recon.squeeze()[data_seg.squeeze() > 0],data_orig[data_seg > 0]) 
        l1err_healthy = nn.functional.l1_loss(data_recon.squeeze()[data_seg.squeeze() == 0],data_orig[data_seg == 0]) 
        l2err_anomal = nn.functional.mse_loss(data_recon.squeeze()[data_seg.squeeze() > 0],data_orig[data_seg > 0]) 
        l2err_healthy = nn.functional.mse_loss(data_recon.squeeze()[data_seg.squeeze() == 0],data_orig[data_seg == 0])

        # store in eval dict
        self.eval_dict['l1recoErrorAll'].append(l1err.item())
        self.eval_dict['l1recoErrorUnhealthy'].append(l1err_anomal.item())
        self.eval_dict['l1recoErrorHealthy'].append(l1err_healthy.item())
        self.eval_dict['l2recoErrorAll'].append(l2err.item())
        self.eval_dict['l2recoErrorUnhealthy'].append(l2err_anomal.item())
        self.eval_dict['l2recoErrorHealthy'].append(l2err_healthy.item())

        # move data to CPU
        data_seg = data_seg.cpu() 
        data_mask = data_mask.cpu()
        diff_volume = diff_volume.cpu()
        data_orig = data_orig.cpu()
        data_recon = data_recon.cpu()
        # binarize the segmentation
        data_seg[data_seg > 0] = 1
        data_mask[data_mask > 0] = 1

def apply_brainmask(x, brainmask, erode , iterations):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=iterations)
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))

def apply_brainmask_volume(vol,mask_vol,erode=True, iterations=10) : 
    for s in range(vol.squeeze().shape[2]): 
        slice = vol.squeeze()[:,:,s]
        mask_slice = mask_vol.squeeze()[:,:,s]
        eroded_vol_slice = apply_brainmask(slice, mask_slice, erode = True, iterations=vol.squeeze().shape[1]//25)
        vol.squeeze()[:,:,s] = eroded_vol_slice
    return vol

def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume

def apply_2d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    img = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize))
    return img

