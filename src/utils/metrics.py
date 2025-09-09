import json
import os
from functools import partial
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from einops import rearrange
# from src.utils.metrics import compute_prc, compute_roc, dice, fpr, tpr  # noqa: E402


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
    """
    Remove outlier intensities from a brain component,
    similar to Tukey's fences method.
    """
    upperbound = np.percentile(arr, uperc)
    lowerbound = np.percentile(arr, lperc)
    arr[arr > upperbound] = upperbound
    arr[arr < lowerbound] = lowerbound
    return arr


def error_filter(arr, value=None):
    """
    Filter out error based on a value. Value is default to 95th percentile of arr.
    """
    if value is None:
        value = np.percentile(arr, 95)
    mask = arr < value
    new_arr = np.where(mask, 0, arr)
    return new_arr


def mse(x, x_recons):
    """
    Calculate the MSE errors between original images and reconstruction
    """
    se = (x - x_recons) ** 2  # [150, 10, 64, 64]
    mse_per_img = se.mean(dim=(2, 3))  # [150, 10]
    mse_per_img = mse_per_img.view(-1)  # [1500]

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


# def ssim(x, x_recons):
#     """
#     Compute SSIM between x and x_recons
#     Apply ScaleIntensity to [0, 1] to x_recons before calculating.
#     """

#     ssim_metric = SSIMMetric(
#         spatial_dims=2,  # Use 2 for 2D images, 3 for 3D
#         data_range=1.0,  # Max value of input images (e.g. 1.0 if normalized to [0, 1])
#         k1=0.01,
#         k2=0.03,
#         win_size=16,
#     )

#     return ssim_metric(x, x_recons).cpu().numpy()


# def psnr(x, x_recons):
#     """
#     Compute PSNR metric between x_origin and x_recons
#     max_val = 1.0
#     """
#     psnr = PSNRMetric(max_val=1.0, reduction="mean")
#     return psnr(x, x_recons)


def get_eval_dictionary():
    _eval = {
        "test_ddim": "",
        "test_noise_level": "",
        "IDs": [],
        "reconstructionTimes": [],
        "Age": [],
        "AgeGroup": [],
        "labelPerVol": [],
        "anomalyType": [],
    }
    return _eval


def apply_brainmask(x, brainmask, erode=True, iterations=10):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(
            np.squeeze(brainmask), structure=strel, iterations=iterations
        )
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))


def apply_brainmask_volume(vol, mask_vol, erode=True, iterations=10):
    for s in range(vol.squeeze().shape[2]):
        slice = vol.squeeze()[:, :, s]
        mask_slice = mask_vol.squeeze()[:, :, s]
        eroded_vol_slice = apply_brainmask(
            slice, mask_slice, erode=True, iterations=vol.squeeze().shape[1] // 25
        )
        vol.squeeze()[:, :, s] = eroded_vol_slice
    return vol


def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(
        volume, (kernelsize, kernelsize, kernelsize)
    )
    return volume


def apply_2d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    img = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize))
    return img


def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions, pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(
        labels.astype(int), predictions
    )
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds


# def dice(P, G):
#     psum = np.sum(P.flatten())
#     gsum = np.sum(G.flatten())
#     pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
#     score = (2 * pgsum) / (psum + gsum)
#     return score

def dice(pred, truth):
    num = 2 * ((pred * truth).sum(dim=(1, 2, 3)).type(torch.float))
    den = (pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))).type(torch.float)
    return num / den

def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tp)


def test_step_metrics(
    self,
    id,
    data_recon,
    data_orig,
    data_seg,
    data_mask=None,
    label_vol=1,
    anomaly_type=None,
    threshold_pct=0.95,
):
    """
    Calculate metrics for 1 test sample.
    Args:
        id: patient id
        data_recon: reconstruction image.
        data_orig: original image.
        data_seg: segmentation ground truth mask of anomaly.
        data_mask: unused?
        label_vol: patient label. Use binary 0 for healthy and 1 for anomalous.
        anomaly_type (str, optional): type of anomaly, or "healthy"
    """

    # lazy import for safe init
    from src.ldae.utils.anomaly_map import heat_map

    eval_dict = self.eval_dict

    # Some common information
    eval_dict["IDs"].append(id)
    eval_dict["labelPerVol"].append(label_vol)
    eval_dict["anomalyType"].append(anomaly_type)

    diff_volume = data_recon - data_orig

    # l1 and l2 loss
    l1err = nn.functional.l1_loss(data_recon.squeeze(), data_orig.squeeze())
    l2err = nn.functional.mse_loss(data_recon.squeeze(), data_orig.squeeze())
    l1err_anomal = nn.functional.l1_loss(
        data_recon[data_seg > 0], data_orig[data_seg > 0]
    )
    l1err_healthy = nn.functional.l1_loss(
        data_recon[data_seg == 0], data_orig[data_seg == 0]
    )
    l2err_anomal = nn.functional.mse_loss(
        data_recon[data_seg > 0], data_orig[data_seg > 0]
    )
    l2err_healthy = nn.functional.mse_loss(
        data_recon[data_seg == 0], data_orig[data_seg == 0]
    )

    # store in eval dict
    eval_dict["l1recoErrorAll"].append(l1err.item())
    eval_dict["l1recoErrorUnhealthy"].append(l1err_anomal.item())
    eval_dict["l1recoErrorHealthy"].append(l1err_healthy.item())
    eval_dict["l2recoErrorAll"].append(l2err.item())
    eval_dict["l2recoErrorUnhealthy"].append(l2err_anomal.item())
    eval_dict["l2recoErrorHealthy"].append(l2err_healthy.item())

    l1ratio = l1err_anomal.item() / l1err_healthy.item()
    eval_dict["l1ratio"].append(l1ratio)

    # # perceptual loss
    # ssim = self.ssim(data_recon, data_orig)
    # mssim = self.mssim(data_recon, data_orig)
    # lpips = self.lpips(data_recon, data_orig)

    # eval_dict["ssim"].append(ssim.item())
    # eval_dict["mssim"].append(mssim.item())
    # eval_dict["lpips"].append(lpips.item())

    # move data to CPU
    data_seg = data_seg.detach().cpu()
    # binarize the segmentation
    data_seg[data_seg > 0] = 1

    if data_mask is not None:
        data_mask = data_mask.detach().cpu()
        data_mask[data_mask > 0] = 1

    data_orig = data_orig.detach().cpu()
    data_recon = data_recon.detach().cpu()
    diff_volume = diff_volume.detach().cpu()
    threshold_pct = None

    ### Compute Metrics per Volume / Step ###
    if label_vol == 1:
        is_anomaly = True
    else:
        is_anomaly = False

    if is_anomaly:  # only compute metrics if segmentation is available
        # Pixel-Wise Segmentation Error Metrics based on Differenceimage
        P_AUC, _fpr, _tpr, _threshs = compute_roc(
            diff_volume.squeeze().flatten().numpy(),
            data_seg.squeeze().flatten().numpy().astype(bool),
        )
        P_AUPRC, _precisions, _recalls, _threshs = compute_prc(
            diff_volume.squeeze().flatten().numpy(),
            data_seg.squeeze().flatten().numpy().astype(bool),
        )

        # # Heatmap segmentation error metrics based on combination of pixel-distance and perceptual distance
        # ano_map, _, _ = heat_map(data_recon, data_orig, self.fe, v=self.v, fe_layers=self.fe_layers, use_gaussian_blue=True, sigma=3)
        # AUC, _fpr, _tpr, _threshs = compute_roc(ano_map.squeeze().flatten().numpy(),
        #                                     data_seg.squeeze().flatten().numpy().astype(bool))
        # AUPRC, _precisions, _recalls, _threshs = compute_prc(ano_map.squeeze().flatten().numpy(),
        #                                                  data_seg.squeeze().flatten().numpy().astype(bool))

        # thresholding the differnce
        if threshold_pct is not None:
            threshold = torch.quantile(diff_volume, threshold_pct)
        else:  # use the 95% threhold
            threshold = torch.quantile(diff_volume, 0.95)
            # threshold = 0  # not use threshold

        diff_thresholded = diff_volume > threshold

        # Calculate Dice Score with thresholded volumes
        diceScore = dice(
            diff_thresholded.squeeze().numpy(), data_seg.squeeze().flatten().numpy()
        )

        # Other Metrics
        TP, FP, TN, FN = confusion_matrix(
            diff_thresholded.squeeze().flatten().numpy(),
            data_seg.squeeze().flatten().numpy().astype(bool),
            labels=[0, 1],
        ).ravel()
        TPR = tpr(
            diff_thresholded.squeeze().numpy(),
            data_seg.squeeze().flatten().numpy().astype(bool),
        )
        FPR = fpr(
            diff_thresholded.squeeze().numpy(),
            data_seg.squeeze().flatten().numpy().astype(bool),
        )

        # Store result
        eval_dict["lesionSizePerVol"].append(
            np.count_nonzero(data_seg.squeeze().flatten().numpy().astype(bool))
        )
        eval_dict["DiceScorePerVol"].append(diceScore)
        eval_dict["AUCPerVol"].append(P_AUC)
        eval_dict["AUPRCPerVol"].append(P_AUPRC)
        # eval_dict['AUCPerVol'].append(P_AUC)
        # eval_dict['AUPRCPerVol'].append(P_AUPRC)
        eval_dict["TPPerVol"].append(TP)
        eval_dict["FPPerVol"].append(FP)
        eval_dict["TNPerVol"].append(TN)
        eval_dict["FNPerVol"].append(FN)
        eval_dict["TPRPerVol"].append(TPR)
        eval_dict["FPRPerVol"].append(FPR)

        PrecRecF1PerVol = precision_recall_fscore_support(
            data_seg.squeeze().flatten().numpy().astype(bool),
            diff_thresholded.squeeze().flatten().numpy(),
            labels=[0, 1],
        )
        eval_dict["AccuracyPerVol"].append(
            accuracy_score(
                data_seg.squeeze().flatten().numpy().astype(bool),
                diff_thresholded.squeeze().numpy().flatten(),
            )
        )
        eval_dict["PrecisionPerVol"].append(PrecRecF1PerVol[0][1])
        eval_dict["RecallPerVol"].append(PrecRecF1PerVol[1][1])
        eval_dict["SpecificityPerVol"].append(TN / (TN + FP + 0.0000001))


def test_metrics_one_type(
    self, 
    x_orgs, 
    x_recons, 
    x_ano_gts=None,
    eval_dict=None, 
    recon_type="semantic", 
    save_dict=True, 
    save_path=None,
    is_step=False
):
    """
    Calculate metrics
    Args: 
        is_step: if is True, the eval_dict is calculate for 1 step (1 batch of test dataset).
            eval_dict is not save. return the result of x_recons_semantic.
            Currently not use?
    """
    from src.ldae.utils.anomaly_map import heat_map

    if save_path is None:
        save_path = self.result_dict_path

    if eval_dict is None:
        eval_dict = self.eval_dict

    # Sanity check: convert np to torch.Tensor
    to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

    x_orgs = to_torch(x_orgs) if not isinstance(x_orgs, torch.Tensor) else x_orgs
    x_recons = to_torch(x_recons) if not isinstance(x_recons, torch.Tensor) else x_recons
    x_ano_gts = to_torch(x_ano_gts) if not isinstance(x_ano_gts, torch.Tensor) else x_ano_gts

    # pixel distance - l1error
    l1errAll = F.l1_loss(x_orgs, x_recons, reduction="none")

    # pixel distance - l2error
    l2errAll = F.mse_loss(x_orgs, x_recons, reduction="none")

    # pixel distance for anomaly and healthy region
    ano_mask = x_ano_gts > 0
    if torch.any(ano_mask):
        l1errAnomaly = l1errAll[x_ano_gts > 0]
        l1errHealthy = l1errAll[x_ano_gts == 0]
        l2errAnomaly = l2errAll[x_ano_gts > 0]
        l2errHealthy = l2errAll[x_ano_gts == 0]
    else:
        l1errAnomaly = torch.tensor(float("nan"))
        l1errHealthy = torch.tensor(float("nan"))
        l2errAnomaly = torch.tensor(float("nan"))
        l2errHealthy = torch.tensor(float("nan"))

    # feature distance
    B = x_orgs.shape[0]

    f_ds = []
    ano_maps = []
    for i in range(B):
        x0_ano = x_orgs[i]
        x0_hat = x_recons[i]

        ano_map, f_d, i_d = heat_map(
            x0_hat, x0_ano, self.fe, v=self.heatmap_v, fe_layers=["layer1", "layer2", "layer3"]
        )

        f_ds.append(f_d)
        ano_maps.append(ano_map)

    f_ds = torch.stack(f_ds)
    ano_maps = torch.stack(ano_maps)

    if torch.any(ano_mask):
        fdAnomaly = f_ds[x_ano_gts > 0]
        fdHealthy = f_ds[x_ano_gts == 0]
        ano_maps_anomaly = ano_maps[x_ano_gts > 0]
        ano_maps_healthy = ano_maps[x_ano_gts == 0]
    else:
        fdAnomaly = torch.tensor(float("nan"))
        fdHealthy = torch.tensor(float("nan"))
        ano_maps_anomaly = torch.tensor(float("nan"))
        ano_maps_healthy = torch.tensor(float("nan"))

    """
    Similarity metrics
    """
    if x_orgs.dim() == 5:  # B, T, C, H, W
        x0 = rearrange(x_orgs, "b t c h w -> (b t) c h w")
        xhat = rearrange(x_recons, "b t c h w -> (b t) c h w")
    else:
        x0 = x_orgs
        xhat = x_recons

    ssim_metric = self.ssim(x0, xhat)
    mssim_metric = self.mssim(x0, xhat)
    psnr_metric = self.psnr(x0, xhat)

    # LPIPS returns average of all images, so we have to loop through x0 to get the list
    # lpips_metric = self.lpips(x0, xhat)  -> will return a scalar - torch.Tensor([])
    lpips_metric = []
    for i in range(x0.shape[0]):
        lpips_img = self.lpips(x0[[i]], xhat[[i]])
        lpips_metric.append(lpips_img.item())

    if x_orgs.dim() == 5:
        ssim_metric = rearrange(ssim_metric, "(b t) 1 -> b t", b=x_orgs.shape[0])
        mssim_metric = rearrange(mssim_metric, "(b t) 1 -> b t", b=x_orgs.shape[0])
        psnr_metric = rearrange(psnr_metric, "(b t) 1 -> b t", b=x_orgs.shape[0])
        # lpips_metric = rearrange(lpips_metric, "(b t) 1 -> b t", b=x_orgs.shape[0])
    else:
        ssim_metric = rearrange(ssim_metric, "b 1 -> 1 b")
        mssim_metric = rearrange(mssim_metric, "b 1 -> 1 b")
        psnr_metric = rearrange(psnr_metric, "b 1 -> 1 b")
        # lpips_metric = rearrange(lpips_metric, "b 1 -> 1 b")

    # Write to eval_dict
    error_dict = eval_dict.get(f"error_{type}", dict())
    error_dict["SSIMMetrics"] = ssim_metric.cpu().numpy().tolist()
    error_dict["MSSIMMetrics"] = mssim_metric.cpu().numpy().tolist()
    error_dict["PSNRMetrics"] = psnr_metric.cpu().numpy().tolist()
    error_dict["LPIPSMetrics"] = lpips_metric


    """
    Average results and write to eval_dict
    """

    def safe_nanstd(x: torch.Tensor, ddof: int = 1) -> torch.Tensor:
        mask = ~torch.isnan(x)
        valid = mask.sum()

        if valid <= ddof:
            return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
        return torch.std(x[mask], unbiased=(ddof == 1))

    # l1error
    l1errAllMean = torch.nanmean(l1errAll)
    l1errAllStd = safe_nanstd(l1errAll)
    l1errAnomalyMean = torch.nanmean(l1errAnomaly)
    l1errAnomalyStd = safe_nanstd(l1errAnomaly)
    l1errHealthyMean = torch.nanmean(l1errHealthy)
    l1errHealthyStd = safe_nanstd(l1errHealthy)

    l1ratio = l1errAnomalyMean / l1errHealthyMean
    l1ratio

    # l2error
    l2errAllMean = torch.nanmean(l2errAll)
    l2errAllStd = safe_nanstd(l2errAll)
    l2errAnomalyMean = torch.nanmean(l2errAnomaly)
    l2errAnomalyStd = safe_nanstd(l2errAnomaly)
    l2errHealthyMean = torch.nanmean(l2errHealthy)
    l2errHealthyStd = safe_nanstd(l2errHealthy)

    # feature distance
    fdAllMean = torch.nanmean(f_ds)
    fdAllStd = safe_nanstd(f_ds)
    fdAnomalyMean = torch.nanmean(fdAnomaly)
    fdAnomalyStd = safe_nanstd(fdAnomaly)
    fdHealthyMean = torch.nanmean(fdHealthy)
    fdHealthyStd = safe_nanstd(fdHealthy)

    # write to eval_dict
    error_dict = {}
    error_dict["type"] = recon_type
    error_dict["l1errAllMean"] = l1errAllMean
    error_dict["l1errAllStd"] = l1errAllStd
    error_dict["l1errAnomalyMean"] = l1errAnomalyMean
    error_dict["l1errAnomalyStd"] = l1errAnomalyStd
    error_dict["l1errHealthyMean"] = l1errHealthyMean
    error_dict["l1errHealthyStd"] = l1errHealthyStd
    error_dict["l1ratio"] = l1ratio
    error_dict["l2errAllMean"] = l2errAllMean
    error_dict["l2errAllStd"] = l2errAllStd
    error_dict["l2errAnomalyMean"] = l2errAnomalyMean
    error_dict["l2errAnomalyStd"] = l2errAnomalyStd
    error_dict["l2errHealthyMean"] = l2errHealthyMean
    error_dict["l2errHealthyStd"] = l2errHealthyStd

    error_dict["fdAllMean"] = fdAllMean
    error_dict["fdAllStd"] = fdAllStd
    error_dict["fdAnomalyMean"] = fdAnomalyMean
    error_dict["fdAnomalyStd"] = fdAnomalyStd
    error_dict["fdHealthyMean"] = fdHealthyMean
    error_dict["fdHealthyStd"] = fdHealthyStd

    # Similarity metrics
    error_dict["SSIMMetricsMean"] = np.nanmean(ssim_metric.cpu()).item()
    error_dict["SSIMMetricsStd"] = np.nanstd(ssim_metric.cpu()).item()

    error_dict["MSSIMMetricsMean"] = np.nanmean(mssim_metric.cpu()).item()
    error_dict["MSSIMMetricsStd"] = np.nanstd(mssim_metric.cpu()).item()

    error_dict["PSNRMetricsMean"] = np.nanmean(psnr_metric.cpu()).item()
    error_dict["PSNRMetricsStd"] = np.nanstd(psnr_metric.cpu()).item()

    error_dict["LPIPSMetricsMean"] = np.nanmean(lpips_metric).item()
    error_dict["LPIPSMetricsStd"] = np.nanstd(lpips_metric).item()

    # sanitty check
    for k, v in error_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            error_dict[k] = v.item()
        if isinstance(v, np.ndarray) and v.dim() == 0:
            error_dict[k] = v.item()

    eval_dict[f"error_{recon_type}"] = error_dict

    # save eval_dict
    if save_dict:
        with open(save_path, "w") as f:
            json.dump(eval_dict, f)
        print(f"Save eval_dict to {save_path}")

    return eval_dict

def test_metrics(
    self, 
    x_orgs, 
    x_recons, 
    x_ano_gts=None,
    eval_dict=None, 
    recon_type="semantic", 
    save_dict=True, 
    save_path=None,
    is_step=False
):

    if save_path is None:
        save_path = self.result_dict_path

    if eval_dict is None:
        eval_dict = self.eval_dict

    # check the number of anomaly/healthy type
    ano_types = set(eval_dict["anomalyType"])
    if len(ano_types) > 1: 
        for i, ano in enumerate(ano_types):
            # calculate eval_dict for each anomaly_type
            ano_idx = [i for i, x in enumerate(eval_dict["anomalyType"]) if x == ano]
            if len(ano_idx) == 0:
                continue

            eval_dict = test_metrics_one_type(self,
                                              x_orgs=x_orgs[ano_idx],
                                              x_recons=x_recons[ano_idx],
                                              x_ano_gts=x_ano_gts[ano_idx],
                                              eval_dict=eval_dict,
                                              recon_type=f"{recon_type}_{ano}",
                                              save_dict=False,
                                              )
    
    # Calculate overall metrics
    eval_dict = test_metrics_one_type(self, 
                                    x_orgs=x_orgs,
                                    x_recons=x_recons,
                                    x_ano_gts=x_ano_gts,
                                    eval_dict=eval_dict,
                                    recon_type=recon_type,
                                    save_dict=save_dict,
                                    save_path=save_path,
                                    )
    return eval_dict