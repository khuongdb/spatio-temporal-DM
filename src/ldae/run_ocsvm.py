import os
import sys

sys.path.append(os.getcwd())

import argparse
import json
import joblib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from lightning.pytorch.utilities.model_summary import ModelSummary
from matplotlib.gridspec import GridSpec
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler

from monai.data import DataLoader
from src.data.datamodules.starmen import StarmenDataset
from src.ldae import CondDDPM, FeatureExtractorLitmodel
from src.ldae.utils import feature_distance, heat_map, patchify
from src.utils.metrics import mae, mse, percnorm
from src.utils.visualization import (
    draw_featmap,
    filter_gt_ano_region,
    plot_comparison_starmen,
    plot_kde_pixel,
)

# Helper function ========================================


to_torch = partial(torch.tensor, dtype=torch.float32, device=torch.device("cpu"))

def load_test_dataset(work_dir, data_dir, split="growing_circle20"):

    infer_dir = os.path.join(work_dir, "infer", f"{split}_ddim100_noise250")

    # Test dataset
    test_ds = StarmenDataset(
    data_dir=data_dir,
    split=split,
    nb_subject=None,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)
    print(f"Len of {split} dataset: {len(test_loader)}")


    # Read the reconstruction error files to get the index and corresponding reonconstruction image
    try: 
        eval_result_file = os.path.join(infer_dir, "results", "eval_dict.json")
        with open(eval_result_file, "r") as f:
            eval_dict = json.load(f)
        test_ids = eval_dict["IDs"]
    # test_ids = np.load(recons_idx_path)
    # test_ids = test_ids.squeeze().tolist()
    except FileNotFoundError:
        test_ids = test_ds.ids

    # Load the original (anomalious) image from test dataset by index
    x_orgs = [test_ds.get_images_by_id(id)["img"].squeeze() for id in test_ids]
    x_orgs = np.stack(x_orgs).astype("float32")

    # Load the original healthy images
    # healthy path example= data/starmen/output_random_noacc/images/SimulatedData__Reconstruction__starman__subject_s0__tp_0.npy
    if split not in ["test", "val", "train"]:
        is_anomaly = True
        x_hts = []
        for id in test_ids:
            for t in range(0, 10):
                ht_path = os.path.join(data_dir, "images", f"SimulatedData__Reconstruction__starman__subject_s{id}__tp_{t}.npy")
                ht = np.load(ht_path)
                x_hts.append(ht)
        x_hts = np.stack(x_hts)
        x_hts = rearrange(x_hts, "(b t) h w -> b t h w", t=10)
        x_hts.shape

        # Load the groundtruth anomaly mask 
        x_ano_gts = []
        for id in test_ds.ids:
            x_ano_gt = test_ds.get_images_by_id(id)["anomaly_gt_seg"]
            if x_ano_gt is None:
                break
            else:
                x_ano_gts.append(x_ano_gt.squeeze())
        if len(x_ano_gts) > 0: 
            x_ano_gts = np.stack(x_ano_gts).astype("float32")
            x_ano_gts.shape
    else:
        is_anomaly = False
        x_hts = None
        x_ano_gts = None

    # add C dimension
    x_orgs = rearrange(x_orgs, "b t h w -> b t 1 h w")

    if is_anomaly:
        x_ano_gts = rearrange(x_ano_gts, "b t h w -> b t 1 h w")
        x_hts = rearrange(x_hts, "b t h w -> b t 1 h w")

    try:
        # Load reconstruction - xT_inferred
        x_recons = np.load(os.path.join(infer_dir, "results", "recons.npy"))
        x_recons.shape

        # Load reconstruction - from random noise
        x_recons_semantic = np.load(os.path.join(infer_dir, "results", "recons_semantic.npy"))
        x_recons_semantic.shape
    except FileNotFoundError: 
        x_recons, x_recons_semantic = None, None

    # convert to torch
    x_orgs = to_torch(x_orgs) if not isinstance(x_orgs, torch.Tensor) else x_orgs
    if is_anomaly:
        x_hts = to_torch(x_hts) if not isinstance(x_hts, torch.Tensor) else x_hts
        x_ano_gts = to_torch(x_ano_gts) if not isinstance(x_ano_gts, torch.Tensor) else x_ano_gts
    if x_recons is not None and x_recons_semantic is not None: 
        x_recons = to_torch(x_recons) if not isinstance(x_recons, torch.Tensor) else x_recons
        x_recons_semantic = to_torch(x_recons_semantic) if not isinstance(x_recons_semantic, torch.Tensor) else x_recons_semantic

    return {
        "x_orgs": x_orgs,
        "x_hts": x_hts,
        "x_ano_gts": x_ano_gts,
        "x_recons": x_recons,
        "x_recons_semantic": x_recons_semantic,
    }

@torch.no_grad()
def train_ocsvm_pixel(fe, device, limit=None):
    # Train dataset - Pixel anomaly score ==============
    out_train = load_test_dataset(data_dir=args.data_path, work_dir=args.workdir, split="train")
    xhats = out_train["x_recons_semantic"]
    x0 = out_train["x_orgs"]

    if limit is not None: 
        xhats = xhats[:limit]
        x0 = x0[:limit]

    fe_device = next(fe.parameters()).device
    print(f"FE is on {fe_device}")

    f_ds = []
    i_ds = []

    for i in range(xhats.shape[0]):
        x = x0[i].to(fe_device)
        xhat = xhats[i].to(fe_device)

        
        _, f_d, i_d = heat_map(
        xhat, x, fe, v=7., fe_layers=["layer1", "layer2"], device=device
        )

        f_ds.append(f_d)
        i_ds.append(i_d)

    f_ds = torch.stack(f_ds)
    i_ds = torch.stack(i_ds)

    # Save f_d and i_d for future analysis
    torch.save(f_ds, os.path.join(args.result_dir, "fd_train.pt"))
    torch.save(i_ds, os.path.join(args.result_dir, "id_train.pt"))


    # concat f_d and i_d at each pixel
    # each pixel will have 2 channels [f_d, i_d], i.e a vector dis \in R^2
    distances = torch.cat((f_ds, i_ds), dim=2)
    distances.shape

    # OneClss-SVM for train dataset =====================
    distances_train = rearrange(distances, "b t c h w -> (b t) c h w").detach().cpu()
    distances_train.shape
    *_, h, w = distances_train.shape

    clf_fits = {}

    for i in range(h):
        for j in range(w): 
            scaler = StandardScaler()
            x_train = scaler.fit_transform(distances_train[:, :, i, j])
            clf_fit_pixel = svm.OneClassSVM(nu=0.03, gamma="scale").fit(x_train)

            key = (i, j)
            clf_fits[key] = {
                "scaler": scaler,
                "clf": clf_fit_pixel
            }
    
    # Save fitted model
    joblib.dump(clf_fits, os.path.join(args.result_dir, "ocsvm-fit-dict.pkl"))
    
    # Plot decision boundary
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(3):
        for j in range(5):

            ax = axes[i, j]

            fit_dict = clf_fits[(i, j)]

            x_train = distances_train[:, :, i, j]
            x_train = fit_dict["scaler"].transform(x_train)

            estimator = fit_dict["clf"]

            df = pd.DataFrame(x_train, columns=["fd", "id"])
            sns.scatterplot(data=df, x="fd", y="id", alpha=0.5, ax=ax, label="x_train")
            ax.set_title(f"pixel position: {(i, j)}")
            DecisionBoundaryDisplay.from_estimator(
                estimator,
                x_train,
                response_method="decision_function",
                plot_method="contour",
                levels=[0],
                colors="red",
                ax=ax,
            )
    plt.suptitle("Example decision function at pixel level")
    plt.tight_layout()
    fig.savefig(f"{args.result_dir}/example-train.pdf", format="pdf", bbox_inches="tight")
    return {
        "fit_dicts": clf_fits, 
    }

@torch.no_grad()
def test_ocsvm_pixel(split, fe, clf_fits, device):
    out_test = load_test_dataset(data_dir=args.data_path, work_dir=args.workdir, split=split)

    x0_test = out_test["x_orgs"].to(device)
    xhats_test = out_test["x_recons_semantic"].to(device)
    print(f"Loaded {split} dataset")

    # Pixel anomaly score
    print("Calculate pixel anomaly score...")
    f_ds_test = []
    i_ds_test = []

    for i in range(xhats_test.shape[0]):
        _, f_d, i_d = heat_map(
        xhats_test[i], x0_test[i], fe, v=7., fe_layers=["layer1", "layer2"], device=device
        )

        f_ds_test.append(f_d)
        i_ds_test.append(i_d)

    f_ds_test = torch.stack(f_ds_test)
    i_ds_test = torch.stack(i_ds_test)

    f_ds_test.shape

    # concat f_d and i_d at each pixel
    # each pixel will have 2 channels [f_d, i_d], i.e a vector dis \in R^2
    distances_test = torch.cat((f_ds_test, i_ds_test), dim=2)
    distances_test.shape
    distances_test = rearrange(distances_test, "b t c h w -> (b t) c h w").detach().cpu()

    # OC-SVM prediction
    print("Run OC-SVM prediction....")

    b, t, c, h, w = xhats_test.shape
    pixel_pred = np.empty((b * t, h, w))
    pixel_dec_score = np.empty((b * t, h, w))

    for i in range(h):
        for j in range(w):
            fit_dict = clf_fits[(i, j)]

            x_test = fit_dict["scaler"].transform(distances_test[:, :, i, j])

            estimator = fit_dict["clf"]
            x_test_pred = estimator.predict(x_test)
            x_test_decision = estimator.decision_function(x_test)

            pixel_pred[:, i, j] = x_test_pred
            pixel_dec_score[:, i, j] = x_test_decision

    pixel_pred = rearrange(pixel_pred, "(b t) h w -> b t h w", t=10)
    pixel_dec_score = rearrange(pixel_dec_score, "(b t) h w -> b t h w", t=10)

    out = {
        "pred": pixel_pred,
        "decision_fnc": pixel_dec_score
    }
    joblib.dump(out, os.path.join(args.result_dir, f"ocsvm-{split}.pkl"))
    return out

def main(args, device):

    result_dir = os.path.join(args.workdir, "run-ocsvm")
    os.makedirs(result_dir, exist_ok=True)
    args.result_dir = result_dir

    # Load models ======================================
    ckpt_path = os.path.join(args.workdir, "fe-train", "checkpoints", "best.ckpt")
    diffae_litmodel = FeatureExtractorLitmodel.load_from_checkpoint(ckpt_path, map_location=device)
    print(f"Load model from {ckpt_path}")

    ema_encoder = diffae_litmodel.encoder
    ema_encoder.eval()
    ema_decoder = diffae_litmodel.unet
    ema_decoder.eval()
    fe = diffae_litmodel.fe
    fe = fe.float()
    fe.eval()

    summary = ModelSummary(diffae_litmodel, max_depth=1)
    print(summary)

    # Train OCSVM on train dataset
    if args.train:
        print("Train OC-SVM...")
        out = train_ocsvm_pixel(fe, device)
        clf_fits = out["fit_dicts"]
        print("finish training.")
    else: 
        clf_fits = joblib.load(os.path.join(result_dir, "ocsvm-fit-dict.pkl"))
        print("Loaded OC-SVM fitted model.")

    # Test on other datasets
    for split in ["test", "growing_circle20", "darker_line20", "darker_circle20"]:
        out = test_ocsvm_pixel(split, fe, clf_fits, device)



parser = argparse.ArgumentParser(description="Run OneclassSVM")
# parser.add_argument('--model-name', type=str)
parser.add_argument('--workdir', type=str, default="./workdir/diffae_starmen/")
parser.add_argument('--data-path', type=str, default="./data/starmen/output_random_noacc/")
parser.add_argument('--train', action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    main(args, device)
