# Modify from LDAE
# This model works with 2D toy example (Starmen) dataset so we dont need the AEKL module to calculate latent space.
# This is the conditioned DDPM (DDIM) model. Instead of pretraining an unconditional model
# and then finetune using gradient estimator network, we train the model from scratch directly with conditional signal from clean input x0
# The idea inspired from DiffAE and cDDPMS.


# Copyright (c) 2025 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
#
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
#
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------

# import wandb
import copy  # For copying model parameters
import os

import lightning as L
import numpy as np
import torch
from einops import rearrange
from generative.metrics import FIDMetric
from monai.losses import PerceptualLoss
from monai.metrics.regression import MultiScaleSSIMMetric, PSNRMetric, SSIMMetric
from src.ldae.diffusion import GaussianDiffusion
from src.tadm.models.unet import TADMUnet, RRDBNet

from src.utils.metrics import get_eval_dictionary, test_metrics
from src.utils.visualization import plot_comparison_starmen
import matplotlib.pyplot as plt
import json


class TADMLitModel(L.LightningModule):
    def __init__(
        self,
        spartial_dim = 2,
        args: dict = None, 
        timesteps_args: dict = None,
        lr: float = 2.5e-5,
        mode: str = "tadm",
        pretrained_path: str = None,
        ema_decay: float = 0.9999,
        ema_update_after_step: int = 1,
        log_img_every_epoch: int = 10,
        test_ddim_style: str = "ddim100",
        test_noise_level: int = 250,
    ):
        """
        Initialize Temporal Aware DDPMs LightningModule.

        Parameters
        ----------
        spartial_dim : Number of spatial dimensions of the input data (e.g., 2 for images, 3 for volumetric data).
        enc_args : Dictionary of arguments for the encoder network configuration.
        unet_args : Dictionary of arguments for the U-Net architecture configuration.
        vae_args : Dictionary of arguments for the Variational Autoencoder (VAE) configuration.
        vae_path : str, optional. Path to a pretrained VAE model checkpoint to load.
        timesteps_args : define diffusion timesteps. It's dict require: the number of timesteps (timesteps, default 1000)
            and betas schedule type (betas_type - default "linear")
        lr : Learning rate for the optimizer.
        mode : Mode of operation, e.g., "pretrain" or "representation-learning"
        pretrained_path : if mode == "representation-learning", path to a pretrained diffusion model checkpoint.
        ema_decay : Exponential moving average (EMA) decay rate for model parameters.
        ema_update_after_step : Number of steps (not epoch) after which EMA updates start.
        log_img_every_epoch : Frequency (in epochs) of logging reconstructed images for visualization.
        test_ddim_style : String in format "ddim<nb of infer step> during inference with DDIM sampling.
            For example: "ddim100" for 100 DDIM sampling steps.

        """
        super().__init__()
        # self.example_input_array = torch.Tensor(2, 10, 1, 64, 64)  # [B, T, C, H, W]
        self.spartial_dim = spartial_dim
        self.log_img_every_epoch = log_img_every_epoch
        self.mode = mode
        self.lr = lr
        self.timesteps_args = timesteps_args
        self.ema_decay = ema_decay
        self.ema_update_after_step = ema_update_after_step
        self.test_ddim_style = test_ddim_style
        self.test_noise_level = test_noise_level

        # Set the similarity metrics
        self.ssim = SSIMMetric(
            spatial_dims=self.spartial_dim, data_range=1.0, win_size=4
        )
        self.mssim = MultiScaleSSIMMetric(
            spatial_dims=self.spartial_dim, data_range=1.0, kernel_size=4
        )
        self.psnr = PSNRMetric(max_val=1.0, reduction="mean")

        # Diffusion helper
        self.gaussian_diffusion = GaussianDiffusion(timesteps_args, device=self.device)

        self.encoder = None
        self.decoder = None

        # ------------------- Longitudinal Learning Models -------------------
        if mode == "tadm":
            self.args = args
            # RRDB encoder
            if self.args['use_rrdb']:
                self.encoder = RRDBNet(1, 3, self.args['rrdb_num_feat'], self.args['rrdb_num_block'],
                                    self.args['rrdb_num_feat'] // 2, hparams=self.args)
            else:
                self.encoder = None

            # TADMUNet decoder
            self.decoder = TADMUnet(dim=32, 
                                    out_dim=1, 
                                    dim_mults=(1, 2, 2, 4), 
                                    cond_dim=self.args['rrdb_num_feat'], 
                                    hparams=self.args)

            # Create EMA copies (disable grad, so they won't update by backprop)
            if self.encoder is not None:
                self.ema_encoder = copy.deepcopy(self.encoder).eval()
                for p in self.ema_encoder.parameters():
                    p.requires_grad = False

            self.ema_decoder = copy.deepcopy(self.decoder).eval()
            for p in self.ema_decoder.parameters():
                p.requires_grad = False

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Save hyperparameters
        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        """
        Called when fit begins. Re-initialize the diffusion if needed and set train modes.
        """
        self.gaussian_diffusion = GaussianDiffusion(
            self.timesteps_args, device=self.device
        )
        print("on_fit_start device:", self.device)

        self.encoder.train()
        self.decoder.train()
        print("Encoder and Decoder set to train mode.")

    def on_test_start(self) -> None:
        self.gaussian_diffusion = GaussianDiffusion(
            self.timesteps_args, device=self.device
        )
        # Add similarity metrics
        self.lpips = PerceptualLoss(
            spatial_dims=self.spartial_dim,
            network_type="squeeze",
            is_fake_3d=True if self.spartial_dim == 3 else False,
            fake_3d_ratio=0.5,
        )
        self.lpips = self.lpips.to(self.device)

        print("LPIPS models loaded successfully.")

        # Add FID metric if testing pretraining stage
        if self.mode == "pretrain":
            del self.unet
            print("Deleted UNet model, maintaining only EMA model.")
        elif self.mode == "tadm":
            # # Delete the non-EMA models
            # del self.encoder
            # del self.decoder
            # print("Deleted encoder and decoder models, maintaining only EMA models.")

            # Setup eval dict
            self.eval_dict_previous = {
                "ids": [],
                "ages": [],
                "diff_ages": [],
                "ssim_metric": [],
                "mssim_metric": [],
                "psnr_metric": [],
                "l1err": [],
                "l2err": [],
                "lpips_metric": []
            }

            self.eval_dict_random_pair = {
                "ids": [],
                "ages": [],
                "diff_ages": [],
                "ssim_metric": [],
                "mssim_metric": [],
                "psnr_metric": [],
                "l1err": [],
                "l2err": [],
                "lpips_metric": []
            }

            self.eval_dict_masked_input = {
                "ids": [],
                "mask": [],
                "perc_seen": [],
                "ages": [],
                "diff_ages": [],
                "ssim_metric": [],
                "mssim_metric": [],
                "psnr_metric": [],
                "l1err": [],
                "l2err": [],
                "lpips_metric": []
            }

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.result_path = os.path.join(
            self.trainer.logger.save_dir, self.trainer.logger.name, "results"
        )
        os.makedirs(self.result_path, exist_ok=True)
        self.result_dict_path = os.path.join(self.result_path, "eval_dict.json")

    @torch.no_grad()
    def decode(self, z):
        # Decode latents using the VAE (stage 2)
        # We include logical flow to check if we need to perform decode
        # for some toy example, we dont need to use VAE to compress original image
        if self.vae:
            z = z / self.scale_factor
            self.vae.eval()
            with torch.no_grad():
                recon = self.vae.decode_stage_2_outputs(z)
            return recon
        else:
            return z

    def training_step(self, batch, batch_idx):
        """
        Standard training loop:
          - Either pretraining the UNet in latent space
          - Or learning a latent representation (encoder + ShiftUNet)
        """
        x0 = batch["x_origin"]
        all_ages = batch["age"]

        # # Option 1: Select pair of current and previous scan
        # img_lr = x0[:, :-1, ...]
        # img_hr = x0[:, 1:, ...]
        # img_hr = rearrange(img_hr, "b t ... -> (b t) ...")  # prior image
        # img_lr = rearrange(img_lr, "b t ... -> (b t) ...")  # target image to predict
        # img_lr_up = img_lr.clone()  # just to align with TADM source code.

        # # ages and diff ages
        # ages = batch["age"][:, :-1, ...]
        # diff_ages = torch.diff(batch["age"])
        # diff_ages = rearrange(diff_ages, "b t ... -> (b t) ...")
        # ages = rearrange(ages, "b t ... -> (b t) ...")

        # Option 2: Select randomly a pair of past and future scan
        # with age_lr < age_hr
        # sample two distinct indices and sort them
        b, t = all_ages.shape

        idx = torch.stack([torch.randperm(t)[:2] for _ in range(b)])

        # Ensure a < b
        idx, _ = torch.sort(idx, dim=1)

        all_batch_idx = torch.arange(b)
        img_lr = x0[all_batch_idx, idx[:,0]]
        img_hr = x0[all_batch_idx, idx[:,1]]
        img_lr_up = img_lr.clone()

        ages = all_ages[all_batch_idx, idx[:, 0]]
        ages_hr = all_ages[all_batch_idx, idx[:, 1]]
        diff_ages = ages_hr - ages

        # ------------------- 1) TADM learning -------------------
        output = None
        if self.mode == "tadm":

            # 1. Encoder image using RRDB network
            if self.args['use_rrdb']:
                if self.args['fix_rrdb']:
                    self.encoder.eval()
                    with torch.no_grad():
                        rrdb_out, cond = self.encoder(img_lr, True)
                else:
                    rrdb_out, cond = self.encoder(img_lr, True)
            else:
                rrdb_out = img_lr_up
                cond = img_lr

            # 2. Regular train 1 batch with residual image (target - start)
            x = img_hr - img_lr

            output = self.gaussian_diffusion.regular_train_tadm_one_batch(
                denoise_fn=self.decoder, 
                x_0=x, 
                ages=ages,
                diff_ages=diff_ages,
                condition=cond
            )

            if output is None:
                raise ValueError(f"Invalid mode: {self.mode}")

            loss = output["prediction_loss"]

            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0]
            )

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        After each training batch, update the EMA parameters if the step
        matches the `ema_update_after_step` criteria.
        """
        # Global step in PL counts across epochs
        if self.global_step % self.ema_update_after_step == 0 and self.global_step > 0:
            if self.mode == "pretrain":
                self.update_ema(
                    ema_model=self.ema_unet,
                    current_model=self.unet,
                    decay=self.ema_decay,
                )
            elif self.mode == "tadm":
                self.update_ema(
                    ema_model=self.ema_encoder,
                    current_model=self.encoder,
                    decay=self.ema_decay,
                )
                self.update_ema(
                    ema_model=self.ema_decoder,
                    current_model=self.decoder,
                    decay=self.ema_decay,
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

    @torch.no_grad()
    def update_ema(self, ema_model, current_model, decay):
        """
        Update EMA model parameters by a factor of `decay`.
        """
        ema_state_dict = ema_model.state_dict()
        current_state_dict = current_model.state_dict()

        for key, param in current_state_dict.items():
            if param.dtype in (torch.float16, torch.float32, torch.float64):
                ema_state_dict[key].data.mul_(decay).add_(param.data, alpha=1 - decay)

    def validation_step(self, batch, batch_idx):
        """
        Here we run inference to visualize outputs.
        """

        x0 = batch["x_origin"]
        all_ages = batch["age"]

        # Option 2: Select randomly a pair of past and future scan
        # with age_lr < age_hr
        # sample two distinct indices and sort them
        b, t = all_ages.shape

        idx = torch.stack([torch.randperm(t)[:2] for _ in range(b)])

        # Ensure a < b
        idx, _ = torch.sort(idx, dim=1)

        all_batch_idx = torch.arange(b)
        img_lr = x0[all_batch_idx, idx[:,0]]
        img_hr = x0[all_batch_idx, idx[:,1]]
        img_lr_up = img_lr.clone()

        ages = all_ages[all_batch_idx, idx[:, 0]]
        ages_hr = all_ages[all_batch_idx, idx[:, 1]]
        diff_ages = ages_hr - ages

        if self.mode == "tadm":
            if self.args["use_rrdb"]:
                self.encoder.eval()
                with torch.no_grad():
                    rrdb_out, cond = self.encoder(img_lr, True)
            else:
                rrdb_out = img_lr_up, 
                cond = img_lr

            # 2. Regular train 1 batch with residual image (target - start)
            x = img_hr - img_lr
            output = self.gaussian_diffusion.regular_train_tadm_one_batch(
                denoise_fn=self.decoder, 
                x_0=x, 
                ages=ages,
                diff_ages=diff_ages,
                condition=cond
            )

            val_dict = {
                "val_loss": output["prediction_loss"]
            }

            self.log_dict(val_dict,
                          prog_bar=True,
                          on_step=True,
                          on_epoch=True,
                          sync_dist=True,
                          batch_size=x.shape[0]
                            )        
        else: 
            raise ValueError(f"Invalid mode: {self.mode}")

    def test_step(self, batch, batch_idx):
        """
        Here we test the model reconstruction and generation capabilities.
        """

        if self.mode == "pretrain":
            pass

        elif self.mode == "tadm":

            log_fig = batch_idx % self.log_img_every_epoch == 0
            if log_fig:
                save_idx = (
                    batch_idx // self.log_img_every_epoch
                )  # so that every test run will store img with the same save_idx (easier to browse)
            writer = self.logger.experiment

            # 1. Eval using previous scan
            print(f"Evaluation using previous image...[batch_idx: {batch_idx}]")
            _, *out = self.tadm_test_step_previous(
                batch, self.eval_dict_previous, return_fig=log_fig
            )
            if log_fig and len(out) > 0:
                fig = out[0]
                caption = "prev_img"
                writer.add_figure(
                    f"test_{save_idx:03}/{caption}", fig, self.global_step
                )

            # 2. Eval using random pair
            print(f"Evaluation using random pair...[batch_idx: {batch_idx}]")
            _, *out = self.tadm_test_step_random_pair(
                batch=batch,
                eval_dict=self.eval_dict_random_pair,
                return_fig=log_fig,
                nb_pair=5,
            )
            if log_fig and len(out) > 0:
                fig = out[0]
                caption = "random_pair"
                writer.add_figure(
                    f"test_{save_idx:03}/{caption}", fig, self.global_step
                )

            # 3. Eval with masked input
            print(f"Evaluation using masked input...[batch_idx: {batch_idx}]")
            perc_seen_list = np.arange(0.1, 0.7, 0.1)
            for i, perc_seen in enumerate(perc_seen_list):
                _, *out = self.tadm_test_step_masked_input(
                    batch=batch,
                    eval_dict=self.eval_dict_masked_input,
                    perc_seen=perc_seen,
                    return_fig=log_fig,
                )

                if log_fig and len(out) > 0:
                    fig = out[0]
                    caption = f"masked_input_{perc_seen:.1f}"
                    writer.add_figure(f"test_{save_idx:03}/{caption}", fig, self.global_step)

            # 4. oversampling
            if log_fig: 
                print(f"Oversampling...[batch_idx: {batch_idx}]")
                fig = self.tadm_test_step_oversample(batch)
                writer.add_figure(f"test_{save_idx:03}/oversample", fig, self.global_step)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def on_test_epoch_end(self):

        # import pdb; pdb.set_trace()

        # save eval dict previous
        save_path = os.path.join(self.result_path, "eval_dict_previous.json")
        with open(save_path, "w") as f:
            json.dump(self.eval_dict_previous, f)
        print(f"Save eval_dict to {save_path}")

        # save eval dict random pair
        save_path = os.path.join(self.result_path, "eval_dict_random_pair.json")
        with open(save_path, "w") as f:
            json.dump(self.eval_dict_random_pair, f)
        print(f"Save eval_dict to {save_path}")

        # save eval dict random pair
        save_path = os.path.join(self.result_path, "eval_dict_masked_input.json")
        with open(save_path, "w") as f:
            json.dump(self.eval_dict_masked_input, f)
        print(f"Save eval_dict to {save_path}")

    def tadm_test_step_previous(self, batch, eval_dict=None, return_fig=False):
        """
        TADM test step using the previous scan as signal to predict the current scan
        """

        if eval_dict is None:
            eval_dict = self.eval_dict

        # subject metadata
        pidx = batch["id"].item()
        anomaly = batch["anomaly"].item()
        anomaly_type = batch["anomaly_type"][0]
        anomaly_gt_seg = batch["anomaly_gt_seg"]

        x0 = batch["x_origin"]
        img_lr = x0[:, :-1, ...]
        img_hr = x0[:, 1:, ...]
        img_hr = rearrange(img_hr, "b t ... -> (b t) ...")  # prior image
        img_lr = rearrange(img_lr, "b t ... -> (b t) ...")  # target image to predict
        img_lr_up = img_lr.clone()  # just to align with TADM source code. 

        # ages and diff ages
        ages = batch["age"][:, :-1, ...]
        diff_ages = torch.diff(batch["age"])
        diff_ages = rearrange(diff_ages, "b t ... -> (b t) ...")
        ages = rearrange(ages, "b t ... -> (b t) ...")

        # 1. Encoder image using RRDB network
        if self.args['use_rrdb']:
            rrdb_out, cond = self.encoder(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr

        # 2. Sampling x from trained TADMUnet
        x_pred = self.gaussian_diffusion.regular_tadm_sample(
            ddim_style=self.test_ddim_style,
            encoder=self.encoder,
            unet=self.decoder,
            x_0=img_lr_up,
            x_T=torch.randn_like(img_lr_up),
            z=cond,
            diff_ages=diff_ages,
            ages=ages,
            disable_tqdm=True,
        )

        # 3. Reconstruc future image
        img_pred = img_lr_up + x_pred

        # Calculate eval metrics
        ssim_metric = self.ssim(img_pred, img_hr).mean().item()
        mssim_metric = self.mssim(img_pred, img_hr).mean().item()
        psnr_metric = self.psnr(img_pred, img_hr).mean().item()
        l1err = torch.nn.functional.l1_loss(img_pred, img_hr, reduction="mean").item()
        l2err = torch.nn.functional.mse_loss(img_pred, img_hr).item()
        lpips_metric = self.lpips(img_pred, img_hr).item()

        # write to eval dict
        eval_dict["ids"].append(pidx)
        eval_dict["ages"].append(ages.cpu().numpy().tolist())
        eval_dict["diff_ages"].append(diff_ages.cpu().numpy().tolist())
        eval_dict["ssim_metric"].append(ssim_metric)
        eval_dict["mssim_metric"].append(mssim_metric)
        eval_dict["psnr_metric"].append(psnr_metric)
        eval_dict["lpips_metric"].append(lpips_metric)
        eval_dict["l1err"].append(l1err)
        eval_dict["l2err"].append(l2err)

        # Plot comparison
        if return_fig: 
            fig = plot_comparison_starmen(
                imgs=[img_pred.squeeze(), img_hr.squeeze(), torch.abs(img_hr-img_pred).squeeze()],
                labels=["recons", "origin", "error"],
                is_errors=[False, False, True],
                opt={
                    "title": "Predict current scan using previous scan (jump 1 timestep)"
                }
            )
            return eval_dict, fig
        else:
            return eval_dict

    def tadm_test_step_random_pair(self, batch, eval_dict=None, return_fig = False, nb_pair=1):
        """
        TADM test step using a random pair from a batch
        """

        if eval_dict is None:
            eval_dict = self.eval_dict

        # subject metadata
        pidx = batch["id"].item()
        anomaly = batch["anomaly"].item()
        anomaly_type = batch["anomaly_type"][0]
        anomaly_gt_seg = batch["anomaly_gt_seg"]

        x0 = batch["x_origin"]

        # ages and diff ages
        all_ages = batch["age"]

        # select a random pair
        # sample two distinct indices and sort them
        b, t = all_ages.shape

        for i in range(nb_pair):
            idx = torch.stack([torch.randperm(t)[:2] for _ in range(b)])
            idx, _ = torch.sort(idx, dim=1)  # ensure a < b

            batch_idx = torch.arange(b)
            x = x0[batch_idx, idx[:,0]]  # (b, 1, 64, 64)
            x_target = x0[batch_idx, idx[:,1]]  # (b, 1, 64, 64)

            age = all_ages[batch_idx, idx[:, 0]]
            age_target = all_ages[batch_idx, idx[:, 1]]
            diff_age = age_target - age

            if i == 0:
                img_lr = x
                img_hr = x_target
                ages = age
                diff_ages = diff_age
            else:
                img_lr = torch.cat((img_lr, x))
                img_hr = torch.cat((img_hr, x_target))
                ages = torch.cat((ages, age))
                diff_ages = torch.cat((diff_ages, diff_age))

        img_lr_up = img_lr.clone()
        pidx = np.repeat(pidx, nb_pair)

        # 1. Encoder image using RRDB network
        if self.args['use_rrdb']:
            rrdb_out, cond = self.ema_encoder(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr

        # 2. Sampling x from trained TADMUnet
        x_pred = self.gaussian_diffusion.regular_tadm_sample(
            ddim_style=self.test_ddim_style,
            encoder=self.encoder,
            unet=self.decoder,
            x_0=img_lr_up,
            x_T=torch.randn_like(img_lr_up),
            z=cond,
            diff_ages=diff_ages,
            ages=ages,
            disable_tqdm=True,
        )

        # 3. Reconstruc future image
        img_pred = img_lr_up + x_pred

        # Calculate eval metrics
        ssim_metric = self.ssim(img_pred, img_hr).cpu().numpy().squeeze()
        mssim_metric = self.mssim(img_pred, img_hr).cpu().numpy().squeeze()
        psnr_metric = self.psnr(img_pred, img_hr).cpu().numpy().squeeze()
        l1err = torch.nn.functional.l1_loss(img_pred, img_hr, reduction="none").cpu().numpy()
        l2err = torch.nn.functional.mse_loss(img_pred, img_hr, reduction="none").cpu().numpy()
        l1err = np.mean(l1err, axis=(2, 3)).squeeze()
        l2err = np.mean(l2err, axis=(2, 3)).squeeze()
        lpips_metric = []
        for i in range(img_pred.shape[0]):
            lpips_img = self.lpips(img_hr[[i]], img_pred[[i]])
            lpips_metric.append(lpips_img.item())

        # write to eval dict
        eval_dict["ids"].append(pidx.tolist())
        eval_dict["ages"].append(ages.cpu().numpy().tolist())
        eval_dict["diff_ages"].append(diff_ages.cpu().numpy().tolist())
        eval_dict["ssim_metric"].append(ssim_metric.tolist())
        eval_dict["mssim_metric"].append(mssim_metric.tolist())
        eval_dict["psnr_metric"].append(psnr_metric.tolist())
        eval_dict["lpips_metric"].append(lpips_metric)
        eval_dict["l1err"].append(l1err.tolist())
        eval_dict["l2err"].append(l2err.tolist())

        # Plot comparison
        if return_fig: 
            title_fontsize = 10
            b = img_hr.shape[0]
            fig, axes = plt.subplots(b, 4, figsize=(8, b*2))
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            for ax in axes.flatten():
                ax.axis("off")

            for i in range(b):
                # based image
                ax = axes[i, 0]
                ax.imshow(img_lr_up[i].cpu().squeeze(), cmap="gray")
                ax.set_title(f"based age={ages[i]:.2f}", fontsize=title_fontsize)

                # taret image
                ax = axes[i, 1]
                ax.imshow(img_hr[i].cpu().squeeze(), cmap="gray")
                ax.set_title(f"target age={ages[i] + diff_ages[i]:.2f}", fontsize=title_fontsize)

                # recons image
                ax = axes[i, 2]
                ax.imshow(img_pred[i].cpu().squeeze(), cmap="gray")
                ax.set_title(f"predicted image", fontsize=title_fontsize)

                # diff
                ax = axes[i, 3]
                diff_img = img_hr[i] - img_pred[i]
                im = ax.imshow(diff_img.cpu().squeeze(), cmap="jet")
                ax.set_title("error", fontsize=title_fontsize)
                cax = inset_axes(ax,
                                width="5%",   # width relative to ax
                                height="100%",  # full height
                                loc='right',
                                )  
                fig.colorbar(im, cax=cax)
            plt.subplots_adjust(hspace=0.2, wspace=0.2)
            return eval_dict, fig
        else:
            return eval_dict

    def tadm_test_step_masked_input(self, batch, eval_dict, return_fig=False, perc_seen=0.2):
        """
        TADM test step using the masked input
        """

        # subject metadata
        pidx = batch["id"].item()
        anomaly = batch["anomaly"].item()
        anomaly_type = batch["anomaly_type"][0]
        anomaly_gt_seg = batch["anomaly_gt_seg"]

        x0 = batch["x_origin"].squeeze(0)
        ages = batch["age"].squeeze()

        # generate mask
        t, c, h, w = x0.shape

        seen_mask = torch.full((t,), 0., device=x0.device)
        nb_seen = int(perc_seen * t)

        seen_idxs = torch.randperm(t - 1)[:nb_seen-1] + 1
        seen_mask[seen_idxs] = 1.
        seen_mask[0] = 1.

        x_masked = x0 * rearrange(seen_mask, "t -> t 1 1 1")
        ages_masked = ages * seen_mask

        # Generate the whole sequence from masked input
        x_recons = x_masked.clone()
        prev_x = x_recons[0]

        for i, x in enumerate(x_recons):
            if seen_mask[i] == 1.:
                prev_x = x
                prev_age = ages[i]
            else:
                img_lr = prev_x.unsqueeze(0)

                diff_age = (ages[i] - prev_age).unsqueeze(0)

                # 1. encode image with RRDB
                rrdb_out, cond = self.ema_encoder(img_lr, True)

                # 2. Sampling x from trained TADMUnet
                x_pred = self.gaussian_diffusion.regular_tadm_sample(
                    ddim_style=self.test_ddim_style,
                    encoder=self.encoder,
                    unet=self.decoder,
                    x_0=img_lr,
                    x_T=torch.randn_like(img_lr),
                    z=cond,
                    diff_ages=diff_age,
                    ages=prev_age.unsqueeze(0),
                    disable_tqdm=True,
                )

                # 3. Predict future image
                img_pred = (img_lr + x_pred).squeeze(0)
                x_recons[i] = img_pred
                prev_x = img_pred
                prev_age = ages[i]

        # Calculate eval metrics
        img_pred = x_recons[seen_mask == 0.]
        img_hr = x0[seen_mask == 0.]

        ssim_metric = self.ssim(img_pred, img_hr).mean().item()
        mssim_metric = self.mssim(img_pred, img_hr).mean().item()
        psnr_metric = self.psnr(img_pred, img_hr).mean().item()
        l1err = torch.nn.functional.l1_loss(img_pred, img_hr, reduction="mean").item()
        l2err = torch.nn.functional.mse_loss(img_pred, img_hr).item()
        lpips_metric = self.lpips(img_pred, img_hr).item()

        # write to eval dict
        eval_dict["ids"].append(pidx)
        # eval_dict["ages"].append(ages.cpu().numpy())
        # eval_dict["diff_ages"].append(diff_ages.cpu().numpy())
        eval_dict["perc_seen"].append(perc_seen)
        eval_dict["mask"].append(seen_mask.cpu().numpy().tolist())
        eval_dict["ssim_metric"].append(ssim_metric)
        eval_dict["mssim_metric"].append(mssim_metric)
        eval_dict["psnr_metric"].append(psnr_metric)
        eval_dict["lpips_metric"].append(lpips_metric)
        eval_dict["l1err"].append(l1err)
        eval_dict["l2err"].append(l2err)

        # Plot comparison
        if return_fig: 
            # Plot comparison
            imgs = [
                x0.detach().cpu().squeeze(), 
                x_masked.detach().cpu().squeeze(),
                x_recons.detach().cpu().squeeze(),
                torch.abs(x0 - x_recons).detach().cpu().squeeze(),
            ]

            labels = [
                "Origin",
                "Mask",
                "Reconstruction",
                "Error"
            ]
            opt = {
                "title": f"Reconstruction error - mask %: {(1 - perc_seen):.1%}"
            }

            fig = plot_comparison_starmen(imgs, labels, is_errors=[False, False, False, True], opt=opt)
            return eval_dict, fig
        else:
            return eval_dict

    def tadm_test_step_oversample(self, batch, oversample_step=7):
        """
        TADM test step with oversample from the last sample
        """

        # subject metadata
        pidx = batch["id"].item()

        x0 = batch["x_origin"].squeeze(0)
        ages = batch["age"].squeeze()

        # Start from the last
        img_lr = x0[9].unsqueeze(0)
        prev_age = ages[9].unsqueeze(0)

        x_oversample = torch.zeros((oversample_step, 1, 64, 64))

        diff_age = torch.tensor([1.], device=x0.device)
        for i in range(oversample_step):            
            # 1. encode image with RRDB
            rrdb_out, cond = self.encoder(img_lr, True)

            # 2. Sampling x from trained TADMUnet
            x_pred = self.gaussian_diffusion.regular_tadm_sample(
                ddim_style=self.test_ddim_style,
                encoder=self.encoder,
                unet=self.decoder,
                x_0=img_lr,
                x_T=torch.randn_like(img_lr),
                z=cond,
                diff_ages=diff_age,
                ages=prev_age,
                disable_tqdm=True,
            )

            # 3. Predict future image
            img_pred = (img_lr + x_pred)
            x_oversample[i] = img_pred.squeeze(0)
            img_lr = img_pred
            prev_age += diff_age

        # Plot
        import matplotlib.gridspec as gridspec

        # move result to cpu
        x0 = x0.cpu()
        x_oversample = x_oversample.cpu()

        fig = plt.figure(figsize=(10, 3))
        gs = gridspec.GridSpec(2, 2, 
                            figure=fig,
                            width_ratios=[4, 7], 
                            height_ratios=[1, 1] )

        fig.subplots_adjust(
            # left=0.05, right=0.95,  # shrink/grow figure margins
            # bottom=0.1, top=1.0,
            wspace=0.05, hspace=0.0  # spacing between subplots
        )

        ax1 = fig.add_subplot(gs[0, 0])    
        ax2 = fig.add_subplot(gs[0, 1])     
        ax3 = fig.add_subplot(gs[1, 1])     

        # original sequence
        img = x0[6:]
        img = rearrange(img, "b c h w -> h (b c w)")
        ax1.imshow(img, cmap="gray") 
        ax1.axis("off")
        ax1.set_title("original")

        # oversample sequence
        img = rearrange(x_oversample, "b c h w -> h (b c w)")
        ax2.imshow(img, cmap="gray") 
        ax2.axis("off")
        ax2.set_title("oversample")

        # difference
        img = x_oversample - x0[-1]
        img = rearrange(img, "b c h w -> h (b c w)")
        im = ax3.imshow(img, cmap="jet") 
        ax3.axis("off")
        ax3.set_title("difference compare to the last sample")

        bbox = ax3.get_position()  # [x0, y0, width, height] in figure coords
        cax = fig.add_axes([bbox.x0 - 0.02, bbox.y0, 0.01, bbox.height])  
        # [left, bottom, width, height] in figure coordinates

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        for ax in fig.get_axes():  # get all axes in the figure
            ax.title.set_fontsize(11)
        fig.suptitle(f"Oversample - idx: {pidx} - increasing timestep: 1.", fontsize=11, y=0.95)
        return fig

    def tb_display_generation(self, step, images, tag="Generated"):
        """
        Display generation result in TensorBoard during training
        """
        images = images.detach().cpu().numpy()[:, 0, :, :]
        images = rearrange(images, "b h w -> h (b w)")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 3))
        plt.style.use("dark_background")
        plt.imshow(images, cmap="gray")

        plt.axis("off")
        plt.tight_layout()
        # Use the built-in SummaryWriter from Lightning
        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_figure(tag, plt.gcf(), global_step=step)
        plt.close(fig)

    def tb_log_image_batch_2d(self, images, job="fit", caption="generate", idx=0):
        import torchvision

        """
        Log the guided generated images (2D normal img) to TensorBoard

        Args: 
            job: str, type of job in ["fit", "validate", "test" ]. This is used to created Tensorboard tag. 
            images: Tensor images [B, C, H, W]. We use torchvision so it needs to be tensor - not numpy. 
            caption: Used to generate Tensorboard tags, for example: val_ep010_generate, val_ep050_origin, etc...
        """

        writer = self.logger.experiment

        B = images.shape[0]
        max_patient = 5  # log maximum 3 patients
        slices_per_patient = 10
        max_slices = max_patient * slices_per_patient
        # Truncate to desired number of slices
        if B > max_slices:
            images = images[:max_slices]

        grid = torchvision.utils.make_grid(
            images, nrow=slices_per_patient, normalize=True, scale_each=True
        )
        if job == "fit":
            tag = f"{job}_ep{self.current_epoch:03d}/{caption}"
        elif job in ["validate", "test"]:
            # tag = f"{job}_{idx}/{caption}"
            tag = f"test_{idx:03}/{caption}"
        writer.add_image(tag, grid, self.global_step)

    def tb_log_plot_comparison_2d(
        self,
        idx,
        caption,
        x_origin,
        x_recons,
        x_recons_semantic,
    ):
        """
        Log plot of comparison to TensorBoard.
        Plot will have 3 rows and 1 column for color bar cmap.
            - Original images. list[np.ndarray]
            - Reconstruction images (can be x_inferred or x_hat)
            - Error
        All images need to be in np.ndarray format [B, H, W]. B = 10 for StarmenDataset.
        """
        writer = self.logger.experiment

        imgs = [
            x_origin.detach().cpu().squeeze(),
            x_recons.detach().cpu().squeeze(),
            torch.abs(x_origin - x_recons).detach().cpu().squeeze(),
            x_recons_semantic.detach().cpu().squeeze(),
            torch.abs(x_origin - x_recons_semantic).detach().cpu().squeeze(),
        ]

        labels = [
            "origin",
            "recons_infer",
            "error_infer",
            "recons_semantic",
            "error_semantic",
        ]

        is_errors = [False, False, True, False, True]

        opts = {"title": f"Reconsuction errors", "base_size": 1.2}

        fig = plot_comparison_starmen(imgs, labels, is_errors, opt=opts, same_cbar=True)
        writer.add_figure(f"test_{idx:03}/{caption}", fig, self.global_step)

    def tb_log_plot_heatmap(
        self,
        idx,
        caption,
        x_origin,
        x_recons,
    ):

        from src.ldae.utils.anomaly_map import heat_map

        writer = self.logger.experiment

        ano_map, f_d, i_d = heat_map(
            x_recons, x_origin, self.fe, v=self.heatmap_v, fe_layers=self.fe_layers
        )

        imgs = [
            x_origin.detach().cpu().squeeze(),
            x_recons.detach().cpu().squeeze(),
            f_d.detach().cpu().squeeze(),
            i_d.detach().cpu().squeeze(),
            ano_map.detach().cpu().squeeze(),
        ]

        labels = ["origin", "recons", "f_d", "i_d", "ano_score_map"]

        title = f"Anomaly score map - v={self.heatmap_v}"
        opt = {"title": title}

        fig = plot_comparison_starmen(
            imgs,
            labels,
            is_errors=[False, False, True, True, True],
            opt=opt,
            same_cbar=False,
            display_cbar=True,
        )
        writer.add_figure(f"test_{idx:03}/{caption}", fig, self.global_step)

    def tb_log_plot_example_histogram_heat_map(self, x, xhat, ano_gt):

        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        from src.ldae.utils.anomaly_map import heat_map

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(
            3, 4, figure=fig, wspace=0.05, hspace=0.3, width_ratios=[1, 1, 1, 1]
        )

        ano_map, f_d, i_d = heat_map(
            x, xhat, self.fe, v=self.heatmap_v, fe_layers=self.fe_layers
        )

        # Turn off all axes by default
        axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)]
        for row in axes:
            for ax in row:
                ax.axis("off")

        # Plot original image
        ax = axes[0][0]
        ax.imshow(x, cmap="gray")
        ax.set_title("origin")

        # Plot ground truth anomaly
        ax = axes[0][3]
        ax.imshow(ano_gt, cmap="gray")
        ax.set_title("annotation")

        # Reconstruction xT_infer
        ax = axes[1][0]
        ax.imshow(xhat, cmap="gray")
        ax.set_title("reconstruction")
        ax.set_ylabel("xT_infer")

        # Pixel distance
        ax = axes[1][1]
        ax.imshow(i_d.squeeze(), cmap="inferno")
        ax.set_title("pixel distance")

        # Feature distance
        ax = axes[1][2]
        ax.imshow(f_d.squeeze(), cmap="inferno")
        ax.set_title("feature distance")

        # Anomaly map
        ax = axes[1][3]
        ax.imshow(ano_map.squeeze(), cmap="inferno")
        ax.set_title(f"anomaly score v={self.heatmap_v}")

        # Histogram: pixel density between x and xhat
        ax = axes[2][0]
        xs = [x.squeeze().flatten(), xhat.squeeze().flatten()]
        labels = ["input", "recons"]
        ax.hist(xs, bins=30, alpha=0.6, label=labels, density=True, histtype="step")
        ax.set_title("pixel density")

        # Histogram: pixel distance
        ax = axes[2][1]
        i_d_ano = i_d.squeeze()[ano_gt > 0]
        i_d_ht = i_d.squeeze()[ano_gt == 0]

        xs = [i_d_ano, i_d_ht]
        labels = ["anomaly", "healthy"]
        ax.hist(xs, bins=30, alpha=0.6, label=labels, density=True, histtype="stepfilled")
        ax.legend()

        # Histogram: feature distance
        ax = axes[2][2]
        f_d_ano = f_d.squeeze()[ano_gt > 0]
        f_d_ht = f_d.squeeze()[ano_gt == 0]
        xs = [f_d_ano, f_d_ht]
        labels = ["anomaly", "healthy"]
        ax.hist(xs, bins=30, alpha=0.6, label=labels, density=True, histtype="stepfilled")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        # Histogram: anomaly map
        ax = axes[2][3]
        ano_map_ano = ano_map.squeeze()[ano_gt > 0]
        ano_map_ht = ano_map.squeeze()[ano_gt == 0]
        xs = [ano_map_ano, ano_map_ht]
        labels = ["anomaly", "healthy"]
        ax.hist(xs, bins=30, alpha=0.6, label=labels, density=True, histtype="stepfilled")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        # Customize histogram axes in third row
        for ax in [axes[2][0], axes[2][1], axes[2][2], axes[2][3]]:
            ax.axis("on")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.tick_params(
                axis="both",
                which="both",
                bottom=True,
                top=False,
                left=False,
                right=False,
                labelbottom=True,
                labelleft=False,
            )

        # plt.tight_layout()
        plt.show()

    # def log_image(self, recon, caption="generated"):
    #     """
    #     Helper for logging 3D volume slices to Weights & Biases.
    #     """
    #     images_recon = [
    #         recon[:, :, recon.shape[2] // 2],
    #         recon[:, recon.shape[1] // 2, :],
    #         recon[recon.shape[0] // 2, :, :]
    #     ]
    #     captions_recon = [
    #         f"Center axial {caption}",
    #         f"Center coronal {caption}",
    #         f"Center sagittal {caption}"
    #     ]

    #     # Log each image separately using Wandb
    #     wandb_logger = self.logger.experiment
    #     for i, (img_recon, cap_recon) in enumerate(zip(images_recon, captions_recon)):
    #         wandb_logger.log({
    #             f"val_{caption}_slice_{i}": wandb.Image(img_recon, caption=cap_recon),
    #         })

    # def log_image_batch(self, recons, caption="generated"):
    #     """
    #     Log the center slices (axial, coronal, and sagittal) for an entire batch
    #     of 3D volumes to Weights & Biases.

    #     Args:
    #         recons (np.ndarray or torch.Tensor): A 4D array of shape [B, H, W, D].
    #         caption (str): Optional string to append to logging captions.
    #     """
    #     # If you're using PyTorch Tensors, you may want to convert to CPU & numpy:
    #     # recons = recons.detach().cpu().numpy()

    #     # Unpack shapes
    #     B, C, H, W, D = recons.shape

    #     # Grab center slices for each item in the batch
    #     # Axial slice is the middle slice along depth D
    #     axial_slices = recons[:, 0, :, :, D // 2]  # shape [B, H, W]

    #     # Coronal slice is the middle slice along width W
    #     # (note: shape [B, H, D])
    #     coronal_slices = recons[:, 0, :, W // 2, :]

    #     # Sagittal slice is the middle slice along height H
    #     # (note: shape [B, W, D])
    #     sagittal_slices = recons[:, 0, H // 2, :, :]

    #     # Prepare logger
    #     wandb_logger = self.logger.experiment

    #     # Helper to turn a batch of 2D slices into a list of wandb.Images
    #     def slices_to_wandb_images(slices_2d, view_name):
    #         """
    #         slices_2d is a 3D array of shape [B, height, width] for that particular view.
    #         We'll make a list of wandb.Image objects, one per batch item.
    #         """
    #         images = []
    #         for idx in range(B):
    #             # Each slice is a 2D array: [height, width]
    #             img_2d = slices_2d[idx]
    #             # Convert to wandb.Image, optionally applying a caption
    #             images.append(
    #                 wandb.Image(img_2d, caption=f"{caption} {view_name}, batch idx {idx}")
    #             )
    #         return images

    #     # Convert the slices for each view into a list of wandb.Images
    #     axial_wandb_images = slices_to_wandb_images(axial_slices, "axial")
    #     coronal_wandb_images = slices_to_wandb_images(coronal_slices, "coronal")
    #     sagittal_wandb_images = slices_to_wandb_images(sagittal_slices, "sagittal")

    #     # Finally, log them to W&B. Each key will appear as a gallery of images in the UI.
    #     wandb_logger.log({
    #         f"val_{caption}_axial": axial_wandb_images,
    #         f"val_{caption}_coronal": coronal_wandb_images,
    #         f"val_{caption}_sagittal": sagittal_wandb_images,
    #     })

    def configure_optimizers(self):
        """
        Return optimizer for either the UNet or (encoder + ShiftUNet).
        """
        if self.mode == "pretrain":
            return torch.optim.Adam(
                [
                    {"params": self.unet.parameters()},
                ],
                lr=self.lr,
                eps=1.0e-08,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )
        elif self.mode == "tadm":
            return torch.optim.Adam(
                [
                    {"params": self.encoder.parameters()},
                    {"params": self.decoder.parameters()},
                ],
                lr=self.lr,
                eps=1.0e-08,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
