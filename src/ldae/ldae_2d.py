# Modify from LDAE
# This model works with 2D toy example (Starmen) dataset so we dont need the AEKL module to calculate latent space.


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
from generative.networks.nets import AutoencoderKL

from monai.losses import PerceptualLoss
from monai.metrics.regression import MultiScaleSSIMMetric, PSNRMetric, SSIMMetric
from src.ldae.diffusion import GaussianDiffusion
from src.ldae.nets import AttentionSemanticEncoder, ShiftUNet, UNet, FeatureExtractor
from src.utils.metrics import get_eval_dictionary, test_metrics
from src.utils.visualization import (
    tb_log_plot_comparison_2d,
    tb_log_plot_heatmap,
)


class LatentDiffusionAutoencoders2D(L.LightningModule):
    def __init__(
        self,
        spartial_dim: int = 2,
        enc_args: dict = None,
        unet_args: dict = None,
        vae_args: dict = None,
        vae_path: str = None,
        timesteps_args: dict = None,
        lr: float = 2.5e-5,
        mode: str = "pretrain",
        pretrained_path: str = None,
        ema_decay: float = 0.9999,
        ema_update_after_step: int = 1,
        log_img_every_epoch: int = 10,
        test_ddim_style: str = "ddim100",
        test_noise_level: int = 250,
        use_xT_inferred: bool = False,
        heatmap_v: float = 6.0,
        fe_layers: list = ["layer1", "layer2", "layer3"],
    ):
        """
            Initialize LDAE LightningModule.

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
        self.use_xT_inferred = use_xT_inferred
        self.heatmap_v = heatmap_v
        self.fe_layers = fe_layers

        # Set the metrics (for validation only)
        self.ssim = SSIMMetric(
            spatial_dims=self.spartial_dim, data_range=1.0, win_size=4
        )  

        # Diffusion helper
        self.gaussian_diffusion = GaussianDiffusion(
            timesteps_args, device=self.device
        )

        self.encoder = None
        self.unet = None
        self.decoder = None
        self.vae = None

        # ------------------- Autoencoder (VAE) -------------------
        if vae_args is not None: 
            self.vae = AutoencoderKL(**vae_args)
            if vae_path is not None:
                print(f"Loading VAE from {vae_path}")
                self.vae.load_state_dict(torch.load(vae_path))
                print("VAE model loaded successfully.")

            # Init the scale factor
            self.scale_factor = torch.tensor(0.8730)
        """
                Computes the scaling factor for normalizing the latent space representation of inputs.

                This function encodes the inputs from the first batch of the provided dataloader using the
                specified autoencoder. It then calculates the reciprocal of the standard deviation of the
                resulting latent variables. This scaling factor is used to normalize the latent space,
                ensuring that the latent variables have unit standard deviation. Such normalization is
                essential for the stable training of diffusion models in the latent space, as it standardizes
                the latent variables, making the diffusion process more stable and effective.

                See https://arxiv.org/pdf/2112.10752 Appendix G Details on Autoencoder Models for more details.
        """

        # ------------------- Pretrain Models -------------------
        if mode == "pretrain":
            self.unet = UNet(**unet_args)

            # Create EMA copy of UNet (disable grad, so it won't update by backprop)
            self.ema_unet = copy.deepcopy(self.unet).eval()
            for p in self.ema_unet.parameters():
                p.requires_grad = False

        # ------------------- Representation-Learning Models -------------------
        elif mode == "representation-learning":
            # Load the pretrained denoising model
            weights = None
            try:
                data = torch.load(pretrained_path, map_location=self.device)
                weights = {key.replace('ema_unet.', ''): value for key, value in data['state_dict'].items()}
            except FileNotFoundError:
                print(f"Pretrained model not found at {pretrained_path}")

            # ShiftUNet decoder
            self.decoder = ShiftUNet(
                latent_dim=enc_args["emb_chans"],
                **unet_args,
            )
            if weights is not None:
                self.decoder.load_state_dict(weights, strict=False)
                print("Successfully loaded the pretrained decoder model.")
            else:
                print("No weights loaded for the decoder.")

            # Semantic encoder
            self.encoder = AttentionSemanticEncoder(**enc_args)

            # Create EMA copies (disable grad, so they won't update by backprop)
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

        if self.mode == "representation-learning":
            self.encoder.train()
            self.decoder.set_train_mode()
            print("Encoder and decoder set to training mode.")

    def on_test_start(self) -> None:
        self.gaussian_diffusion = GaussianDiffusion(
            self.timesteps_args, device=self.device
        )
        # Add similarity metrics
        self.lpips = PerceptualLoss(spatial_dims=self.spartial_dim,
                                    network_type="squeeze",
                                    is_fake_3d=True if self.spartial_dim == 3 else False,
                                    fake_3d_ratio=0.5)
        self.lpips = self.lpips.to(self.device)
        self.ssim = SSIMMetric(
            spatial_dims=self.spartial_dim, data_range=1.0, win_size=4
        )
        self.mssim = MultiScaleSSIMMetric(
            spatial_dims=self.spartial_dim, data_range=1.0, kernel_size=4
        )
        self.psnr = PSNRMetric(max_val=1.0, reduction="mean")
        print("LPIPS/SSIM/MSSIM/PSNR models loaded successfully.")

        # Add FID metric if testing pretraining stage
        if self.mode == "pretrain":
            del self.unet
            print("Deleted UNet model, maintaining only EMA model.")
        elif self.mode == "representation-learning":
            # Delete the non-EMA models
            del self.encoder
            del self.decoder
            print("Deleted encoder and decoder models, maintaining only EMA models.")

            self.fe = FeatureExtractor(encoder=self.ema_encoder)
            self.fe = self.fe.float().to(self.device)
            # Setup eval dict
            self.eval_dict = get_eval_dictionary()
            self.eval_dict["test_ddim"] = self.test_ddim_style
            self.eval_dict["test_noise_level"] = self.test_noise_level

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Init array to store reconstruction result
        if self.use_xT_inferred:
            self.recons_np = []
        self.recons_semantic_np = [] 
        self.recons_idx = []
        self.origins = []  # we dont need to store the origin, this is just to calculate histogram at on_test_epoch_end
        self.result_path = os.path.join(
            self.trainer.logger.save_dir,
            self.trainer.logger.name,
            "results"
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

        if self.vae is None: 
            z0 = batch["x_origin"]
            z0 = rearrange(z0, "b t c h w -> (b t) c h w")
        else:
            z0 = batch["latent"]
            z0 = z0 * self.scale_factor
        output = None

        # ------------------- 1) Denoiser pretraining -------------------
        if self.mode == "pretrain":
            output = self.gaussian_diffusion.regular_train_one_batch(
                denoise_fn=self.unet,
                x_0=z0,
            )

        # ------------------- 2) Representation learning -------------------
        elif self.mode == "representation-learning" or self.mode == "refining":
            if self.vae is None: 
                # we train directly on original images (PDAE)
                # x0 = z0.clone().detach().requires_grad_(z0.requires_grad)
                output = self.gaussian_diffusion.representation_learning_train_one_batch(
                    encoder=self.encoder, 
                    decoder=self.decoder,
                    x_0=z0
                )
            elif self.vae is not None: 
                # we train on compressed latent representation of original images (LDAE)
                # Original images still used to generate semantic encoded input
                x0 = batch["image"]
                output = self.gaussian_diffusion.latent_representation_learning_train_one_batch(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    x_0=x0,
                    z_0=z0,
                )

        if output is None:
            raise ValueError(f"Invalid mode: {self.mode}")

        loss = output['prediction_loss']
        self.log("train_loss", 
                 loss, 
                 prog_bar=True, 
                 on_step=True, 
                 on_epoch=True, 
                 sync_dist=True,
                 batch_size=z0.shape[0])
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
                    decay=self.ema_decay
                )
            elif self.mode == "representation-learning":
                self.update_ema(
                    ema_model=self.ema_encoder,
                    current_model=self.encoder,
                    decay=self.ema_decay
                )
                self.update_ema(
                    ema_model=self.ema_decoder,
                    current_model=self.decoder,
                    decay=self.ema_decay
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
        if self.vae:
            z0 = batch["latent"]
            z0 = z0 * self.scale_factor
        else: 
            z0 = batch["x_origin"]
            z0 = rearrange(z0, "b t c h w -> (b t) c h w")

        if self.mode == "representation-learning":
            x0 = z0
            x_recon = self.gaussian_diffusion.representation_learning_ddim_sample(
                ddim_style='ddim100',
                encoder=self.ema_encoder if hasattr(self, 'ema_encoder') else self.encoder,
                decoder=self.ema_decoder if hasattr(self, 'ema_decoder') else self.decoder,
                x_0=x0,
                x_T=torch.randn_like(z0),
                disable_tqdm=True
            )
            x_recon = self.decode(x_recon)
            ssim = self.ssim(x_recon, self.decode(z0))
            loss = self.gaussian_diffusion.p_loss(x_recon, x0, loss_type="l2")
            self.log_dict(
                {"val_ssim": ssim.mean(),
                 "val_loss": loss},
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True
            )

        # For visualization (only for batch_idx == 0 on global_rank == 0)
        if batch_idx == 0 and self.global_rank == 0:
            print("Generating image...")
            z_0 = None
            if self.mode == "pretrain":
                # Use the EMA version (or the main unet) for sampling
                z_0 = self.gaussian_diffusion.regular_ddim_sample(
                    ddim_style='ddim50',
                    denoise_fn=self.ema_unet if hasattr(self, 'ema_unet') else self.unet,
                    x_T=torch.randn_like(z0),
                )
            elif self.mode == "representation-learning":
                x0 = z0
                # Use the EMA version (or the main encoder/decoder) for sampling
                z_0 = self.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style='ddim100',
                    encoder=self.ema_encoder if hasattr(self, 'ema_encoder') else self.encoder,
                    decoder=self.ema_decoder if hasattr(self, 'ema_decoder') else self.decoder,
                    x_0=x0,
                    x_T=torch.randn_like(z0),
                    disable_tqdm=True
                )

            if z_0 is not None:
                print("Logging image...")
                if self.mode == "pretrain":
                    print("skip log img for pretrain")

                elif self.mode == "representation-learning":
                    # Log the original for comparison
                    self.tb_log_image_batch_2d(
                        self.decode(x0),
                        caption="origin",
                        job="val"
                    )
                    self.tb_log_image_batch_2d(
                        self.decode(z_0),
                        caption="generate",
                        job="val"
                    )
                else:
                    raise ValueError(f"Invalid mode: {self.mode}")
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            # return ssim

    def test_step(self, batch, batch_idx):
        """
        Here we test the model reconstruction and generation capabilities.
        """
        # z0 = batch["latent"]
        # z0 = z0 * self.scale_factor
        # x0 = batch["image"]

        # subject metadata
        pidx = batch["id"].item()
        anomaly = batch["anomaly"].item()
        anomaly_type = batch["anomaly_type"][0]
        anomaly_gt_seg = batch["anomaly_gt_seg"]

        x0 = batch["x_origin"]
        x0 = rearrange(x0, "b t c h w -> (b t) c h w")
        anomaly_gt_seg = rearrange(anomaly_gt_seg, "b t c h w -> (b t) c h w")

        if self.vae is not None: 
            z0 = batch["latent"]
            z0 = z0 * self.scale_factor
        else:
            z0 = None

        if self.mode == "pretrain":
            z_T = self.gaussian_diffusion.regular_ddim_encode(
                ddim_style=self.test_ddim_style,
                denoise_fn=self.ema_unet,
                x_0=z0,
                disable_tqdm=True
            )
            # Use the EMA version (or the main unet) for sampling
            z_0 = self.gaussian_diffusion.regular_ddim_sample(
                ddim_style=self.test_ddim_style,
                denoise_fn=self.ema_unet,
                x_T=torch.randn_like(z0),
            )
            x0_gen = self.decode(z_0)
            x0_recon = self.decode(z0)
            x0_real = x0.permute(0, 1, 3, 4, 2)
            # Log 10 images in total
            if batch_idx < 10:
                self.log_image_batch(x0_gen.cpu().numpy(), caption="generated")
                self.log_image_batch(x0_recon.cpu().numpy(), caption="reconstructed")
                self.log_image_batch(x0_real.cpu().numpy(), caption="real")
            # Compute the features with pretrained model
            gen_features = self.fid_model(x0_gen)
            gen_features = torch.nn.functional.adaptive_avg_pool3d(gen_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)
            real_features = self.fid_model(x0_real)
            real_features = torch.nn.functional.adaptive_avg_pool3d(real_features, 1).squeeze(-1).squeeze(-1).squeeze(
                -1)
            recon_features = self.fid_model(x0_recon)
            recon_features = torch.nn.functional.adaptive_avg_pool3d(recon_features, 1).squeeze(-1).squeeze(-1).squeeze(
                -1)
            # Append the features to the lists
            self.real_features = torch.cat([self.real_features, real_features], dim=0)
            self.generated_features = torch.cat([self.generated_features, gen_features], dim=0)
            self.reconstructed_features = torch.cat([self.reconstructed_features, recon_features], dim=0)
            # Now get reconstruction metrics with regular DDIM
            z_infer_ddim = self.gaussian_diffusion.regular_ddim_sample(
                ddim_style=self.test_ddim_style,
                denoise_fn=self.ema_unet,
                x_T=z_T,
            )
            x_infer_ddim = self.decode(z_infer_ddim)
            mse = torch.nn.functional.mse_loss(x_infer_ddim, x0_real)
            ssim = self.ssim(x_infer_ddim, x0_real)
            mssim = self.mssim(x_infer_ddim, x0_real)
            lpips = self.lpips(x_infer_ddim, x0_real)
            if batch_idx < 10:
                self.log_image_batch(x_infer_ddim.cpu().numpy(), caption="inferred_ddim")
            self.log("test_ddim_mse", mse.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_ddim_ssim", ssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_ddim_mssim", mssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("test_ddim_lpips", lpips.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == "representation-learning":
            if self.vae is None: 
                x0_hat = None
                x0_semantic = None

                if self.use_xT_inferred:
                    print(f"Encoding the image...[batch_idx: {batch_idx}]")
                    x_T_inferred = self.gaussian_diffusion.representation_learning_ddim_encode(
                        ddim_style=self.test_ddim_style,
                        encoder=self.ema_encoder,
                        decoder=self.ema_decoder,
                        x_0=x0
                    )
                    print(f"Decoding with LDAE...[batch_idx: {batch_idx}]")
                    x0_hat = self.gaussian_diffusion.representation_learning_ddim_sample(
                        ddim_style=self.test_ddim_style,
                        encoder=self.ema_encoder,
                        decoder=self.ema_decoder,
                        x_0=x0,
                        x_T=x_T_inferred,
                        disable_tqdm=True
                    )

                print(f"Semantic sampling with LDAE...[batch_idx: {batch_idx}]")
                x_T_noise = self.gaussian_diffusion.q_sample(x0, self.test_noise_level)
                x0_semantic = self.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style=self.test_ddim_style,
                    encoder=self.ema_encoder,
                    decoder=self.ema_decoder,
                    x_0=x0,
                    x_T=x_T_noise,
                    start_t=self.test_noise_level,
                    disable_tqdm=True
                )
            elif self.vae is not None: 

                # Use the EMA version (or the main encoder/decoder) for sampling
                print(f"Encoding the image...[batch_idx: {batch_idx}]")
                z_T = self.gaussian_diffusion.latent_representation_learning_ddim_encode(
                    ddim_style=self.test_ddim_style,
                    encoder=self.ema_encoder,
                    decoder=self.ema_decoder,
                    x_0=x0,
                    z_0=z0,
                    disable_tqdm=True
                )
                print(f"Decoding with LDAE...[batch_idx: {batch_idx}]")
                z0_hat = self.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style=self.test_ddim_style,
                    encoder=self.ema_encoder,
                    decoder=self.ema_decoder,
                    x_0=x0,
                    x_T=z_T,
                    disable_tqdm=True
                )

                print(f"Semantic sampling with LDAE...[batch_idx: {batch_idx}]")
                z0_semantic = self.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style=self.test_ddim_style,
                    encoder=self.ema_encoder,
                    decoder=self.ema_decoder,
                    x_0=x0,
                    x_T=torch.randn_like(z0),
                    disable_tqdm=True
                )
                x0_semantic = self.decode(z0_semantic)
                x0_hat = self.decode(z0_hat)
                x_T_inferred = self.decode(z_T)
                x0_recon = self.decode(z0)

            if self.spartial_dim == 3: 
                x0_real = x0.permute(0, 1, 3, 4, 2)  # [B, C, H, W, D]
            elif self.spartial_dim == 2:
                x0_real = x0

            # Store the result
            x0_real_np = x0_real.cpu().numpy()  # [B, C, H, W]
            x0_semantic_np = x0_semantic.cpu().numpy()
            if self.use_xT_inferred:
                x0_hat_np = x0_hat.cpu().numpy()  # [B, C, H, W]
            else:
                x0_hat_np = None
            self.ano_gts.append(anomaly_gt_seg.cpu().numpy())
            self.recons_idx.append(batch["id"].cpu().numpy())
            self.recons_np.append(x0_hat_np)
            self.recons_semantic_np.append(x0_semantic_np)
            self.origins.append(x0_real_np)

            # Log infos to eval_dict
            self.eval_dict["IDs"].append(pidx)
            self.eval_dict["labelPerVol"].append(anomaly)
            self.eval_dict["anomalyType"].append(anomaly_type)

            # if batch_idx < 20:  # only use for WandB because we can store multiple images inside 1 tag
            if batch_idx % self.log_img_every_epoch == 0: 
                pidx = batch["id"].item()
                save_idx = batch_idx // self.log_img_every_epoch  # so that every test run will store img with the same save_idx (easier to browse)
                if self.spartial_dim == 3: 
                    pass  # 3D MRI scan

                elif self.spartial_dim == 2: 
                    job = self.trainer.state.stage.value
                    self.tb_log_image_batch_2d(x0_semantic, job=job, caption="recon_semantic", idx=pidx)
                    self.tb_log_image_batch_2d(x0_real, job=job, caption="real", idx=pidx)
                    if self.vae: 
                        self.tb_log_image_batch_2d(x0_recon, job=job, caption="recon_vaekl", idx=pidx)
                    if self.use_xT_inferred:
                        self.tb_log_image_batch_2d(x_T_inferred, job=job, caption="xT_inferred", idx=pidx)
                        self.tb_log_image_batch_2d(x0_hat, job=job, caption="recon_inferred", idx=pidx)

                    # Log plot of comparison
                    tb_log_plot_heatmap(
                        self,
                        idx=save_idx,
                        caption="pixel_score_semantic",
                        x_origin=x0_real,
                        x_recons=x0_semantic,
                    )

                    if self.use_xT_inferred:
                        tb_log_plot_comparison_2d(
                            self,
                            idx=save_idx, 
                            caption="comparison",
                            x_origin=x0_real,
                            x_recons=x0_hat, 
                            x_recons_semantic=x0_semantic,
                        )

                        tb_log_plot_heatmap(
                            self,
                            idx=save_idx,
                            caption="pixel_score_infer",
                            x_origin=x0_real,
                            x_recons=x0_hat,
                        )

            # Compute reconstruction metrics of the autoencoder
            if self.vae: 
                aekl_mse = torch.nn.functional.mse_loss(x0_recon, x0_real)
                aekl_ssim = self.ssim(x0_recon, x0_real)
                aekl_mssim = self.mssim(x0_recon, x0_real)
                aekl_lpips = self.lpips(x0_recon, x0_real)
                self.log("test_mse_aekl", aekl_mse.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log("test_ssim_aekl", aekl_ssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log("test_mssim_aekl", aekl_mssim.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log("test_lpips_aekl", aekl_lpips.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def on_test_epoch_end(self):
        if self.mode == "pretrain":
            fid = FIDMetric()
            fid_original = fid(self.generated_features, self.real_features)
            fid_reconstructed = fid(self.generated_features, self.reconstructed_features)
            self.log("test_fid_original", fid_original, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_fid_reconstructed", fid_reconstructed, prog_bar=True, on_step=False, on_epoch=True,
                     sync_dist=True)

        # Collect reconstruction result and save
        def save_np(np_path, np_name, x):
            """
            Save numpy array in np_path with a given np_name
            """
            np.save(
                os.path.join(np_path, np_name),
                x
            )

        self.recons_semantic_np = np.stack(self.recons_semantic_np)  # [N, B, C, H, W]
        self.recons_idx = np.stack(self.recons_idx)
        self.origins = np.stack(self.origins)  # [N, B, C, H, W])
        self.ano_gts = np.stack(self.ano_gts)

        save_np(self.result_path, "recons_semantic", self.recons_semantic_np)
        save_np(self.result_path, "recons_idx", self.recons_idx)
        if self.use_xT_inferred:
            self.recons_np = np.stack(self.recons_np)  # [N, B, C, H, W]
            save_np(self.result_path, "recons", self.recons_np)
        print("Save reconstruction numpy.")

        # Calculate test metrics and save eval_dict
        self.eval_dict = test_metrics(
            self,
            eval_dict=self.eval_dict,
            x_orgs=self.origins,
            x_recons=self.recons_semantic_np,
            x_ano_gts=self.ano_gts,
            recon_type="semantic",
            save_dict=True,
            save_path=self.result_dict_path,
        )

        if self.use_xT_inferred:
            self.eval_dict = test_metrics(
                self,
                eval_dict=self.eval_dict,
                x_orgs=self.origins,
                x_recons=self.recons_np,
                x_ano_gts=self.ano_gts,
                recon_type="infer",
                save_dict=True,
                save_path=self.result_dict_path,
            )

    def tb_display_generation(self, step, images, tag="Generated"):
        """
        Display generation result in TensorBoard during training
        """
        images = images.detach().cpu().numpy()[:, 0, :, :]
        images = rearrange(images, "b h w -> h (b w)")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 3))
        plt.style.use('dark_background')
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

        grid = torchvision.utils.make_grid(images, nrow=slices_per_patient, normalize=True, scale_each=True)
        if job == "fit":
            tag = f"{job}_ep{self.current_epoch:03d}/{caption}"
        elif job == "test":
            # tag = f"{job}_{idx}/{caption}"
            tag = f"test_{idx:03}/{caption}"
        elif job == "val":
            tag = f"val_ep{self.current_epoch:03d}/{caption}"
        writer.add_image(tag, grid, self.global_step)

    def configure_optimizers(self):
        """
        Return optimizer for either the UNet or (encoder + ShiftUNet).
        """
        if self.mode == "pretrain":
            return torch.optim.Adam([
                {"params": self.unet.parameters()},
            ], lr=self.lr, eps=1.0e-08, betas=(0.9, 0.999), weight_decay=0.0)
        elif self.mode == "representation-learning":
            return torch.optim.Adam([
                {"params": self.encoder.parameters()},
                {"params": self.decoder.label_emb.parameters()},
                {"params": self.decoder.shift_middle_block.parameters()},
                {"params": self.decoder.shift_output_blocks.parameters()},
                {"params": self.decoder.shift_out.parameters()},
            ], lr=self.lr, eps=1.0e-08, betas=(0.9, 0.999), weight_decay=0.0)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
