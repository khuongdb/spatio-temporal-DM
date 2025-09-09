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
import pandas as pd
import torch
from einops import rearrange
from generative.metrics import FIDMetric
from generative.networks.nets import AutoencoderKL
from leaspy.algo import AlgorithmSettings
from leaspy.io.data import Data, Dataset
from leaspy.models import LinearModel, LogisticModel
from torch.utils.data import DataLoader

from monai.losses import PerceptualLoss
from monai.metrics.regression import MultiScaleSSIMMetric, PSNRMetric, SSIMMetric
from src.ldae.diffusion import GaussianDiffusion, LongitudinalDiffusion
from src.ldae.nets import (
    CondUNet,
    FeatureExtractor,
    SemanticEncoder,
)

from src.utils.metrics import get_eval_dictionary, test_metrics
from src.utils.visualization import plot_comparison_starmen


class CondDDPM(L.LightningModule):
    def __init__(
        self,
        #  fe_net: FeatureExtractorLitmodel,
        spartial_dim: int = 2,
        enc_args: dict = None,
        unet_args: dict = None,
        tadm_args: dict = None, 
        vae_args: dict = None,
        vae_path: str = None,
        timesteps_args: dict = None,
        leaspy_lambda: float = 100.,
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
        Initialize Conditional DDPMs LightningModule.

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
        self.leaspy_lambda = leaspy_lambda

        # Set the metrics
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
        self.unet = None
        self.decoder = None
        self.vae = None
        self.fe = None

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
            self.unet = CondUNet(**unet_args)

            # Create EMA copy of UNet (disable grad, so it won't update by backprop)
            self.ema_unet = copy.deepcopy(self.unet).eval()
            for p in self.ema_unet.parameters():
                p.requires_grad = False

        # ------------------- Representation-Learning Models -------------------
        elif mode == "representation-learning":
            # # Load the pretrained denoising model
            # weights = None
            # try:
            #     data = torch.load(pretrained_path, map_location=self.device)
            #     weights = {key.replace('ema_unet.', ''): value for key, value in data['state_dict'].items()}
            # except FileNotFoundError:
            #     print(f"Pretrained model not found at {pretrained_path}")

            # CondUNet decoder
            self.decoder = CondUNet(
                latent_dim=enc_args["emb_chans"],
                **unet_args,
            )
            # if weights is not None:
            #     self.decoder.load_state_dict(weights, strict=False)
            #     print("Successfully loaded the pretrained decoder model.")
            # else:
            #     print("No weights loaded for the decoder.")

            # Semantic encoder
            self.encoder = SemanticEncoder(**enc_args)

            # Create EMA copies (disable grad, so they won't update by backprop)
            self.ema_encoder = copy.deepcopy(self.encoder).eval()
            for p in self.ema_encoder.parameters():
                p.requires_grad = False

            self.ema_decoder = copy.deepcopy(self.decoder).eval()
            for p in self.ema_decoder.parameters():
                p.requires_grad = False

            # FeatureExtractor network
            # dummy variable to register fe
            self.fe = None

        # ------------------- Longitudinal Learning Models -------------------
        elif mode == "longitudinal_learning":
            # CondUNet decoder
            self.decoder = CondUNet(
                latent_dim=enc_args["emb_chans"],
                **unet_args,
            )

            # Semantic encoder
            self.encoder = SemanticEncoder(**enc_args)

            # leaspy model for longitudinal model
            # self.model_leaspy = LogisticModel(name="logistic", source_dimension=enc_args["emb_chans"] - 1)
            self.model_leaspy = LinearModel(name="linear", source_dimension=enc_args["emb_chans"] - 1)
            self.model_leaspy.move_to_device(self.device)

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

    def load_state_dict(self, state_dict, strict=True, assign=False):
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("fe.")
        }
        return super().load_state_dict(filtered_state_dict, strict, assign)

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
            self.decoder.train()
            print("Encoder and decoder set to training mode.")
        elif self.mode == "longitudinal_learning":
            self.encoder.train()
            self.decoder.train()
            self.leaspy_dir = os.path.join(
                self.trainer.logger.save_dir, self.trainer.logger.name, "leaspy_model"
            )
            os.makedirs(self.leaspy_dir, exist_ok=True)
            self.leaspy_diffusion = LongitudinalDiffusion(device=self.device)

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

        self.ssim = SSIMMetric(
            spatial_dims=self.spartial_dim, data_range=1.0, win_size=4
        )
        # self.ssim.to(self.device)
        self.mssim = MultiScaleSSIMMetric(
            spatial_dims=self.spartial_dim, data_range=1.0, kernel_size=4
        )
        # self.mssim.to(self.device)
        self.psnr = PSNRMetric(max_val=1.0, reduction="mean")
        # self.psnr.to(self.device)
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
        self.recons_np = []
        self.recons_semantic_np = []
        self.origins = (
            []
        )  # we dont need to store the origin, this is just to calculate histogram at on_test_epoch_end
        self.ano_gts = []
        self.result_path = os.path.join(
            self.trainer.logger.save_dir, 
            self.trainer.logger.name, "results"
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
        
    @torch.no_grad()
    def train_leaspy_model(self):
        
        # Get dataset
        dataset = self.trainer.datamodule.train_dataloader().dataset
        leaspy_train_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        # Extract semantic representation
        ysem_df, ysem_data, ysem_dataset = self.leaspy_diffusion.extract_semantic_encoder_dataloader(
            dataloader=leaspy_train_loader,
            encoder=self.encoder,
        )
        ysem_dataset.move_to_device(self.device)

        # Fit leaspy model
        algo_settings = AlgorithmSettings('mcmc_saem', 
                                        seed=42, 
                                        n_iter=1,           
                                        progress_bar=False,
                                        device=self.device)

        # algo_settings.set_logs(
        #     path=os.path.join(self.leaspy_dir, "logs"), # Creates a logs file ; if existing, ask if rewrite it
        #     plot_periodicity=None,  # Saves the values to display in pdf every 50 iterations
        #     save_periodicity=None,  # Saves the values in csv files every N iterations
        #     console_print_periodicity=None,  # If = N, it display logs in the console/terminal every N iterations
        #     overwrite_logs_folder=True       # Default behaviour raise an error if the folder already exists.
        # )    

        algo_settings.set_logs(path=None)

        self.model_leaspy.fit( ysem_dataset, algorithm_settings=algo_settings)
        # print("Fitted leaspy model!")

        # # Save leaspy model
        # self.model_leaspy.save(os.path.join(self.leaspy_dir, "leaspy_model.json"))

    def on_train_epoch_start(self):
        if self.mode == "longitudinal_learning":
            self.train_leaspy_model()
        else:
            pass
    
    def train_longitudinal_one_step(self, batch, batch_idx, is_validation=False):
        """
        Train longitudinal (leaspy) with normal diffusion model. 
        """

        x0 = batch["x_origin"]
        x0 = rearrange(x0, "b t c h w -> (b t) c h w")

        # ages = batch["age"]
        # ages = rearrange(ages, "b t -> (b t)")
        # ages_np = ages.detach().cpu().numpy()

        # ids_all = []
        # p_ids = batch["id"].cpu().numpy()
        # p_ids = np.repeat(p_ids, 10)
        # ids_all.append(p_ids)
        # ids_all = np.concatenate(ids_all)

        # 1. Semantic encoder
        x0.to(self.device)
        out = self.leaspy_diffusion.extract_semantic_encoder_one_batch(
            ids=batch["id"],
            x=batch["x_origin"],
            age=batch["age"],
            mask=None,
            encoder=self.encoder,
        )
        cond = torch.concat(out["features"]).to(self.device)

        # 2. Expected position of semantic encode from leaspy
        ysem_df, ysem_data, ysem_dataset = self.leaspy_diffusion.create_leaspy_dataset(
            ids=out["ids"],
            ages=out["ages"],
            feats=out["features"],)
        ysem_dataset.move_to_device(self.device)

        # init leaspy_model for sanity check
        if is_validation:
            if not self.model_leaspy.is_initialized:
                leaspy_model_path = os.path.join(self.leaspy_dir, "leaspy_model.json")
                if not os.path.exists(leaspy_model_path):
                    print(f"No pretrained leaspy model found at {self.leaspy_dir}. Initialize new leaspy model.")
                    self.model_leaspy.initialize(ysem_dataset)
                else: 
                    with torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=False):
                        self.model_leaspy = self.model_leaspy.load(leaspy_model_path)
        
        settings_personalization = AlgorithmSettings('scipy_minimize', progress_bar=False, use_jacobian=False)
        ysem_ips = self.model_leaspy.personalize(ysem_dataset, algorithm_settings=settings_personalization)

        # timepoints_est = dict()
        # for i in set(ids_all):
        #     timepoints_est[i] = ages_np[ids_all == i]
        # cond_expect = [torch.Tensor(v) for v in cond_expect.items()]
        # cond_expect = torch.cat(cond_expect)
        cond_expect = self.model_leaspy.estimate(ysem_df.index, ysem_ips)
        cond_expect = torch.tensor(cond_expect.to_numpy(), device=self.device)

        # 3. Normal diffusion model training
        output = self.gaussian_diffusion.regular_train_one_batch(
            denoise_fn=self.decoder, x_0=x0, condition=cond
        )

        # 4. combine loss
        loss_align = torch.sum((cond_expect - cond) ** 2)
        loss_unet = output["prediction_loss"]
        loss = loss_unet + self.leaspy_lambda * loss_align
 
        out = {
            "loss_align": loss_align,
            "loss_unet": loss_unet,
            "loss": loss
        }

        if is_validation: 
            out = {f"val_{k}": v for k, v in out.items()}

        return out

    def training_step(self, batch, batch_idx):
        """
        Standard training loop:
          - Either pretraining the UNet in latent space
          - Or learning a latent representation (encoder + ShiftUNet)
        """
        x0 = batch["x_origin"]
        x0 = rearrange(x0, "b t c h w -> (b t) c h w")

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

            cond = self.encoder(x0)
            output = self.gaussian_diffusion.regular_train_one_batch(
                denoise_fn=self.decoder, x_0=x0, condition=cond
            )

            if self.vae is None:
                # we train directly on original images (PDAE)
                # x0 = z0.clone().detach().requires_grad_(z0.requires_grad)
                cond = self.encoder(z0)
                output = self.gaussian_diffusion.regular_train_one_batch(
                    denoise_fn=self.decoder, x_0=z0, condition=cond
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

            loss = output["prediction_loss"]

            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=z0.shape[0]
            )

        # ------------------- 3) Longitudinal learning -------------------
        elif self.mode == "longitudinal_learning":
            out = self.train_longitudinal_one_step(batch, batch_idx)

            loss = out["loss"]

            self.log_dict(out,
                          prog_bar=True,
                          on_step=True,
                          on_epoch=True,
                          sync_dist=True,
                          batch_size=z0.shape[0]
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
            elif self.mode == "representation-learning" or self.mode == "longitudinal_learning":
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
        if self.vae:
            z0 = batch["latent"]
            z0 = z0 * self.scale_factor
        else:
            z0 = batch["x_origin"]
            z0 = rearrange(z0, "b t c h w -> (b t) c h w")

        if self.mode == "representation-learning":
            cond = self.encoder(z0)
            output = self.gaussian_diffusion.regular_train_one_batch(
                denoise_fn=self.decoder, x_0=z0, condition=cond
            )
            if output is None:
                raise ValueError(f"Invalid mode: {self.mode}")

            loss = output["prediction_loss"]
            self.log(
                "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=z0.shape[0]
            )
        elif self.mode == "longitudinal_learning":
            out = self.train_longitudinal_one_step(batch, batch_idx, is_validation=True)

            self.log_dict(out,
                            prog_bar=True,
                            on_step=True,
                            on_epoch=True,
                            sync_dist=True,
                            batch_size=z0.shape[0]
                            )
        
        else: 
            raise ValueError(f"Invalid mode: {self.mode}")
    
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
            pass

        elif self.mode == "representation-learning":
            if self.vae is None:
                x0_hat = None
                x0_semantic = None

                if self.use_xT_inferred:
                    print(f"Encoding the image...[batch_idx: {batch_idx}]")
                    x_T_inferred = (
                        self.gaussian_diffusion.representation_learning_diffae_encode(
                            ddim_style=self.test_ddim_style,
                            encoder=self.ema_encoder,
                            unet=self.ema_decoder,
                            x_0=x0,
                            noise_level=self.test_noise_level,
                            disable_tqdm=True,
                        )
                    )
                    print(f"Decoding from stochastic subcode...[batch_idx: {batch_idx}]")
                    x0_hat = self.gaussian_diffusion.representation_learning_diffae_sample(
                        ddim_style=self.test_ddim_style,
                        encoder=self.ema_encoder,
                        unet=self.ema_decoder,
                        x_0=x0,
                        x_T=x_T_inferred,
                        start_t=self.test_noise_level,
                        disable_tqdm=True,
                    )

                print(f"Semantic sampling with LDAE...[batch_idx: {batch_idx}]")
                if self.test_noise_level < self.timesteps_args["timesteps"]:
                    x_T_noise = self.gaussian_diffusion.q_sample(x0, self.test_noise_level)
                    x0_semantic = (
                        self.gaussian_diffusion.representation_learning_diffae_sample(
                            ddim_style=self.test_ddim_style,
                            encoder=self.ema_encoder,
                            unet=self.ema_decoder,
                            x_0=x0,
                            x_T=x_T_noise,
                            start_t=self.test_noise_level,
                            disable_tqdm=True,
                        )
                    )
                else: 
                    x0_semantic = (
                        self.gaussian_diffusion.representation_learning_diffae_sample(
                            ddim_style=self.test_ddim_style,
                            encoder=self.ema_encoder,
                            unet=self.ema_decoder,
                            x_0=x0,
                            x_T=torch.randn_like(x0),
                            disable_tqdm=True,
                        )
                    )
            elif self.vae is not None:

                # Use the EMA version (or the main encoder/decoder) for sampling
                print(f"Encoding the image...[batch_idx: {batch_idx}]")
                z_T = (
                    self.gaussian_diffusion.latent_representation_learning_ddim_encode(
                        ddim_style=self.test_ddim_style,
                        encoder=self.ema_encoder,
                        decoder=self.ema_decoder,
                        x_0=x0,
                        z_0=z0,
                        disable_tqdm=True,
                    )
                )
                print(f"Decoding with LDAE...[batch_idx: {batch_idx}]")
                z0_hat = self.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style=self.test_ddim_style,
                    encoder=self.ema_encoder,
                    decoder=self.ema_decoder,
                    x_0=x0,
                    x_T=z_T,
                    disable_tqdm=True,
                )

                print(f"Semantic sampling with LDAE...[batch_idx: {batch_idx}]")
                z0_semantic = (
                    self.gaussian_diffusion.representation_learning_ddim_sample(
                        ddim_style=self.test_ddim_style,
                        encoder=self.ema_encoder,
                        decoder=self.ema_decoder,
                        x_0=x0,
                        x_T=torch.randn_like(z0),
                        disable_tqdm=True,
                    )
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
            self.ano_gts.append(anomaly_gt_seg.cpu().numpy())
            self.recons_semantic_np.append(x0_semantic_np)
            self.origins.append(x0_real_np)
            if self.use_xT_inferred:
                x0_hat_np = x0_hat.cpu().numpy()  # [B, C, H, W]
                self.recons_np.append(x0_hat_np)

            # Log infos to eval_dict
            self.eval_dict["IDs"].append(pidx)
            self.eval_dict["labelPerVol"].append(anomaly)
            self.eval_dict["anomalyType"].append(anomaly_type)

            # Compute reconstruction metrics of the autoencoder
            if self.vae:
                aekl_mse = torch.nn.functional.mse_loss(x0_recon, x0_real)
                aekl_ssim = self.ssim(x0_recon, x0_real)
                aekl_mssim = self.mssim(x0_recon, x0_real)
                aekl_lpips = self.lpips(x0_recon, x0_real)
                self.log(
                    "test_mse_aekl",
                    aekl_mse.mean(),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test_ssim_aekl",
                    aekl_ssim.mean(),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test_mssim_aekl",
                    aekl_mssim.mean(),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test_lpips_aekl",
                    aekl_lpips.mean(),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )

            # # log metrics
            # log_metrics = {
            #     "l1errMean": self.eval_dict["l1recoErrorAllMean"],
            #     "l2errMean": self.eval_dict["l2recoErrorAllMean"],
            #     "AUPRCPerVolMean": self.eval_dict["AUPRCPerVolMean"],
            #     "AUCPerVolMean": self.eval_dict["AUCPerVolMean"],
            # }
            # self.log_dict(log_metrics, prog_bar=True, on_epoch=True, on_step=True)

            # Log image to Tensorboard
            # if batch_idx < 20:  # only use for WandB because we can store multiple images inside 1 tag
            if batch_idx % self.log_img_every_epoch == 0:
                save_idx = (
                    batch_idx // self.log_img_every_epoch
                )  # so that every test run will store img with the same save_idx (easier to browse)
                if self.spartial_dim == 3:
                    pass

                elif self.spartial_dim == 2:
                    job = self.trainer.state.stage.value
                    # self.tb_log_image_batch_2d(
                    #     x0_semantic, job=job, caption="recon_semantic", idx=save_idx
                    # )
                    # self.tb_log_image_batch_2d(
                    #     x0_real, job=job, caption="real", idx=save_idx
                    # )
                    
                    if self.use_xT_inferred:
                        self.tb_log_image_batch_2d(
                            x_T_inferred, job=job, caption="xT_inferred", idx=save_idx
                        )
                        # self.tb_log_image_batch_2d(
                        #     x0_hat, job=job, caption="recon_inferred", idx=save_idx
                        # )

                    if self.vae:
                        self.tb_log_image_batch_2d(
                            x0_recon, job=job, caption="recon_vaekl", idx=save_idx
                        )

                    # Log plot of comparison
                    self.tb_log_plot_comparison_2d(
                        idx=save_idx,
                        caption="comparison",
                        x_origin=x0_real,
                        x_recons=x0_hat,
                        x_recons_semantic=x0_semantic,
                    )
                    self.tb_log_plot_heatmap(
                        idx=save_idx,
                        caption="pixel_score_semantic",
                        x_origin=x0_real,
                        x_recons=x0_semantic,
                    )

                    if self.use_xT_inferred:
                        self.tb_log_plot_heatmap(
                            idx=save_idx,
                            caption="pixel_score_infer",
                            x_origin=x0_real,
                            x_recons=x0_hat,
                        )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def on_test_epoch_end(self):
        if self.mode == "pretrain":
            fid = FIDMetric()
            fid_original = fid(self.generated_features, self.real_features)
            fid_reconstructed = fid(
                self.generated_features, self.reconstructed_features
            )
            self.log(
                "test_fid_original",
                fid_original,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test_fid_reconstructed",
                fid_reconstructed,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Collect reconstruction result and save
        def save_np(np_path, np_name, x):
            """
            Save numpy array in np_path with a given np_name
            """
            np.save(os.path.join(np_path, np_name), x)

        self.recons_semantic_np = np.stack(self.recons_semantic_np)  # [N, B, C, H, W]
        self.origins = np.stack(self.origins)  # [N, B, C, H, W])
        self.ano_gts = np.stack(self.ano_gts)  # [N, B, C, H, W]
        save_np(self.result_path, "recons_semantic", self.recons_semantic_np)

        if self.use_xT_inferred:
            self.recons_np = np.stack(self.recons_np)  # [N, B, C, H, W]
            save_np(self.result_path, "recons", self.recons_np)

        print("Save reconstruction numpy.")

        # Calculate test metrics and save eval_dict
        ## Result for x_recon from xT_infer

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
                save_dict=False,
            )


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

        if self.use_xT_inferred:
            imgs = [
                x_origin.detach().cpu().squeeze(),
                x_recons_semantic.detach().cpu().squeeze(),
                torch.abs(x_origin - x_recons_semantic).detach().cpu().squeeze(),
                x_recons.detach().cpu().squeeze(),
                torch.abs(x_origin - x_recons).detach().cpu().squeeze(),
            ]    
        else: 
            imgs = [
                x_origin.detach().cpu().squeeze(),
                x_recons_semantic.detach().cpu().squeeze(),
                torch.abs(x_origin - x_recons_semantic).detach().cpu().squeeze(),
            ]   

        labels = [
            "origin",
            "recons_semantic",
            "error_semantic",
            "recons_infer",
            "error_infer",
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
        elif self.mode == "representation-learning" or self.mode == "longitudinal_learning":
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
