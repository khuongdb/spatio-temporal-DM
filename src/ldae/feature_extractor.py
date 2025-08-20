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
import random

import lightning as L
import numpy as np
import torch
from einops import rearrange
from generative.losses import PerceptualLoss
from generative.metrics import FIDMetric
from generative.networks.nets import AutoencoderKL

from monai.metrics.regression import MultiScaleSSIMMetric, PSNRMetric, SSIMMetric
from src.ldae import CondDDPM
from src.ldae.diffusion import GaussianDiffusion
from src.ldae.nets import (
    AttentionSemanticEncoder,
    CondUNet,
    FeatureExtractor,
    SemanticEncoder,
    ShiftUNet,
    UNet,
)
from src.utils import plot_comparison_starmen


def fe_loss_fnc(x, y, x_frozen, y_frozen, dl_lambda):
    """
    https://github.com/arimousa/DDAD/blob/main/feature_extractor.py
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss1 = 0
    loss2 = 0
    loss3 = 0
    for item in range(len(x)):
        loss1 += torch.mean(1 - cos_loss(x[item].view(x[item].shape[0], -1), 
                                         y[item].view(y[item].shape[0], -1)))
        loss2 += torch.mean(1 - cos_loss(y[item].view(y[item].shape[0], -1), 
                                         y_frozen[item].view(y_frozen[item].shape[0], -1),)) * dl_lambda
        loss3 += torch.mean(1 - cos_loss(x[item].view(x[item].shape[0], -1), 
                                         x_frozen[item].view(x_frozen[item].shape[0], -1),)) * dl_lambda
        
    loss = loss1 + loss2 + loss3
    return loss


class FeatureExtractorLitmodel(L.LightningModule):
    def __init__(
        self,
        spartial_dim: int = 2,
        ckpt_path: str = None,
        dl_lambda: float = 0.1,
        test_ddim_style: str = "ddim250",
        train_noise_level: list = [100, 250, 500],
        fe_features: list = ["level2", "level3"],
        lr:  float = 1.0e-4,
        mode: str = "fe-train",
        timesteps_args: dict = None,
    ):
        """
        Initialize Feature Extractor LightningModule.

        """
        super().__init__()
        self.dl_lambda = dl_lambda
        self.lr = lr
        self.test_ddim_style = test_ddim_style
        self.timesteps_args = timesteps_args
        self.train_noise_level = train_noise_level
        # self.example_input_array = torch.Tensor(2, 10, 1, 64, 64)  # [B, T, C, H, W]
        if ckpt_path is not None:
            self.litmodel = CondDDPM.load_from_checkpoint(
                ckpt_path, map_location=self.device
            )
        else:
            raise ValueError("Checkpoint path is needed to initialize FE model.")
        self.encoder = self.litmodel.ema_encoder
        self.unet = self.litmodel.ema_decoder

        # Diffusion helper
        self.gaussian_diffusion = GaussianDiffusion(
            timesteps_args, device=self.device
        )

        # initialize FE model
        self.fe = FeatureExtractor(encoder=self.encoder)

        self.litmodel.cpu()
        del self.litmodel
        print("Delete litmodel. Only keep UNet and Encoder")

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

        self.encoder.eval()
        self.unet.eval()
        print("Encoder and decoder set to eval mode.")

        self.fe.train()
        print("FeatureExtractor set to training mode.")

        # Freeze UNet
        self.unet.requires_grad_(requires_grad=False)
        print("Freeze UNet model")

    def on_test_start(self) -> None:
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        Standard training loop:
          - Either pretraining the UNet in latent space
          - Or learning a latent representation (encoder + ShiftUNet)
        """

        x0 = batch["x_origin"]
        x0 = rearrange(x0, "b t c h w -> (b t) c h w")

        # select a noise level
        noise_level = random.choice(self.train_noise_level)
        x_T = self.gaussian_diffusion.q_sample(x0, noise_level)

        # Calculate reconstruction x0_hat
        x0_hat = self.gaussian_diffusion.representation_learning_diffae_sample(
            ddim_style=self.test_ddim_style,
            encoder=self.encoder,
            unet=self.unet,
            x_0=x0,
            x_T=x_T,
            disable_tqdm=True,
        )

        # Compute similarity Loss from features
        x0_fe_frozen, x0_fe = self.fe(x0)
        xhat_fe_frozen, xhat_fe = self.fe(x0_hat)

        loss = fe_loss_fnc(xhat_fe, x0_fe, x0_fe_frozen, xhat_fe_frozen, self.dl_lambda)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Here we run inference
        """
        x0 = batch["x_origin"]
        x0 = rearrange(x0, "b t c h w -> (b t) c h w")

        # Calculate reconstruction x0_hat
        # we choose noise level 250
        xT = self.gaussian_diffusion.q_sample(x0, t=250)
        x0_hat = self.gaussian_diffusion.representation_learning_diffae_sample(
            ddim_style=self.test_ddim_style,
            encoder=self.encoder,
            unet=self.unet,
            x_0=x0,
            x_T=xT,
            disable_tqdm=True,
        )

        # Compute similarity Loss from features
        x0_fe_frozen, x0_fe = self.fe(x0)
        xhat_fe_frozen, xhat_fe = self.fe(x0_hat)

        loss = fe_loss_fnc(xhat_fe, x0_fe, x0_fe_frozen, xhat_fe_frozen, self.dl_lambda)

        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test the model 
        """
        return NotImplementedError

    def on_test_epoch_end(self):
        return NotImplementedError

    def configure_optimizers(self):
        """
        Return optimizer
        """
        return torch.optim.Adam([
            {"params": self.fe.parameters()},
        ], lr=self.lr, eps=1.0e-08, betas=(0.9, 0.999), weight_decay=0.0)
