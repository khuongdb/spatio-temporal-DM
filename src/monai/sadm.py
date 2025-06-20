import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from src.data import StarmenDataset
from src.monai.vivit import ViT
from src.utils import convert_sec_hms, get_logger
from src.sadm.utils import plot_multi_imgs, save_ckp


class SADM(nn.Module):
    """
    Sequence aware diffusion model based on https://arxiv.org/abs/2212.08228
    """

    def __init__(
        self,
        spatial_dims=2,
        image_size: int = 64,
        frames: int = 9,
        image_patch_size: int = 16,
        frame_patch_size: int = 1,
        vit_dim: int = 512,
        vit_spartial_depth: int = 6,
        vit_temporal_depth: int = 6,
        vit_head: int = 8,
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels=(64, 64, 64),
        # norm_num_groups=16,
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=8,
        cross_attention_dim=8,
    ):
        super(SADM, self).__init__()
        self.vivit = ViT(
            image_size=image_size,  # image size
            frames=frames,  # number of frames
            image_patch_size=image_patch_size,  # image patch size
            frame_patch_size=frame_patch_size,  # frame patch size
            channels=in_channels,
            out_channels=cross_attention_dim,
            dim=cross_attention_dim,
            spatial_depth=vit_spartial_depth,  # depth of the spatial transformer
            temporal_depth=vit_temporal_depth,  # depth of the temporal transformer
            heads=vit_head,
            mlp_dim=512,
            variant="factorized_encoder",  # or 'factorized_self_attention'
            reduce_dim=True,  # perform global pooling or exercise cls token.
            out_upsample=False, 
        )
        self.ddpm = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            with_conditioning=True,
            cross_attention_dim=cross_attention_dim,
        )

        self.cross_attn_dim = cross_attention_dim

    def forward(self, x, x_prev, drop_prob, train=True, inferer=None):

        # 1. Generate the conditional context
        ctx = self.vivit(x_prev)
        ctx = rearrange(ctx, "b d -> b 1 d")
        # ctx = rearrange(ctx, "b c h w -> b (h w) c")

        # 2. Drop context during training with probability drop_prob
        # ctx_mask = 1: drop context.
        # ctx_mask = 0: keep context.
        if train and (drop_prob is not None):
            ctx_mask = torch.bernoulli(torch.zeros(ctx.shape[0]) + drop_prob).to(
                x.device
            )
            ctx_mask = ctx_mask[:, None, None]

            # Flip context_mask 0 <-> 1 and apply the mask
            ctx = ctx * (-1 * (1 - ctx_mask))            

        # Generate random noise
        noise = torch.randn_like(x).to(x.device)

        # 3. Create timesteps
        timesteps = torch.randint(
            0,
            inferer.scheduler.num_train_timesteps,
            (x.shape[0],),
            device=x.device,
        ).long()

        # 4. Get model prediction
        noise_pred = inferer(
            inputs=x,
            diffusion_model=self.ddpm,
            noise=noise,
            timesteps=timesteps,
            condition=ctx,
            mode="crossattn",
        )

        # 5. Calculate L2 loss
        loss = F.mse_loss(noise, noise_pred)

        return loss
    

    def forward_ddpm_only(self, x, x_prev, drop_prob, train=True, inferer=None):

        """
        Train the ddpm module only with zero condition signal 
        """

        # 1. Generate the conditional context (zero tensor)
        b, c, h, w = x.shape
        ctx = torch.zeros(b, self.cross_attn_dim, h, w)
        # ctx = self.vivit(x_prev)
        ctx = rearrange(ctx, "b c h w -> b (h w) c")

        # 2. Generate random noise
        noise = torch.randn_like(x).to(x.device)

        # 3. Create timesteps
        timesteps = torch.randint(
            0,
            inferer.scheduler.num_train_timesteps,
            (x.shape[0],),
            device=x.device,
        ).long()

        # 4. Get model prediction
        noise_pred = inferer(
            inputs=x,
            diffusion_model=self.ddpm,
            noise=noise,
            timesteps=timesteps,
            condition=ctx,
            mode="crossattn",
        )

        # 5. Calculate L2 loss
        loss = F.mse_loss(noise, noise_pred)

        return loss


    def sample(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor,
        scheduler: None,
        guide_w=2.,
        ddim_eta=0.0,
        verbose=False,
        save_intermediates=False,
        intermediate_steps=10,
        context_mode="crossattn",
    ):
        """
        Sample from trained SADM models
        We follow the inference algorithm in https://arxiv.org/abs/2207.12598
        We concat two versions of the dataset: 1 with context_mask = 0 and 1 with context_mask = 1 (unconditional)
        Mix the output with the guidance scale w (larger w means more guidance)

        noise_pred = (1 + guidance) * noise_cond - guidance * nois_uncond

        Sample codes follows tutorial from MONAI at:
        https://github.com/Project-MONAI/tutorials/blob/main/generation/classifier_free_guidance/2d_ddpm_classifier_free_guidance_tutorial.ipynb
        """

        if verbose:
            pbar = tqdm(scheduler.timesteps)
        else:
            pbar = iter(scheduler.timesteps)

        intermediates = []

        for t in pbar:
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True
            ):
                with torch.no_grad():

                    # 1. Generate the conditional context
                    ctx = self.vivit(x_prev)
                    und_ctx = torch.zeros_like(ctx)
                    ctx_input = torch.cat([und_ctx, ctx], dim=0).to(x.device)
                    ctx_input = rearrange(ctx_input, "b d -> b 1 d")
                    # ctx_input = rearrange(ctx_input, "b c h w -> b (h w) c")

                    # 2. Generate random noise and repeat 2 times for uncondition and condition
                    # shape [2B, ...]
                    noise = torch.randn_like(x).to(x.device)
                    noise_input = torch.cat([noise] * 2)

                    # 3. Generate time step
                    timesteps = torch.Tensor((t,)).to(noise_input.device)

                    # 3. Calculate noise_pred from DDPM
                    if context_mode == "concat":
                        model_input = torch.cat([noise_input, ctx_input], dim=1)
                        model_output = self.ddpm(
                            model_input, timesteps=timesteps, context=None
                        )
                    else:
                        model_input = noise_input
                        model_output = self.ddpm(
                            model_input,
                            timesteps=timesteps,
                            context=ctx_input,
                        )

                    # 4. Split predictions and compute weighted noise
                    noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                    noise_pred = (1 + guide_w) * noise_pred_cond - guide_w * noise_pred_uncond

            if isinstance(scheduler, DDPMScheduler):
                noise, _ = scheduler.step(noise_pred, t, noise)
            elif isinstance(scheduler, DDIMScheduler):
                noise, _ = scheduler.step(noise_pred, t, noise, eta=ddim_eta)

            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(noise)

        if save_intermediates:
            return noise_pred, intermediates
        else:
            return noise_pred


def train(
    workdir=None,
    train_loader=None,
    val_loader=None,
    device="cpu",
    model: nn.Module = None,
    logger=None,
    num_train_timesteps=1000,
    learning_rate=2.5e-5,
    n_epochs=100,
    val_interval=10,
    sample_interval=50,
    drop_prob=0.15,
    ddpm_only=True,  # Only train the DDPM with zero condition signal
    autoresume=False,
    checkpoint_path=None
):
    """
    Train SADM model
    """

    if not logger:
        logger = get_logger("train", workdir=workdir, mode="w")

    model.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    inferer = DiffusionInferer(scheduler)

    logger.info("Model:")
    logger.info(model)

    logger.info("optimizer")
    logger.info(optimizer)
    logger.info("scheduler:")
    logger.info(scheduler)

    ###
    # Train the model
    ###

    epoch_loss_list = []
    val_epoch_loss_list = []

    logger.info(f"Nb epochs: {n_epochs}")

    scaler = GradScaler(device=device)
    total_start = time.time()

    # logger.info("Start training: ")

    # make workdir folder
    ckpt_dir = f"{workdir}/ckpt"
    sample_dir = f"{workdir}/eval_samples"
    for dir in [ckpt_dir, sample_dir]:
        os.makedirs(dir, exist_ok=True)

    # Autoresume from previous checkpoint
    start_ep = 0
    if autoresume:
        if checkpoint_path is not None:
            ckpt_path = checkpoint_path
        else:
            ckpt_path = os.path.join(ckpt_dir, "latest.pth")

        if not os.path.exists(ckpt_path):
            logger.info(f"Checkpoint does not exist at {ckpt_path}. Train from start.")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            if optimizer and ckpt.get("optimizer_state", None): 
                optimizer.load_state_dict(ckpt["optimizer_state"])
            logger.info(f"Load checkpoint from {ckpt_path}")
        
        if ckpt.get("epoch", None):
            start_ep = ckpt["epoch"] + 1
            logger.info(f"Resume training from epoch {start_ep}")
        else: 
            start_ep = 0
        

    # DDIMScheduler for sampling during training
    sample_scheduler = DDIMScheduler(num_train_timesteps=1000)
    sample_scheduler.set_timesteps(num_inference_steps=500)

    for epoch in range(start_ep, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            x = batch["x"].to(device)
            x_prev = batch["x_prev"].to(device)

            # to align with ViVit input: [B C T H W]
            x_prev = rearrange(x_prev, "b t c h w -> b c t h w")

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=True):
                # Get model prediction
                if ddpm_only:
                    loss = model.forward_ddpm_only(x=x, x_prev=x_prev, drop_prob=drop_prob, inferer=inferer)
                else:
                    loss = model(x=x, x_prev=x_prev, drop_prob=drop_prob, inferer=inferer)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        logger.info(f"Epoch: {epoch} - loss: {epoch_loss / (step + 1)}")
        epoch_loss_list.append(epoch_loss / (step + 1))

        # Save checkpoint for current epoch
        # save_path = "workdir/monai_ddpm/ckpt/latest.pt"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (
                    scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
                ),
                "epoch": epoch,
            },
            os.path.join(ckpt_dir, "latest.pth"),
        )

        # Evaluation during training
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                x_val = batch["x"].to(device)
                x_prev_val = batch["x_prev"].to(device)
                x_prev_val = rearrange(x_prev_val, "b t c h w -> b c t h w")

                with torch.no_grad():
                    with autocast(device_type=device.type, enabled=True):
                        val_loss = model(
                            x=x_val,
                            x_prev=x_prev_val,
                            drop_prob=drop_prob,
                            inferer=inferer,
                        )
                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

                # Sampling during training
                if step == len(val_loader) - 1: 
                    if (epoch + 1) % sample_interval == 0:
                        idx = batch["id"][0]
                        target_idx = batch["target_idx"][0]
                        mask_smpl = batch["mask"][0]
                        x_prev_smpl = rearrange(x_prev_val[0], "c t h w -> 1 c t h w")
                        ctx_smpl = model.vivit(x_prev_smpl)
                        ctx_smpl = rearrange(ctx_smpl, "b d -> b 1 d")
                        b, c, h, w = x_val.shape
                        noise_smpl = torch.randn(1, c, h, w).to(device)
                        with autocast(device_type=device.type, enabled=True):
                            img_smpl = inferer.sample(
                                input_noise=noise_smpl, 
                                diffusion_model=model.ddpm,
                                scheduler=sample_scheduler,
                                conditioning=ctx_smpl,
                                verbose=False
                            )
                        img_smpl = img_smpl.detach().cpu().numpy().squeeze()
                        save_path = os.path.join(sample_dir, f"sample_ep{epoch:03}_id{idx:03}.png")

                        plt.imshow(img_smpl, vmin=0, vmax=1, cmap="gray")
                        plt.tight_layout()
                        plt.axis("off")
                        plt.title(f"Sample epoch: {epoch+1:03} - idx: {idx:03} - target_idx: {target_idx}\nmask: {mask_smpl}")
                        plt.savefig(save_path, bbox_inches="tight")

            logger.info(f"Epoch: {epoch} - val_loss: {val_epoch_loss / (step + 1)}")
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))


    total_time = time.time() - total_start
    h, m, s = convert_sec_hms(total_time)
    logger.info(f"Train completed, total time: {h:02}:{m:02}:{s}.")


def inference(
    model: nn.Module = None,
    test_loader=None,
    ckpt_path=None,
    device="cpu",
    num_inference_steps=1000,
    verbose=True,
    workdir="workdir",
    logger=None,
    guide_w=1.0,
    ddim_eta=0.0,
    save_numpy=False,
    save_img=True,
):
    """
    Classifier-free guided diffusion model inference.
    """
    if not logger:
        logger = get_logger("INFERENCE", workdir=workdir)

    # Load model checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    logger.info(f"Load checkpoint from {ckpt_path}")
    logger.info(model)

    # Create outdir folders
    if save_numpy:
        np_dir = os.path.join(workdir, "sample_numpy")
        os.makedirs(np_dir, exist_ok=True)
    if save_img:
        img_dir = os.path.join(workdir, "sample_images")
        os.makedirs(img_dir, exist_ok=True)



    # scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    if verbose:
        pbar = tqdm(test_loader)
    else:
        pbar = iter(test_loader)

    for batch in pbar:
        idx = batch["id"][0]

        # x_origin = batch["x_origin"].to(device)
        # mask = batch["mask"].to(device)

        # Use batch = 1 for toy example
        x_origin = batch["x_origin"][0].to(device)
        mask = batch["mask"][0].to(device)

        x_gen = torch.empty_like(x_origin)
        # Assume that we always have the first item.
        x_gen[0] = x_origin[0]

        logger.info(f"idx: {idx} - mask: {mask}")

        for i in range(1, x_origin.shape[0]):
            if mask[i] == 1:
                x_gen[i] = x_origin[i]
            else:
                # generate new image using autoagressive model
                mask_i = mask.clone()
                mask_i[:i] = 1
                mask_i[i:] = 0
                x_prev_i = x_gen[:-1] * mask_i[:-1, None, None, None]
                x_prev_i = rearrange(x_prev_i, "t c h w -> 1 c t h w")
                x_gen_i = model.sample(
                    x=rearrange(x_origin[i], "c h w -> 1 c h w"),
                    x_prev=x_prev_i,
                    scheduler=scheduler,
                    guide_w=guide_w,
                    ddim_eta=ddim_eta
                )

                # Store new generated image
                x_gen[i] = x_gen_i.squeeze(0)

        # Save .numpy result
        if save_numpy:
            np_path = os.path.join(np_dir, f"x_gen_{idx:03}.npy")
            np.save(np_path, x_gen.cpu().numpy())
            # x_masked_np = x_origin * rearrange(mask, "b -> b 1 1 1")
            # x_masked_np = x_masked_np.cpu().detach().numpy()
            # np.save(os.path.join(np_dir, f"x_masked_{idx:03}.npy"), x_masked_np)
            logger.info(f"Saved generated sample {idx:03} to {np_path}")

        if save_img:
            img_path = os.path.join(img_dir, f"x_gen_{idx:03}.png")
            x_origin_np = x_origin.squeeze()
            x_masked_np = x_origin * rearrange(mask, "t -> t 1 1 1")
            x_masked_np = x_masked_np.squeeze()
            x_gen_np = x_gen.squeeze()
            error = x_origin_np - x_gen_np

            imgs = [
                x_origin_np,
                x_masked_np,
                x_gen_np,
                error,
            ]


            labels = [
                "original",
                "masked_input",
                rf"reconstruct\n$\eta={ddim_eta}$\n$w_{{\mathrm{{guide}}}}={guide_w}$",
                "error"
            ]

            plot_multi_imgs(
                imgs=imgs,
                labels=labels,
                save=True,
                save_path=img_path,
                opt={"hspace": 0.05, "cmap": "viridis"},
            )
            logger.info(f"Saved images {idx:03} to {img_path}")


def main(args=None):

    exp_type = args.job_type
    assert exp_type in ["train", "inference"], f"{exp_type} is not implemented"

    DATA_DIR = "data/starmen/output_random_noacc"

    workdir = args.workdir
    experiment = args.experiment
    if workdir is None:
        workdir = f"workdir/{experiment}"

    os.makedirs(workdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger = get_logger("SADM", workdir=workdir)

    # Define SADM model
    model = SADM(
        spatial_dims=2,
        image_size=64,
        frames=9,
        image_patch_size=16,
        frame_patch_size=1,
        vit_dim=512,
        vit_spartial_depth=6,
        vit_temporal_depth=6,
        vit_head=8,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        # norm_num_groups=16,
        attention_levels=(False, False, True),
        num_res_blocks=1,
        num_head_channels=8,
        cross_attention_dim=32,
    )

    if exp_type == "train":
        train_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="train",
            nb_subject=None,
            save_data=False,
            workdir=workdir,
            mask_prob=0.5, 
        )
        train_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=1)

        val_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="val",
            nb_subject=None,
            save_data=False,
            workdir=workdir,
            mask_prob=0.5, 
        )
        val_loader = DataLoader(val_ds, batch_size=10, shuffle=True, num_workers=1)

        train(
            workdir=workdir,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model=model,
            # logger=logger,
            num_train_timesteps=1,
            learning_rate=1.5e-5,
            n_epochs=1000,
            val_interval=10,
            sample_interval=50,
            drop_prob=None,
            ddpm_only=False,
            autoresume=args.autoresume
        )

    else:
        # Only use batch of 1 for toy example
        test_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="test",
            nb_subject=1,
            save_data=False,
            workdir=workdir,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)

        inference(
            model=model,
            test_loader=test_loader,
            ckpt_path=os.path.join(workdir, "ckpt/latest.pth"),
            device=device,
            num_inference_steps=500,
            verbose=True,
            workdir=workdir,
            guide_w=5.0,
            ddim_eta=0.0,
            save_img=True,
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job_type",
        type=str,
        default="train",
        help="Type of job: 'train' or 'inference'",
    )

    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="working directory to store checkpoints, samples, logs.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="sadm",
        help="Name of experience. If workdir is not defined, new directory workdir/<experiment> will be created. ",
    )

    parser.add_argument(
        "--autoresume",
        action="store_true",
        help="Autoresume training from checkpoint. Checkpoint will be located at <workdir>/ckpt/latest.pt",
    )


    args = parser.parse_args()

    set_determinism(42)
    main(args=args)
