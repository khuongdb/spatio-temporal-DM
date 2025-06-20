import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.monai.anomaly_map import AnomalyMap
import torchvision
from einops import rearrange
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# from monai import transforms
# from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from src.data import StarmenDataset
from src.utils import convert_sec_hms, get_logger
from src.utils.metrics import error_filter, summarize_errors
from src.utils.networks import load_checkpoint
from src.utils.visualization import plot_comparison_starmen, plot_error_histogram


class DiffAEResNet18(nn.Module):
    """
    Diffusion Autoencoder with ResNet18 as Encoder
    Based on http://arxiv.org/abs/2111.15640
    """

    def __init__(self, emb_dim=512, scheduler=None):
        super(DiffAEResNet18, self).__init__()
        self.semantic_encoder = torchvision.models.resnet18()
        self.semantic_encoder.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.semantic_encoder.fc = torch.nn.Linear(512, emb_dim)

        self.ddpm = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 64, 64),
            # norm_num_groups=16,
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=8,
            with_conditioning=True,
            cross_attention_dim=emb_dim,
        )
        if scheduler: 
            self.scheduler = scheduler
        else:
            self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.inferer = DiffusionInferer(scheduler)

        # self.ano_map = AnomalyMap()

    def forward(self, xt, x_cond, t):
        latent = self.semantic_encoder(x_cond)
        noise_pred = self.ddpm(x=xt, timesteps=t, context=latent.unsqueeze(1))
        return noise_pred, latent


    @torch.no_grad()
    def sample_from_image(
        self,
        inputs: torch.Tensor,
        noise_level: int | None = 500,
        num_inference_steps: int = 50,
        verbose: bool = False,

    ) -> torch.Tensor :
        """
        Sample to specified noise level and use this as noisy input to sample back.
        Args:
            inputs: input images, NxCxHxW[xD]
            noise_level: noising step until which noise is added before
            num_inference_steps: number of DDIM inference steps. 
            verbose: if true, prints the progression bar of the sampling process.
        """

        device = inputs.device

        # 1. Get the laten representation from semantic encoder
        latent = self.semantic_encoder(inputs)

        # 2. Generate random noise
        noise = torch.randn_like(inputs).to(device)

        # 3. Create noisy image from timestep
        t = torch.full(
            (inputs.shape[0],), noise_level, device=device
        ).long()
        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=t)

        # 4. Run reverse process to get reconstruction of image
        self.scheduler.set_timesteps(num_inference_steps)
        x_recons = self.inferer.sample(
            input_noise=noisy_image,
            diffusion_model=self.ddpm, 
            scheduler=self.scheduler, 
            save_intermediates=False,
            verbose=False,
            conditioning=latent.unsqueeze(1)
        )
        return x_recons

    # @torch.no_grad()
    # def get_autoDDPM_anomaly(
    #     self,
    #     inputs: torch.Tensor,
    #     noise_level_recon: int | None = 200,
    #     noise_level_inpaint: int | None = 50,
    #     save_intermediates: bool | None = False,
    #     verbose: bool = False,
    # ):
    #     """
    #     Generate anomaly map and calculate anomaly score based on mask, stich and resample.
    #     Modify from https://github.com/ci-ber/autoDDPM
    #     """

    #     # 1. Generate initial anomaly mask from a noised input
    #     # autoDDPM uses T = 200 for init x_recons

    #     x_rec = self.sample_from_image(
    #         inputs,
    #         noise_level=noise_level_recon,
    #         num_inference_steps=50
    #     )
    #     x_rec = torch.clamp(x_rec, 0, 1)

    #     # 2. Calculate initial anomlay map from pixel-wise error and LPIPS score
    #     # e.q (9) in AutoDDPM paper https://arxiv.org/abs/2305.19643
    #     # mask = 
                
    #     x_res = self.ano_map.compute_residual(inputs, x_rec, hist_eq=False)
    #     lpips_mask = self.ano_map.lpips_loss(inputs, x_rec, retPerLayer=False)
    #     #
    #     # anomalous: high value, healthy: low value
    #     x_res = np.asarray(
    #         [(x_res[i] / np.percentile(x_res[i], 95)) for i in range(x_res.shape[0])]
    #     ).clip(0, 1)
    #     combined_mask_np = lpips_mask * x_res
    #     combined_mask = torch.Tensor(combined_mask_np).to(inputs.device)
    #     masking_threshold = (
    #         self.masking_threshold
    #         if self.masking_threshold >= 0
    #         else torch.tensor(
    #             np.asarray(
    #                 [
    #                     (np.percentile(combined_mask[i].cpu().detach().numpy(), 95))
    #                     for i in range(combined_mask.shape[0])
    #                 ]
    #             ).clip(0, 1)
    #         )
    #     )
    #     combined_mask_binary = torch.cat(
    #         [
    #             torch.where(
    #                 combined_mask[i] > masking_threshold[i],
    #                 torch.ones_like(torch.unsqueeze(combined_mask[i], 0)),
    #                 torch.zeros_like(combined_mask[i]),
    #             )
    #             for i in range(combined_mask.shape[0])
    #         ],
    #         dim=0,
    #     )

    #     combined_mask_binary_dilated = self.ano_map.dilate_masks(combined_mask_binary)
    #     mask_in_use = combined_mask_binary_dilated

    #     # In-painting setup
    #     # 1. Mask the original image (get rid of anomalies) and the reconstructed image (keep pseudo-healthy
    #     # counterparts)
    #     x_masked = (1 - mask_in_use) * inputs
    #     x_rec_masked = mask_in_use * x_rec
    #     #
    #     #
    #     # 2. Start in-painting with reconstructed image and not pure noise
    #     noise = torch.randn_like(x_rec, device=self.device)
    #     timesteps = torch.full(
    #         [inputs.shape[0]], noise_level_inpaint, device=self.device
    #     ).long()
    #     inpaint_image = self.inference_scheduler.add_noise(
    #         original_samples=x_rec, noise=noise, timesteps=timesteps
    #     )

    #     # 3. Setup for loop
    #     timesteps = self.inference_scheduler.get_timesteps(noise_level_inpaint)
    #     progress_bar = iter(timesteps)
    #     num_resample_steps = self.resample_steps
    #     stitched_images = []

    #     # 4. Inpainting loop
    #     with torch.no_grad():
    #         with autocast(enabled=True):
    #             for t in progress_bar:
    #                 for u in range(num_resample_steps):
    #                     # 4a) Get the known portion at t-1
    #                     if t > 0:
    #                         noise = torch.randn_like(inputs, device=self.device)
    #                         timesteps_prev = torch.full(
    #                             [inputs.shape[0]], t - 1, device=self.device
    #                         ).long()
    #                         noised_masked_original_context = (
    #                             self.inference_scheduler.add_noise(
    #                                 original_samples=x_masked,
    #                                 noise=noise,
    #                                 timesteps=timesteps_prev,
    #                             )
    #                         )
    #                     else:
    #                         noised_masked_original_context = x_masked
    #                     #
    #                     # 4b) Perform a denoising step to get the unknown portion at t-1
    #                     if t > 0:
    #                         timesteps = torch.full(
    #                             [inputs.shape[0]], t, device=self.device
    #                         ).long()
    #                         model_output = self.unet(
    #                             x=inpaint_image, timesteps=timesteps
    #                         )
    #                         inpainted_from_x_rec, _ = self.inference_scheduler.step(
    #                             model_output, t, inpaint_image
    #                         )
    #                     #
    #                     # 4c) Combine the known and unknown portions at t-1
    #                     inpaint_image = torch.where(
    #                         mask_in_use == 1,
    #                         inpainted_from_x_rec,
    #                         noised_masked_original_context,
    #                     )

    #                     ## 4d) Perform resampling: sample x_t from x_t-1 -> get new image to be inpainted
    #                     # in the masked region
    #                     if t > 0 and u < (num_resample_steps - 1):
    #                         inpaint_image = torch.sqrt(
    #                             1 - self.inference_scheduler.betas[t - 1]
    #                         ) * inpaint_image + torch.sqrt(
    #                             self.inference_scheduler.betas[t - 1]
    #                         ) * torch.randn_like(
    #                             inputs, device=self.device
    #                         )

    #     final_inpainted_image = inpaint_image
    #     x_res_2 = self.ano_map.compute_residual(
    #         inputs, final_inpainted_image.clamp(0, 1), hist_eq=False
    #     )
    #     x_lpips_2 = self.ano_map.lpips_loss(
    #         inputs, final_inpainted_image, retPerLayer=False
    #     )
    #     anomaly_maps = x_res_2 * combined_mask.cpu().detach().numpy()
    #     anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)

    #     return (
    #         anomaly_maps,
    #         anomaly_scores,
    #         {
    #             "x_rec_orig": x_rec,
    #             "x_res_orig": combined_mask,
    #             "mask": mask_in_use,
    #             "stitch": x_masked + x_rec_masked,
    #             "x_rec": final_inpainted_image,
    #         },
    #     )


def train_diffae(
    workdir=None,
    train_loader=None,
    val_loader=None,
    device="cpu",
    logger=None,
    num_train_timesteps=1000,
    learning_rate=2.5e-5,
    n_epochs=100,
    val_interval=10,
    sample_interval=50,
    autoresume=False,
    checkpoint_path=None,
):
    """
    Train the DiffAE model
    """

    if not logger:
        logger = get_logger("train", workdir=workdir, mode="a")

    # make workdir folder
    ckpt_dir = f"{workdir}/ckpt"
    sample_dir = f"{workdir}/train_samples"
    for dir in [ckpt_dir, sample_dir]:
        os.makedirs(dir, exist_ok=True)

    # Create model
    model = DiffAEResNet18(emb_dim=512)

    logger.info("Model:")
    logger.info(model)
    logger.info(
        "#model_params:", np.sum([len(p.flatten()) for p in model.parameters()])
    )

    # Autoresume from checkpoint
    if autoresume:
        # Load model checkpoint
        # First, the model looks at checkpoint_path
        # if not found then try to find latest.pth at workdir/ckpt/
        if checkpoint_path:
            ckpt_path = checkpoint_path
        else:
            ckpt_path = os.path.join(workdir, "ckpt/latest.pth")
        if not os.path.exists(ckpt_path):
            raise NameError(f"Checkpoint does not exist at {ckpt_path}")

        model, optimizer, info = load_checkpoint(ckpt_path, model)

        if info.get("epoch", None):
            start_ep = info["epoch"] + 1
            print(f"Resume training from epoch {start_ep}")
        else:
            start_ep = 0

        if info.get("best_val_loss", None):
            best_val_loss = info["best_val_loss"]
            best_val_epoch = info["best_val_epoch"]
        else:
            best_val_loss = float("inf")
            best_val_epoch = -1
    else:
        start_ep = 0
        best_val_loss = float("inf")
        best_val_epoch = -1

    model.to(device)

    # Define model optimizer, scheduler and inferer
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    inferer = DiffusionInferer(scheduler)

    logger.info("optimizer")
    logger.info(optimizer)
    logger.info("scheduler:")
    logger.info(scheduler)
    logger.info(f"Number of epochs: {n_epochs}")

    epoch_loss_list = []
    val_epoch_loss_list = []

    scaler = GradScaler(device=device)
    total_start = time.time()

    print("Start training: ")

    for epoch in range(start_ep, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            x = batch["x_origin"]
            x = rearrange(x, "b t ... -> (b t) ...").to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=True):

                # 1. Get the latent representation from semantic encoder
                latent = model.semantic_encoder(x)

                # 2. Generate random noise
                noise = torch.randn_like(x).to(device)

                # 3. Create timesteps
                timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (x.shape[0],),
                    device=x.device,
                ).long()

                # 4. Get model prediction - conditioned diffusion model
                noise_pred = inferer(
                    inputs=x,
                    diffusion_model=model.ddpm,
                    noise=noise,
                    timesteps=timesteps,
                    condition=latent.unsqueeze(1),
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        logger.info(f"Epoch: {epoch} - loss: {epoch_loss / (step + 1)}")
        epoch_loss_list.append(epoch_loss / (step + 1))

        # Evaluation during training
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                x = batch["x_origin"]
                x = rearrange(x, "b t ... -> (b t) ...").to(device)
                with torch.no_grad():
                    with autocast(device_type=device.type, enabled=True):
                        latent = model.semantic_encoder(x)
                        noise = torch.randn_like(x).to(device)
                        timesteps = torch.randint(
                            0,
                            inferer.scheduler.num_train_timesteps,
                            (x.shape[0],),
                            device=x.device,
                        ).long()
                        noise_pred = inferer(
                            inputs=x,
                            diffusion_model=model.ddpm,
                            noise=noise,
                            timesteps=timesteps,
                            condition=latent.unsqueeze(1),
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

                # Sampling during training
                # we use DDIM with 250 inference steps.
                if (step + 1 == len(val_loader)) and (
                    (epoch + 1) % sample_interval == 0
                ):
                    scheduler.set_timesteps(num_inference_steps=250)
                    with torch.no_grad():
                        with autocast(device_type=device.type, enabled=True):
                            image = inferer.sample(
                                input_noise=noise,
                                diffusion_model=model.ddpm,
                                scheduler=scheduler,
                                save_intermediates=False,
                                conditioning=latent.unsqueeze(1),
                            )

                    # Save samples
                    sample_path = os.path.join(sample_dir, f"sample_ep{epoch:03}.png")
                    grid = torchvision.utils.make_grid(
                        torch.cat([x[:10], image[:10]]),
                        nrow=10,
                        padding=2,
                        normalize=True,
                        scale_each=False,
                        pad_value=0,
                    )
                    plt.figure(figsize=(15, 5))
                    plt.imshow(grid.detach().cpu().numpy()[0], cmap="gray")
                    plt.axis("off")
                    plt.title(
                        f"Sample epoch: {epoch+1:03} - eta: 0.0 - ddim_sample_steps: 250"
                    )
                    plt.savefig(sample_path, bbox_inches="tight")
                    print(f"Epoch: {epoch} - sample saved at {sample_path}")

            avg_val_loss = val_epoch_loss / (step + 1)
            logger.info(f"Epoch: {epoch} - val_loss: {avg_val_loss}")
            val_epoch_loss_list.append(avg_val_loss)

            # Update best validation metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_epoch = epoch

        # Save latest.pth model after each epoch.
        # if the current model has the best val metrics - also save to best.pth
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (
                    scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
                ),
                "info": {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_epoch": best_val_epoch,
                },
            },
            os.path.join(ckpt_dir, "latest.pth"),
        )

        if best_val_epoch == epoch:
            shutil.copyfile(
                os.path.join(ckpt_dir, "latest.pth"), os.path.join(ckpt_dir, "best.pth")
            )
            print(
                f"Save new best model at epoch: {epoch:03} - avg_val_loss: {avg_val_loss}"
            )

    total_time = time.time() - total_start
    h, m, s = convert_sec_hms(total_time)
    logger.info(f"Train completed, total time: {h:02}:{m:02}:{s}.")


def inference_diffae(
    model: nn.Module = None,
    test_loader=None,
    ckpt_path=None,
    device="cpu",
    num_inference_steps=50,
    verbose=True,
    workdir="workdir",
    logger=None,
    ddim_eta=0.0,
    save_numpy=False,
    save_img=True,
    save_img_interval=10,
    experiment=None,
):
    """
    Perform forward-backward to reconstruct input using trained DiffusionModel
    Args:
        experiment: name of the inference experiment. This will define the dir to save result:
            sample images will be saved at dir: infer_images_{experiment}_{num_inference_steps}steps
            errors will be saved at: eval_errors_{experiment}_{num_inference_steps}steps
            Reconstructed images will be saved at: x_recons_{experiment}_{num_inference_steps}steps
    """

    if not logger:
        logger = get_logger("inference", workdir=workdir)

    # Load model checkpoint
    # Find checkpoint path: first we look at ckpt_path,
    # if not found then try to find at workdir/ckpt/best.pth or workdir/ckpt/lastest.pth
    if not os.path.exists(ckpt_path):
        for name in ["best.pth", "latest.pth"]:
            alt_path = os.path.join(workdir, f"ckpt/{name}")
            if os.path.exists(alt_path):
                ckpt_path = alt_path
                break
        else:
            raise NameError(f"Checkpoint does not exist at {ckpt_path}")

    model, _, _ = load_checkpoint(ckpt_path, model)
    model.to(device)

    # Creat outdir folders inside workdir
    # if save_numpy:
    #     np_dir = os.path.join(workdir, "sample_numpy")
    #     os.makedirs(np_dir, exist_ok=True)
    if experiment is not None:
        exp = f"_{experiment}"
    else:
        exp = ""

    img_dir = os.path.join(workdir, f"infer_images{exp}_{num_inference_steps}steps")
    if save_img:
        os.makedirs(img_dir, exist_ok=True)

    # Using DDIM sampling from (Song et al., 2020) allowing for a
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    inferer = DiffusionInferer(scheduler)

    if verbose:
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    else:
        pbar = enumerate(test_loader)

    # if save_numpy:
    error_list = []
    evals = []
    x_recons_list = []

    model.eval()
    for i, batch in pbar:
        eval = dict()

        with torch.no_grad():
            idx = batch["id"][0]
            eval["id"] = idx.item()

            # Get original images and reshape to [B T C H W] -> [(B T) C H W]
            # because for simple DDPM we perform diffusion process on each individual image
            x = batch["x_origin"]
            x = rearrange(x, "b t ... -> (b t) ...").to(device)

            # 1. ENCODER: encode input images using model.encoder
            latent = model.semantic_encoder(x)

            # 2. Generate random noise
            noise = torch.randn_like(x).to(device)

            # 3. DECODER using trained DDIM model.
            x_recons = inferer.sample(
                input_noise=noise,
                diffusion_model=model.ddpm,
                scheduler=scheduler,
                save_intermediates=False,
                conditioning=latent.unsqueeze(1),
            )

        # 4. Save reconstruction images
        x_np = x.cpu().numpy().squeeze()
        x_recons_np = x_recons.detach().cpu().numpy().squeeze()
        error = abs(x_np - x_recons_np)

        x_recons_list.append(x_recons_np)
        error_list.append(error)

        summary = summarize_errors(error, save=False)
        eval["error"] = {k: summary[k] for k in ["global", "img_wise"]}
        evals.append(eval)

        if save_img and (i % save_img_interval == 0):
            img_path = os.path.join(img_dir, f"sample_idx_{idx:03}.png")
            
            # Filter out 95th percentile error
            p95 = np.percentile(error, 95)
            error_p95 = error_filter(error, p95)

            imgs = [x_np, x_recons_np, error, error_p95]

            labels = [
                "Origin",
                f"Reconstruct\nddim_step={num_inference_steps}",
                "Error",
                "Error 95th perc",
            ]

            title = f"Reconstruction error - Model: DiffAE - Scheduler: DDIM ({num_inference_steps} steps)- id: {idx.item()}"
            opt={"title":title}

            plot_comparison_starmen(imgs, 
                                    labels, 
                                    is_errors=[False, False, True, True], 
                                    show=False, 
                                    opt=opt,
                                    save=True,
                                    save_path=img_path)

            logger.info(f"Saved images {idx:03} to {img_path}")
    
    # Save reults
    import json

    with open(
        os.path.join(workdir, f"evals_errors{exp}_{num_inference_steps}step.json"), "w"
    ) as f:
        json.dump(evals, f, indent=4)

    if save_numpy:
        np.save(
            os.path.join(workdir, f"x_recons{exp}_{num_inference_steps}step.npy"),
            x_recons_list,
        )
    
    # plot error histogram
    if img_dir: 
        hist_path = os.path.join(img_dir, "hist_error.png")
    else: 
        hist_path = os.path.join(workdir, "hist_error.png")

    error_list = np.stack(error_list).astype("float32")
    error_list_t = rearrange(error_list, "b t h w -> t (b h w)")
    opt={
        "title": f"Histogram of absolute error (pixelwise) - {exp} - {num_inference_steps}step - Nb of test idx: {error_list.shape}"
    }
    plot_error_histogram(
        error_list_t, 
        opt=opt,
        save=True,
        save_path=hist_path
    )




def main(args=None):

    DATA_DIR = "data/starmen/output_random_noacc"

    workdir = args.workdir
    experiment = args.experiment
    if workdir is None:
        if experiment is not None:
            workdir = f"workdir/{experiment}"
        else:
            workdir = "workdir"

    os.makedirs(workdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.job_type == "train":
        # Load train and val dataset
        train_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="train",
            nb_subject=None,
            save_data=False,
            workdir=workdir,
        )
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1)

        val_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="val",
            nb_subject=None,
            save_data=False,
            workdir=workdir,
        )
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=True, num_workers=1)

        train_diffae(
            workdir=workdir,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            # logger=logger,
            num_train_timesteps=1000,
            learning_rate=1.5e-5,
            n_epochs=1000,
            val_interval=10,
            sample_interval=20,
            autoresume=args.autoresume,
            checkpoint_path=args.checkpoint,
        )
    elif args.job_type == "inference":

        # Select type of dataset to run inference
        if args.datasplit is None:
            args.datasplit = "test"
            infer_exp = None
        else:
            infer_exp = args.datasplit

        test_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split=args.datasplit,
            nb_subject=None,
            save_data=False,
            workdir=workdir,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)

        model = DiffAEResNet18(emb_dim=512)

        inference_diffae(
            model=model,
            test_loader=test_loader,
            ckpt_path=args.checkpoint or os.path.join(workdir, "ckpt/best.pth"),
            device=device,
            num_inference_steps=50,
            verbose=False,
            workdir=workdir,
            save_numpy=True,
            save_img=True,
            save_img_interval=20,
            experiment=infer_exp,
        )
    else:
        NotImplementedError


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

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to load checkpoint if --autoresume is True. If not set, the model will look for latest.pth or best.pth at <workdir>/ckpt/",
    )

    parser.add_argument(
        "--datasplit",
        type=str,
        default=None,
        help="type of dataset to be used (mostly for inference). It can be train, val, test, or specified anomaly dataset <anomaly_name>",
    )

    args = parser.parse_args()

    set_determinism(42)
    main(args=args)
