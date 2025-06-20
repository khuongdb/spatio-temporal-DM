

import os
import shutil
import tempfile
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

# from monai import transforms
# from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from src.data import StarmenDataset
from src.utils import convert_sec_hms, get_logger
from src.utils.metrics import summarize_errors
from src.utils.networks import load_checkpoint
from src.sadm.utils import plot_multi_imgs


def train_ddpm(
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
    autoresume=False,
    checkpoint_path=None
):
    """
    Train the DDPM model
    """

    if not logger:
        logger = get_logger("train", workdir=workdir, mode="w")


    # make workdir folder
    ckpt_dir = f"{workdir}/ckpt"
    sample_dir = f"{workdir}/train_samples"
    for dir in [ckpt_dir, sample_dir]:
        os.makedirs(dir, exist_ok=True)


    logger.info("Model:")
    logger.info(model)

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
        best_val_loss = float('inf')
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
                # Generate random noise
                noise = torch.randn_like(x).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (x.shape[0],), device=x.device
                ).long()

                # Get model prediction
                noise_pred = inferer(inputs=x, diffusion_model=model, noise=noise, timesteps=timesteps)

                loss = F.mse_loss(noise_pred.float(), noise.float())

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

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "epoch": epoch,
        }, os.path.join(ckpt_dir, "latest.pth"))

        # Evaluation during training
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                x = batch["x_origin"]
                x = rearrange(x, "b t ... -> (b t) ...").to(device)
                with torch.no_grad():
                    with autocast(device_type=device.type, enabled=True):
                        noise = torch.randn_like(x).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (x.shape[0],), device=x.device
                        ).long()
                        noise_pred = inferer(inputs=x, diffusion_model=model, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

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
                    "best_val_epoch": best_val_epoch
                }
            },
            os.path.join(ckpt_dir, "latest.pth"),
        )

        if best_val_epoch == epoch:
            shutil.copyfile(
                os.path.join(ckpt_dir, "latest.pth"),
                os.path.join(ckpt_dir, "best.pth")
            )
            print(f"Save new best model at epoch: {epoch:03} - avg_val_loss: {avg_val_loss}")
    

        # Sampling image during training
        if (epoch + 1) % sample_interval == 0: 
            model.eval()
            noise = torch.randn((1, 1, 64, 64))
            noise = noise.to(device)
            scheduler.set_timesteps(num_inference_steps=250)
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=True):
                    image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)

            # Save samples
            image = image.detach().cpu().numpy().squeeze()
            sample_path = os.path.join(sample_dir, f"sample_ep{epoch:03}.png")

            plt.imshow(image, vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.title(f"Sample epoch: {epoch+1:03} - eta: 0.0 - ddim_sample_steps: 250")
            plt.savefig(sample_path, bbox_inches="tight")
            print(f"Epoch: {epoch} - sample saved at {sample_path}")


    total_time = time.time() - total_start
    h, m, s = convert_sec_hms(total_time)
    logger.info(f"train completed, total time: {total_time}.")



def inference_ddpm(
    model: nn.Module = None,
    test_loader=None,
    ckpt_path=None,
    device="cpu",
    num_inference_steps=250,
    verbose=True,
    workdir="workdir",
    logger=None,
    ddim_eta=0.0,
    save_numpy=False,
    save_img=True,
    save_img_interval=10
):

    """
    Perform forward-backward to reconstruct input using trained DiffusionModel
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
    if save_img:
        img_dir = os.path.join(workdir, f"infer_images_{num_inference_steps}steps")
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
    errors = []
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

            img = x.clone().detach()

            # 1. ENCODER: encode the images into (latent) representation x_T
            # here we use the reverse generative process (forward process)

            scheduler.clip_sample = False
            for i in range(num_inference_steps):  # go through the noising process
                t = i
                with torch.no_grad():
                    model_output = model(img, timesteps=torch.Tensor((t,)).to(x.device))
                img, _ = scheduler.reversed_step(model_output, t, img)

            x_latent = img

            # 2. DECODING: reconstruct the images from latent space
            # now img is noisy input. 

            # x_recons = inferer.sample(x_latent, model, scheduler)
            for i in range(num_inference_steps):
                t = num_inference_steps - i
                with torch.no_grad():
                    noise_pred = model(img, timesteps=torch.Tensor((t,)).to(x.device))
                img, _ = scheduler.step(noise_pred, t, img)
            
            x_recons = img.clone()

        # 3. Calculate error 
        x_np = x.cpu().numpy().squeeze()
        x_recons_np = x_recons.detach().cpu().numpy().squeeze()
        error = abs(x_np - x_recons_np)

        summary = summarize_errors(error, save=False)
        eval["error"] = {k: summary[k] for k in ["global", "img_wise"]}
        evals.append(eval)
        errors.append(error)
        x_recons_list.append(x_recons_np)

        # filter out 95 percentile error
        pct95 = summary["img_wise"]["95%"]
        pct95 = np.array(pct95, dtype="float32")[:, np.newaxis, np.newaxis]
        mask = error > pct95
        error_pct95 = np.where(mask, error, 0)

        if save_img and (i % save_img_interval == 0):
            img_path = os.path.join(img_dir, f"sample_idx_{idx:03}.png")

            imgs = [
                x_np,
                x_recons_np,
                error,
                error_pct95
            ]

            labels = [
                "origin",
                f"reconstruct\neta={ddim_eta}\nddim_step={num_inference_steps}",
                "error",
                "error_95pct"
            ]

            plot_multi_imgs(
                imgs=imgs,
                labels=labels,
                save=True,
                save_path=img_path,
                opt={"hspace": 0.05, "cmap": "gray"},
            )

            logger.info(f"Saved images {idx:03} to {img_path}")


    # Save reults
    import json
    with open(os.path.join(workdir, f"evals_errors_{num_inference_steps}step.json"), "w") as f:
        json.dump(evals, f, indent=4)

    if save_numpy:
        np.save(os.path.join(workdir, f"x_recons_{num_inference_steps}step.npy"), 
                x_recons_list)




def main(args=None):

    DATA_DIR = "data/starmen/output_random_noacc"

    workdir = args.workdir
    experiment = args.experiment
    if workdir is None:
        workdir = f"workdir/{experiment}"

    os.makedirs(workdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define ddpm model
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        # norm_num_groups=16,
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=8,
    )

    if args.job_type == "train":

        # Load dataset    
        train_ds = StarmenDataset(data_dir=DATA_DIR, 
                                    split="train", 
                                    nb_subject=None,
                                    save_data=False, 
                                    workdir="workdir/test")
        train_loader = DataLoader(train_ds, 
                                    batch_size=2,
                                    shuffle=True, 
                                    num_workers=1)


        val_ds = StarmenDataset(data_dir=DATA_DIR, 
                                    split="val", 
                                    nb_subject=None,
                                    save_data=False, 
                                    workdir="workdir/test")
        val_loader = DataLoader(val_ds, 
                                    batch_size=2,
                                    shuffle=True, 
                                    num_workers=1)

        train_ddpm(
            workdir=workdir,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model=model,
            # logger=logger,
            num_train_timesteps=1000,
            learning_rate=1.5e-5,
            n_epochs=200,
            val_interval=10,
            sample_interval=50,
            autoresume=args.autoresume
        )

    elif args.job_type == "inference":
        test_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="test",
            nb_subject=None,
            save_data=False,
            workdir=workdir,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)
        
        inference_ddpm(
            model=model,
            test_loader=test_loader,
            ckpt_path=os.path.join(workdir, "ckpt/best.pth"),
            device=device,
            num_inference_steps=50,
            verbose=False,
            workdir=workdir,
            save_numpy=True,
            save_img=True,
            save_img_interval=20,
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


    args = parser.parse_args()

    set_determinism(42)
    main(args=args)
