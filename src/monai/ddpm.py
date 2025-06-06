

import os
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
# from monai import transforms
# from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from src.data import StarmenDataset
from einops import rearrange
from src.sadm.utils import get_logger, save_ckp

def main():

    DATA_DIR = "data/starmen/output_random_noacc"
    WORKDIR = "workdir/monai_ddpm"

    logger=get_logger("ddpm", workdir=WORKDIR)

    train_ds = StarmenDataset(data_dir=DATA_DIR, 
                                split="train", 
                                nb_subject=1,
                                save_data=False, 
                                workdir="workdir/test")
    train_loader = DataLoader(train_ds, 
                                batch_size=2,
                                shuffle=True, 
                                num_workers=1)


    val_ds = StarmenDataset(data_dir=DATA_DIR, 
                                split="val", 
                                nb_subject=1,
                                save_data=False, 
                                workdir="workdir/test")
    val_loader = DataLoader(val_ds, 
                                batch_size=2,
                                shuffle=True, 
                                num_workers=1)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(16, 32, 32),
        norm_num_groups=16,
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=8,
    )
    model.to(device)


    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    inferer = DiffusionInferer(scheduler)


    ###
    # Train the model
    ###

    n_epochs = 1
    val_interval = 10
    epoch_loss_list = []
    val_epoch_loss_list = []

    scaler = GradScaler(device=device)
    # total_start = time.time()


    for epoch in range(n_epochs):
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
        save_path = "workdir/monai_ddpm/ckpt/latest.pt"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "epoch": epoch,
        }, save_path)



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
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Sampling image during training
            noise = torch.randn((1, 1, 64, 64))
            noise = noise.to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(device_type=device.type, enabled=True):
                image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)

            # Save samples
            sample_path = os.path.join(WORKDIR, f"sample_ep{epoch}.np")
            np.save(sample_path, image.cpu().detach().numpy())


if __name__ == "__main__":
    set_determinism(42)
    main()