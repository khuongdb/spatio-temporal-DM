# ------------------------------------------------------------------------
# Modified from SADM (https://github.com/ubc-tea/SADM-Longitudinal-Medical-Image-Generation)
# ------------------------------------------------------------------------


from src.sadm.ddpm import DDPM
from src.sadm.ddpm_module import ContextUnet
from src.sadm.vivit import ViViT
from src.data import StarmenDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os
import shutil
from einops import rearrange

import argparse
import logging
from src.sadm.utils import get_logger

import yaml


# DATA_DIR = "./data/starmen/output_random_noacc"
RESULT_DIR = "./outdirs"

assert os.path.isdir(RESULT_DIR), f"{RESULT_DIR} is not a directory."


def parse_args():
    """
    parser: argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        metavar="L",
        help='log level from : ["INFO", "DEBUG", "WARNING", "ERROR"]',
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/sadm.yaml",
        metavar="C",
        help="path to configuration yaml file",
    )

    return parser.parse_args()


def save_ckp(state, step=None, checkpoint_dir="workdir/", epoch=0):
    """
    Save latest checkpoint after each epoch.
    """
    ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(state, ckpt_path)
    if step: 
        if (epoch % step == 0) and (epoch != 0):
            f_path = os.path.join(checkpoint_dir, f"ep_{epoch}.pt")
            shutil.copyfile(ckpt_path, f_path)
    # if is_best:
    #     best_fpath = os.path.join(checkpoint_dir, "latest.pt")
    #     shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load the latest checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


def train(
    train_loader,
    val_loader=None,
    device="cpu",
    nb_epoch=200,
    learning_rate=1e-4,
    auto_resume=False,
    eval=False,
    eval_guide_w=0.2,
    ddpm_args=None,
    vivit_args=None,
    unet_args=None,
    **options
):
    


    # nb_epoch = 500
    # batch_size = 3
    # image_size = (32, 128, 128)
    # image_size = (64, 64)
    # num_frames = 9

    # DDPM hyperparameters
    # n_T = 500  # 500
    # n_feat = 8  # 128 ok, 256 better (but slower)
    # lrate = 1e-4
    # is_3d = False

    # ViViT hyperparameters
    # patch_size = (8, 32, 32)
    # patch_size = (16, 16)

    # x_val, x_prev_val = next(iter(val_loader))
    # x_prev_val = x_prev_val.to(device)

    # setup logger
    logger = get_logger("train", workdir)

    vivit_model = ViViT(**vivit_args)
    nn_model = ContextUnet(**unet_args)
    

    ddpm = DDPM(
        vivit_model=vivit_model,
        nn_model=nn_model,
        device=device,
        **ddpm_args
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=learning_rate)

    # check if auto resume from previous checkpoint
    ckpt_path = os.path.join(workdir, "ckpt")
    os.makedirs(ckpt_path, exist_ok=True)
    if auto_resume:
        last_ckpt_path = os.path.join(ckpt_path, "latest.pt")
        has_ckpt = os.path.exists(last_ckpt_path)
        if has_ckpt:
            ddpm, optim, start_ep = load_ckp(last_ckpt_path, ddpm, optim)
            start_ep += 1
            print(f"Resuming from {start_ep} - loaded from {ckpt_path}")
    else:
        start_ep = 0
        print("Training from start.")

    for ep in range(start_ep, nb_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        # linear learning_rate decay
        optim.param_groups[0]["lr"] = learning_rate * (1 - ep / nb_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        # for x, x_prev, x_origin in pbar:
        for item in pbar:
            x = item["x"]
            x_prev = item["x_prev"]
            optim.zero_grad()
            x = x.to(device)
            x_prev = x_prev.to(device)
            loss = ddpm(x, x_prev)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss_ema: {loss_ema:.4f}")
            optim.step()

        if eval: 
            ddpm.eval()
            with torch.no_grad():
                # x_gen, x_gen_store = ddpm.sample(x_prev_val, device, guide_w=0.2)
                # np.save(f"{RESULT_DIR}/x_gen_{ep}.npy", x_gen)
                # np.save(f"{RESULT_DIR}/x_gen_store_{ep}.npy", x_gen_store)

                # Calculate the validation loss
                val_losses = []
                # for x_val_batch, x_prev_val_batch, x_origin_val in val_loader:
                for val_item in val_loader:
                    x_val = val_item["x"].to(device)
                    x_prev_val = val_item["x_prev"].to(device)
                    val_loss = ddpm(x_val, x_prev_val)
                    x_gen, x_gen_store = ddpm.sample(x_prev_val, device, guide_w=0.2)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"avg_val_loss: {avg_val_loss:.4f}")
            
            # Log infos
            logger.info(
                f"Epoch {ep+1}/{nb_epoch} - train_loss: {loss:.4f} - train_loss_ema: {loss_ema:.4f} - val_avg_loss: {avg_val_loss:.4f}"
            )
        else: 
            logger.info(
                f"Epoch {ep+1}/{nb_epoch} - train_loss: {loss:.4f} - train_loss_ema: {loss_ema:.4f}"
            )

        # Save checkpoint for current epoch
        checkpoint = {
            "epoch": ep,
            "state_dict": ddpm.state_dict(),
            "optimizer": optim.state_dict(),
        }
        save_ckp(checkpoint, step=None, checkpoint_dir=ckpt_path, epoch=ep)




def inference(
    weights="workdir/ckpt/latest.pt", 
    outdir="workdir/samples", 
    guide_w=0.2,
    test_loader = None, 
    device="cpu",
    ddpm_model=None, 
):
    """
    Inference step on test dataset.
    """

    assert os.path.exists(weights), f"Weights does not exist at {weights}"

    # setup logger
    logger = get_logger("inference", workdir)

    # Load weights
    ddpm_model, optim, start_ep = load_ckp(checkpoint_path=weights, 
                                           model=ddpm_model,
                                           device=device)
    ddpm_model.eval()
    logger.info(f"Load weight from {weights}")

    # Inference loop
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        # only use batch of 1 for toy example.
        pbar = tqdm(test_loader)
        for item in pbar:
            x_origin = item["x_origin"][0].to(device)
            mask = item["mask"][0].to(device)
            # x_prev = item["x_prev"][0].to(device)
            x_gen = torch.empty_like(x_origin)
            # Assume that we always have the first item.
            x_gen[0] = x_origin[0]

            for i in range(1, 10):
                if mask[i] == 1:
                    x_gen[i] = x_origin[i]
                else:
                    # Generate new image using Autogressive model
                    # print(f"Generate image {i}...")
                    mask_i = mask.clone()
                    mask_i[:i] = 1
                    mask_i[i:] = 0
                    x_prev_i = x_gen[:-1] * mask_i[:-1, None, None, None]
                    x_prev_i = rearrange(
                        x_prev_i, "t c h w -> 1 t c h w"
                    ).float()  # work around because ddpm.sample operate on batch.
                    # print(x_prev_i.shape)
                    x_gen_i, _ = ddpm_model.sample(x_prev_i, device=device, guide_w=guide_w)
                    x_gen[i] = x_gen_i.squeeze(0)
                    # print(x_prev_i.shape)
                    # x_gen_i, _ = ddpm.sample(x_prev_i, device, guide_w=0.2)

            # Save the result
            idx = item["id"].item()
            np.save(os.path.join(outdir, f"x_gen_{idx:03}.npy"), x_gen.cpu().numpy())
            x_masked_np = x_origin * rearrange(mask, "b -> b 1 1 1")
            x_masked_np = x_masked_np.cpu().detach().numpy()
            np.save(os.path.join(outdir, f"x_masked_{idx:03}.npy"), x_masked_np)

            logger.info(f"Saved masked_input and generated sample {idx:03} to {outdir}")




if __name__ == "__main__":

    args = parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print(f"Success: Loaded configuration file at: {args.config}")
    except Exception as e:
        print(f"ERROR: Could not load config file: {e}")
        exit()

    exp = config.get("experiment", None)

    # create a workdir folder for the experience
    if exp.get("workdir", None):
        workdir = exp["workdir"]
    elif exp.get("name", None):
        workdir = os.path.join("workdir", exp["name"])
    else:
        # take the name of the config file
        exp_name = os.path.splitext(os.path.basename(args.config))[0]
        workdir = os.path.join("workdir", exp_name)
    os.makedirs(workdir, exist_ok=True)


    # Set up logging

    if args.log_level == "INFO":
        log_level = logging.INFO
    elif args.log_level == "DEBUG":
        log_level = logging.DEBUG
    elif args.log_level == "WARNING":
        log_level = logging.WARNING
    elif args.log_level == "ERROR":
        log_level = logging.ERROR

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{workdir}/logging.log"),
            logging.StreamHandler(),
        ],
    )
    # logger = get_logger("experiment", workdir, log_level)
    

    logging.info(
        "------------------------------- SEQUENCE AWARE DIFFUSION MODEL  -------------------------------"
    )
    logging.info(f"Success: Loaded configuration file at: {args.config}")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    data_dir = config.get("data_dir", None)
    assert os.path.isdir(data_dir), logging.error(f"{data_dir} is not a directory.")


    if exp["task"] == "train":
        logging.info("TRAINING")

        train_ds = StarmenDataset(data_dir=data_dir, 
                                  split="train", 
                                  nb_subject=config["train_ds"]["nb_subject"],
                                  workdir=workdir)
        train_loader = DataLoader(train_ds, 
                                  batch_size=config["train_ds"]["batch_size"],
                                  shuffle=config["train_ds"]["shuffle"], 
                                  num_workers=config["train_ds"]["num_workers"])
        # logger.info(f"Load dataset {train_loader}")
        # if config["train_ds"]["nb_subject"]:

        val_ds = StarmenDataset(data_dir=data_dir, 
                                split="val", 
                                nb_subject=config["val_ds"]["nb_subject"],
                                workdir=workdir)
        val_loader = DataLoader(val_ds, 
                                batch_size=config["val_ds"]["batch_size"],
                                shuffle=config["val_ds"]["shuffle"], 
                                num_workers=config["val_ds"]["num_workers"])
        train(
            train_loader,
            val_loader,
            device,
            ddpm_args=config["ddpm"]["params"],
            vivit_args=config["vivit"]["params"],
            unet_args=config["contextUNET"]["params"],
            **config["trainer"]
        )

    elif exp["task"] == "inference":
        logging.info("INFERENCE")
        if not config["inference"].get("weights", None):
            config["inference"]["weights"] = os.path.join(workdir, "ckpt", "latest.pt")
        if not config["inference"].get("outdir", None):
            config["inference"]["outdir"] = os.path.join(workdir, "samples")

        # Load test dataset
        csv_path = config["test_ds"].get("csv_path", None)
        test_dataset = StarmenDataset(data_dir=data_dir, 
                                      split="test", 
                                      workdir=workdir,
                                      csv_path=csv_path,
                                      nb_subject=config["test_ds"]["nb_subject"])
        test_loader = DataLoader(test_dataset, 
                                batch_size=config["test_ds"]["batch_size"],
                                shuffle=config["test_ds"]["shuffle"], 
                                num_workers=config["test_ds"]["num_workers"])        
        logging.info(f"Load test dataset {test_loader}")

        # Define models
        #!TODO: move this to main() to use for both train() and inference()
        vivit_model = ViViT(**config["vivit"]["params"])
        nn_model = ContextUnet(**config["contextUNET"]["params"])
        ddpm = DDPM(
            vivit_model=vivit_model,
            nn_model=nn_model,
            device=device,
            **config["ddpm"]["params"]
        )
        ddpm.to(device)

        inference(test_loader = test_loader, 
                  device=device,
                  ddpm_model=ddpm,
                  **config["inference"])
        
    else:
        print("Task need to be train or inference.")
        exit()
