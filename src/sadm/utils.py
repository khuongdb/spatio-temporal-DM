
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
import sys

def plot_multi_imgs(imgs, labels, save=False, save_path=None, opt=None):

    if opt is None:
        opt = {}

    if not isinstance(imgs, list):
        imgs = [imgs]


    processed_imgs = []
    for img in imgs:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim > 3: 
            img = img.squeeze()
        processed_imgs.append(img)

    imgs = processed_imgs

    nb_cols = len(imgs[0]) + 1
    fig, axes = plt.subplots(len(imgs), nb_cols, figsize=(2*nb_cols, 2 * len(imgs)))
    axes = np.atleast_2d(axes)
    ws = opt.get("wspace", 0)
    hs = opt.get("hspace", 0) 
    
    plt.subplots_adjust(wspace=ws, hspace=hs)

    # color = opt.get("cmap", "gray")
    color = opt.get("cmap", None)
    if color:
        if not isinstance(color, list):
            color = [color]
        elif len(color) < len(imgs):
            color += [color[-1]] * (len(imgs) - len(color))
    else: 
        color = ["gray"] * len(imgs)

    for i, i_imgs in enumerate(imgs):
        axes[i][0].axis('off')  
        axes[i][0].text(0.5, 0.5, labels[i],
                    fontsize=12, ha='center', va='center')
        for j, img in enumerate(i_imgs):
            if not np.all(img == 0): 
                axes[i][j + 1].matshow(img, aspect="equal", cmap=color[i])

    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])

    if save and save_path is not None:
        # dpi = opt.get("dpi", 150)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

def get_logger(name, workdir, level=logging.INFO, mode="w"):
    """
    Set up logger
    Args: 
        mode: "w" to overwrite existing file, "a": append existing file. 
    """
    logger = logging.getLogger(name)  # noqa: F821
    logger.setLevel(level)
    file_handler = logging.FileHandler(f"{workdir}/{name}.log", mode=mode)
    console_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def save_ckp(state, step=None, checkpoint_dir="workdir/", epoch=0):
    """
    Save latest checkpoint after each epoch.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
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


def plot_sample_process(start, inter):
    """
    Plotting sampling process along DDPM's Markov chain
    """
    if start: 
        inter.insert(0, start)
    chain = torch.cat(inter, dim=-1)

    plt.style.use("default")

    plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="viridis")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def convert_sec_hms(seconds):
    """
    convert time from secone to hour - minutes - seconds
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return int(hours), int(minutes), int(remaining_seconds)