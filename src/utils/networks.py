import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.monai.vivit import ViT
from generative.networks.nets import DiffusionModelUNet


def load_checkpoint(
    checkpoints_path: Optional[str], 
    model: nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int]:
    """
    Load pretrained model and optimizer states if available.

    Args:
        checkpoints_path (Optional[str]): Path to checkpoint file.
        model (nn.Module): Model to load state dict into.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state dict into.

    Returns:
        Tuple[nn.Module, Optional[torch.optim.Optimizer], int]: 
            - Initialized model
            - Initialized optimizer (if provided)
            - Starting epoch (0 if no checkpoint found or epoch info missing)
    """
    if checkpoints_path is not None:
        if not os.path.exists(checkpoints_path):
            print(f"Checkpoint does not exist at {checkpoints_path}. Train from start.")
            return model
        else:
            device = next(model.parameters()).device # Use the same device as the model
            ckpt = torch.load(checkpoints_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            
            if optimizer and ckpt.get("optimizer_state", None): 
                optimizer.load_state_dict(ckpt["optimizer_state"])
            print(f"Load checkpoint from {checkpoints_path}")
        
            info = ckpt.get("info", None)
            if info is None: 
                epoch = ckpt.get("epoch")
                info = {"epoch": epoch}


    return model, optimizer, info


def init_vivit(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ViVit model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ViVit
    """
    vivit = ViT(
            image_size=64,  # image size
            frames=9,  # number of frames
            image_patch_size=16,  # image patch size
            frame_patch_size=1,  # frame patch size
            channels=1,
            out_channels=32,
            dim=32,
            spatial_depth=6,  # depth of the spatial transformer
            temporal_depth=6,  # depth of the temporal transformer
            heads=8,
            mlp_dim=512,
            variant="factorized_encoder",  # or 'factorized_self_attention'
            reduce_dim=True,  # perform global pooling or exercise cls token.
            out_upsample=False, 
        )
    return load_checkpoint(checkpoints_path, vivit)



def init_ddpm(checkpoints_path: Optional[str] = None):
    """
    Load the DDPM model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=8,
    )
    return load_checkpoint(checkpoints_path, model)


def init_sadm(checkpoints_path = None):
    """
    Load the SADM model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the SADM model includes ViVit and DDPM
    """

