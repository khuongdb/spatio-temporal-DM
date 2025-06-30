
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

from torch import nn
import torch

import torch
import torch.nn as nn


def initialize_weights(model: nn.Module, init_type: str = 'kaiming'):
    """
    Given a PyTorch model, initialize its weights using the specified technique.
    This function handles 2D and 3D convolutional layers, linear layers,
    and various normalization layers (BatchNorm, LayerNorm).

    Args:
        model (nn.Module): The PyTorch model to initialize.
        init_type (str): The type of initialization to apply.
                         One of ['kaiming', 'xavier', 'orthogonal', 'normal'].

    Returns:
        nn.Module: The input model with initialized weights.
    """

    for module in model.modules():
        # --- 2D Convolutional layers (Conv2d, ConvTranspose2d) ---
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            elif init_type == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init_type {init_type}")

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        # --- 3D Convolutional layers (Conv3d, ConvTranspose3d) ---
        elif isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            elif init_type == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init_type {init_type}")

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        # --- Linear layers ---
        elif isinstance(module, nn.Linear):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            elif init_type == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init_type {init_type}")

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        # --- 2D BatchNorm layers ---
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

        # --- 3D BatchNorm layers ---
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

        # --- LayerNorm (common in Transformers, e.g. Swin) ---
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    print(f"Initialized {model.__class__.__name__} weights using {init_type} initialization\n")


def freeze_backbone(backbone, freeze_perc=1.0):
    """
    Freeze a percentage of the backbone layers of a model.

    Args:
        backbone (nn.Module): The PyTorch model whose backbone layers will be frozen.
        freeze_perc (float): The percentage of backbone layers to freeze. Default: 1.0 (freeze all layers).

    Returns:
        nn.Module: The modified model with the specified percentage of backbone layers frozen.
    """
    # Calculate the number of layers to freeze
    num_layers = len(list(backbone.parameters()))
    num_layers_to_freeze = int(num_layers * freeze_perc)

    # Freeze the specified percentage of backbone layers
    for i, param in enumerate(backbone.parameters()):
        if i < num_layers_to_freeze:
            param.requires_grad = False

    print(f"Frozen first {num_layers_to_freeze} layers out of {num_layers} in the backbone\n")
    return backbone


def convert_first_layer_rgb_to_grayscale(model):
    """
    Converts the first convolutional layer of a model in place to accept grayscale (1-channel) input.
    This function assumes the first layer is a convolutional layer.

    This is not a perfect mapping but it can works resonably well. Normally, each layers (dim=1) will extract the feature map of 1 color channel (R, G, B).
    If we think of handling gray images as duplicate the value across all 3 channels, then the feature maps (for ech out_channels) can be formulate: 

    feat = w_r * img + w_b * img + w_g * img = (w_r + w_b + w_g) * img

    with img = [B, 1, H, W]
    w_r = W[:, 0, :, :] - weight of Red channel. 
    w_b = W[:, 0, :, :] - weight of Blue channel. 
    w_g = W[:, 0, :, :] - weight of Green channel.
    W = [B, 3, H ,W] - original weights for RGB images.  

    Other approaches can be: 
        1. Take the avg of 3 channel weights:
        ``` 
        grayscale_weight = first_conv.weight.mean(dim=1, keepdim=True)
        ```

        2. Weighted sum using luminance coefficients:
        Since grayscale is often computed as a weighted sum of RGB:
        Gray=0.2989R+0.5870G+0.1140B
        ```
        weights = first_conv.weight
        r, g, b = weights[:, 0:1, :, :], weights[:, 1:2, :, :], weights[:, 2:3, :, :]
        grayscale_weight = 0.2989 * r + 0.5870 * g + 0.1140 * b
        ```

    Args:
        model (nn.Module): The PyTorch model whose first convolutional layer will be converted to accept grayscale input.
    """
    # Find the first convolutional layer
    first_conv = model
    while len(list(first_conv.children())) > 0:
        first_conv = list(first_conv.children())[0]

    # Check if the found layer is a convolutional layer
    if isinstance(first_conv, nn.Conv2d) or isinstance(first_conv, nn.Conv3d):
        if first_conv.in_channels == 3:
            # Convert the first convolutional layer to accept grayscale (1-channel) input
            first_conv.in_channels = 1

            # Sum over the color channels to convert RGB to grayscale weights
            with torch.no_grad():
                grayscale_weight = first_conv.weight.sum(dim=1, keepdim=True)

            # Update the weights while preserving gradient information
            first_conv.weight = nn.Parameter(grayscale_weight)
        else:
            print("First conv layer is not designed for 3-channel (RGB) input")
    else:
        print("No convolutional layer found or first layer is not a convolution.")


def replace_classifier_with_identity(model, emb_dim):
    """
    Replaces the last layer (classifier) of the model with an identity layer.
    Optionally add a nn.Linear layer to match output dimension (of backbone model) with embbed dimension (of LDAE model) 
    This function assumes the classifier is the last layer of the model.

    Args:
        model (nn.Module): The PyTorch model whose classifier will be replaced.
        emb_dim (int): embbed dimension of LDAE model that we need to match out_features dimension with. 

    Returns:
        nn.Module: The modified model with the classifier replaced by an identity layer.
    """
    # Check if the model is a VGG model
    if 'VGG' in model.__class__.__name__:
        # Replace the last layer with an identity layer
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif 'SwinTransformer' in model.__class__.__name__:
        # Replace the head layer with an identity layer
        model.head = nn.Identity()
        return model
    elif 'ConvNeXt' in model.__class__.__name__:
        fc = model.classifier[-1]
        out_dim = fc.out_features
        if out_dim == emb_dim: 
            model.classifier[-1] = nn.Identity()
        else: 
            fc = nn.Linear(in_features=fc.in_features, out_features=emb_dim)
            model.classifier[-1] = fc
        return model
    # Check if the model has a sequential structure
    if isinstance(model, nn.Sequential):
        # Replace the last layer with an identity layer
        model[-1] = nn.Identity()
    else:
        # Identify if the model has a classifier attribute (common in many models like ResNet, etc.)
        if hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'fc'):  # Common in some models like ResNet
            out_dim = model.fc.out_features
            if out_dim == emb_dim: 
                model.fc = nn.Identity()
            else:
                model.fc = nn.Linear(in_features=model.fc.in_features, out_features=emb_dim)
        else:
            raise ValueError("The model does not have a recognized classifier or fully connected layer.")

    return model
