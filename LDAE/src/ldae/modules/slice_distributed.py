# --------------------------------------------------------------------------------
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
from einops import rearrange


class SliceDistributed(nn.Module):
    """
    A module that applies the given module slice-by-slice on a 3D volume (first dimension is slice).
    Args:
        module (torch.nn.Module): The module to apply to each slice.
    Inputs:
        x (torch.Tensor): The input of the module with shape `(batch_size, channels, depth, height, width)`.
    Outputs:
        torch.Tensor: The output with the same shape `(batch_size, num_slices, ...)`.
    """

    def __init__(self, module):
        super(SliceDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # Modify to account for when x has spartial_dim == 2
        # x shape can be either:
        # - 5D tensor: (batch_size, C, D, H, W) for volumes
        # - 4D tensor: (batch_size, C, H, W) for images
        if x.dim() == 5: 
            batch_size, C, D, H, W = x.size()
        elif x.dim() == 4:
            x = rearrange(x, "b c h w -> b c 1 h w")
        else:
            raise ValueError(f"Unsupported input dimensions: {x.shape}")
        batch_size, C, D, H, W = x.size()
        x = x.view(batch_size * D, C, H, W)
        x = self.module(x)
        _, c = x.size()
        x = x.view(batch_size, D, c)

        return x
