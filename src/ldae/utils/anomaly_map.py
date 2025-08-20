# Function from DDAD 
# https://github.com/arimousa/DDAD/

import math
from kornia.filters import gaussian_blur2d
import torch
import torch.nn.functional as F


def heat_map(output, target, FE, v=1.0, fe_layers=["layer2", "layer3"], use_gaussian_blue=True, sigma=4, device=None):
    '''
    Compute the anomaly map based on combination of pixel distance and feature distance
    Modify from https://github.com/arimousa/DDAD/blob/main/anomaly_map.py

    Args: 
        output: the output of the reconstruction, size [B, C, H, W]
        target: the target (original) image, size [B, C, H, W]
        FE: the feature extractor
        sigma: the sigma of the gaussian kernel
        i_d: the pixel distance
        f_d: the feature distance
        v (float): parameter controls the strength of pixel distance. 
        use_guassian_blue (bool): whether to aplly gaussian blur kernel to anomaly_map
        sigma (int): standard variation of the Gaussian kernel. 
            Small sigma, small kernel → light blur
            Large sigma, large kernel → strong blur
    
    Returns: 
        torch.Tensor: size [B, C, H, W]
    '''
    # sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) +1

    # Sanity check for inputs and FE network
    if device is None:
        device = next(FE.parameters()).device

    output = output.to(device)
    target = target.to(device)

    # pixel distance
    # i_d = pixel_distance(output, target)
    i_d = torch.mean(torch.abs(output - target), dim=1).unsqueeze(1)

    # feature distance
    f_d = feature_distance(output,  target, FE, fe_layers=fe_layers, device=device)
    f_d = torch.Tensor(f_d).to(device)

    # combine i_d and f_d to generate anomaly_map
    anomaly_map = torch.zeros_like(f_d)

    anomaly_map += f_d + v * (torch.max(f_d)/ torch.max(i_d)) * i_d  
    if use_gaussian_blue:
        anomaly_map = gaussian_blur2d(
            anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
            )
    anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
    anomaly_map = anomaly_map.float()
    return anomaly_map, f_d, i_d

@torch.no_grad()
def feature_distance(output, target, FE, fe_layers=["layer2", "layer3"], device=None):
    """
    Feature distance between output and target
    Modify from https://github.com/arimousa/DDAD/blob/main/anomaly_map.py

    Args:
        output (torch.Tensor): Reconstructed image from the diffusion model, shape (B, C, H, W).
        target (torch.Tensor): Original image, shape (B, C, H, W).
        FE (torch.nn.Module): Feature extractor network.
        fe_layers (list): list of feature layers to extracted from FE. Default to ["layer2", "layer3"]

    Returns:
        torch.Tensor: Feature distance between `output` and `target`, shape [B, 1, H, W]
    """

    if device is None:
        device = next(FE.parameters()).device

    FE.eval()

    _, _, _, out_size = output.shape

    with torch.no_grad():
        _, inputs_features = FE(target, fe_layers)  # list([B, D, H, W])
        _, output_features = FE(output, fe_layers)  # list([B, D, H, W])
        # _, _, out_size = config.data.image_size

    # init anomaly map shape [B, 1, H, W]
    anomaly_map = torch.zeros([inputs_features[0].shape[0] ,1 ,out_size, out_size]).to(device)
    for i in range(len(inputs_features)):
        # if i == 0:
        #     continue
        a_map = 1 - F.cosine_similarity(patchify(inputs_features[i]), patchify(output_features[i]))
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map
    return anomaly_map 

@torch.no_grad()
def patchify(features, return_spatial_info=False):
    """Convert a tensor into a tensor of respective patches.
    Args:
        x: [torch.Tensor, bs x c x w x h]
    Returns:
        x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
        patchsize]
    
    Ported from https://github.com/amazon-science/patchcore-inspection
    """
    patchsize = 3
    stride = 1
    padding = int((patchsize - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
        n_patches = (
            s + 2 * padding - 1 * (patchsize - 1) - 1
        ) / stride + 1
        number_of_total_patches.append(int(n_patches))
    unfolded_features = unfolded_features.reshape(
        *features.shape[:2], patchsize, patchsize, -1
    )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    max_features = torch.mean(unfolded_features, dim=(3,4))
    features = max_features.reshape(features.shape[0], int(math.sqrt(max_features.shape[1])) , int(math.sqrt(max_features.shape[1])), max_features.shape[-1]).permute(0,3,1,2)
    if return_spatial_info:
        return unfolded_features, number_of_total_patches
    return features