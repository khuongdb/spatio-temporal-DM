import torch
from einops import rearrange
from skimage.filters import threshold_yen
from scipy.signal import medfilt2d
from scipy.ndimage import binary_dilation, generate_binary_structure
import numpy as np
import math
from kornia.filters import gaussian_blur2d
import torch.nn.functional as F


def norm_tensor(tensor, eps=1e-8):
    my_max = torch.max(tensor)
    my_min = torch.min(tensor)
    range_ = my_max - my_min

    if range_ < eps:
        return torch.zeros_like(tensor) 
    else:
        return (tensor - my_min) / range_

# Median filter
def median_filter_2D(volume, kernelsize=5):
    """
    volume: torch.Tensor(c, h, w) | torch.Tensor(b, c, h, w)
    """
    volume = volume.cpu().numpy()
    volume_mf = np.zeros_like(volume)
    if volume.ndim == 4: # (b, c, h w)
        for i in range(len(volume)):
            for j in range(volume.shape[1]):
                volume_mf[i, j, :, :] = medfilt2d(volume[i, j, :, :], kernel_size=kernelsize)
    elif volume.ndim == 3: # (b, h, w)
        for i in range(len(volume)):
            volume_mf[i, :, :] = medfilt2d(volume[i, :, :], kernel_size=kernelsize)
    return torch.Tensor(volume_mf)

# Binary structure
def bin_dilation(volume, structure):
    volume_dil = volume.clone().cpu().numpy()
    volume = volume.cpu().numpy()
    for i in range(len(volume)):
        volume_dil[i] = binary_dilation(volume[i], structure=structure)
    return torch.Tensor(volume_dil)

class FAM():
    """
    Feature Attention Module to combine feature distance with pixel distance. 
    """
    def __init__(self, 
                 fe,
                 fe_layers=["layer1", "layer2"],
                 heatmap_v=1.):
        """
        Args: 
            fe: nn.Module: trained Feature Extractor network. 
            fe_layers: layers from FE to extract features from (to calculate the cosine similarity).
            heatmap_v: parameter to control the importance of pixel distance. 
        """
        self.heatmap_v = heatmap_v
        self.fe =fe
        self.fe.eval()
        self.fe_layers=fe_layers

    @torch.no_grad()
    def feature_distance(self, output, target, FE=None, fe_layers=None, device=None):
        """
        Feature distance between output and target
        Modify from https://github.com/arimousa/DDAD/blob/main/anomaly_map.py

        Args:
            output (torch.Tensor): Reconstructed image from the diffusion model, shape (B, C, H, W).
            target (torch.Tensor): Original image, shape (B, C, H, W).
            FE (torch.nn.Module): Feature extractor network (default to self.FE)
            fe_layers (list): list of feature layers to extracted from FE. Default to ["layer2", "layer3"]

        Returns:
            torch.Tensor: Feature distance between `output` and `target`, shape [B, 1, H, W]
        """

        if FE is None:
            FE = self.fe
        FE.eval()

        if device is None:
            device = next(FE.parameters()).device

        if fe_layers is None:
            fe_layers = self.fe_layers

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
            a_map = 1 - F.cosine_similarity(self.patchify(inputs_features[i]), self.patchify(output_features[i]))
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            anomaly_map += a_map
        return anomaly_map 

    @torch.no_grad()
    def patchify(self, features, return_spatial_info=False):
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

    def heat_map(self, 
                 output, 
                 target, 
                 FE=None, 
                 v=None, 
                 fe_layers=None, 
                 use_gaussian_blue=True, 
                 sigma=2., 
                 device=None):
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
        if FE is None:
            FE = self.fe
        if fe_layers is None:
            fe_layers = self.fe_layers
        if v is None:
            v = self.heatmap_v
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
        f_d = self.feature_distance(output,  target, FE, fe_layers=fe_layers, device=device)
        f_d = torch.Tensor(f_d).to(device)

        # combine i_d and f_d to generate anomaly_map
        anomaly_map = torch.zeros_like(f_d)

        # anomaly_map += f_d + v * (torch.max(f_d)/ torch.max(i_d)) * i_d  # scale i_d to f_d range
        anomaly_map += f_d * (torch.max(i_d)/ torch.max(f_d)) + v * i_d  # scale f_d to i_d range
        if use_gaussian_blue:
            anomaly_map = gaussian_blur2d(
                anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
                )
        anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
        anomaly_map = anomaly_map.float()
        return anomaly_map, f_d, i_d
    
    @torch.no_grad()
    def calculate_anomaly_score_map(self, xhat, x, FE=None, fe_layers=None, sigma=2.):
        """
        Calculate anomaly score map s from feature distance (f_d) and pixel distance (p_d)
        Args: 
            xhat: torch.Tensor(B, C, H, W): reconstructed images. 
            x: torch.Tensor(B, C, H, W): original images. 
            fe_layers: layer from FE network  to extract features from. Default to self.FE
            sigma: float: parameter of Gaussian blur 
        Returns:
            dict(): ano_map, f_d, i_d. 
        """

        if fe_layers is None:
            fe_layers = self.fe_layers
        if FE is None: 
            FE = self.fe
        ano_map, f_d, i_d = self.heat_map(
            xhat, 
            x, 
            self.fe, 
            v=self.heatmap_v, 
            fe_layers=fe_layers, 
            sigma=2.
        )

        return {
            "ano_map": ano_map,
            "f_d": f_d,
            "i_d": i_d
        }
    
    def plot_comparison(self):
        """
        Plot comparison with x, xhat, f_d, i_d and ano_score_map. 
        Default it will plot the most current results. 
        """
        return NotImplementedError
    

class LAFM():
    """
    Class holds LAFM methods to perform temporal smoothing operator. 
    """
    def __init__(self):
        pass

    def temp_filter_gaussian(self, ano_maps, time_idx, ages, sigma=1., use_normalize=True):
        """
        Perform temporal smoothing for 1 given time point (time_idx) based on reference. 
        """

        ano_map_ref = torch.cat((
            ano_maps[:time_idx],
            ano_maps[time_idx+1:]
        ))
        age_ref = torch.cat((
            ages[:time_idx],
            ages[time_idx+1:]
        ))
        diff_age_ref = torch.abs(ages[time_idx] - age_ref)

        # Gaussain weight decay
        if sigma != 0:
            weights = torch.exp(- (diff_age_ref ** 2) / (2 * sigma**2))
            if weights.sum() > 0:
                weights = weights / weights.sum()
            elif weights.sum() == 0:
                weights = weights = torch.ones_like(weights) / len(weights)
            
            ano_map_ref = ano_map_ref * rearrange(weights, "t -> t 1 1 1")

        # Smooth temporal anomaly maps referecne
        maxv = torch.max(ano_map_ref, dim=0)[0]
        meanv = torch.mean(ano_map_ref, dim=0)

        # anomaly_map = 0.0 * maxv + 1. * meanv + 0.5 * ano_map[time_idx].squeeze()
        ano_map_ref = 0.0 * maxv + 1. * meanv
        ano_map_ref_norm = norm_tensor(ano_map_ref)

        # update current anomaly map
        beta = 1.
        gamma = 0.0
        ano_map_new = ano_maps[time_idx] * (1 - beta * (1 - ano_map_ref_norm)) + gamma * ano_map_ref_norm

        # median filter and norm of ano_map_diff
        ano_map_mf = median_filter_2D(ano_map_new, kernelsize=3)
        if use_normalize:
            ano_map_mf = norm_tensor(ano_map_mf)

        return ano_map_mf, ano_map_ref
        
    def temp_filter_gaussian_loop(self, ano_maps, ages, sigma=1., use_normalize=True, nb_loop=1):
        ano_map_new = ano_maps.clone()
        ano_map_ref = ano_maps.clone()

        for loop in range(nb_loop):
            is_last = loop == nb_loop - 1
            for i in range(ano_maps.shape[0]):
                ano_map_new[i], ano_map_ref[i] = self.temp_filter_gaussian(ano_maps, 
                                                                    time_idx=i, 
                                                                    ages=ages, 
                                                                    sigma=sigma,
                                                                    use_normalize=is_last if use_normalize else False,)
            
            ano_maps = ano_map_new
        return ano_map_new, ano_map_ref

    def temp_filter_gaussian_one_patient(self, ano_map, age, i_d, mask=None, sigma=1.0, use_normalize=True):
        if mask is None:
            mask = torch.ones(ano_map.shape[0])
        ano_map = ano_map[mask == 1]
        age = age[mask == 1]
        i_d = i_d[mask == 1]

        ano_map_new = torch.zeros_like(ano_map)
        ano_map_ref = torch.zeros_like(ano_map)

        for i in range(ano_map.shape[0]):
            ano_map_new[i], ano_map_ref[i] = self.temp_filter_gaussian(ano_map, i, age, sigma=sigma, use_normalize=use_normalize)
        if torch.isnan(ano_map_new).any():
            print("nan value")

        # Yen threshold
        ano_map_new_raw = ano_map_new.clone()

        ano_map_new = norm_tensor(ano_map_new)
        thrs = threshold_yen(ano_map_new.numpy())
        thrs
        ano_map_yen_seg = torch.where(ano_map_new > thrs, 1., 0.)

        lafm_score = i_d * ano_map_new_raw
        i_d_new = norm_tensor(lafm_score)
        i_d_thrs = threshold_yen(i_d_new.numpy())
        i_d_yen_seg = torch.where(i_d_new > i_d_thrs, 1. , 0. )

        yen_seg = i_d_yen_seg * ano_map_yen_seg
        
        if not use_normalize:
            # Return the unormalized ano map
            ano_map_new = ano_map_new_raw
            
        return {
            "ano_map": ano_map,
            "ano_map_new": ano_map_new,
            "lafm_score": lafm_score,
            "ano_map_ref": ano_map_ref,
            "yen_seg": yen_seg
        }