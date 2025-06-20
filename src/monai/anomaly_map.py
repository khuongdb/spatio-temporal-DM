import numpy as np
import lpips
import torch
import cv2
from einops import rearrange


class AnomalyMap():
    def __init__(self):

        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

        self.lpips = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True,
                                     spatial=True, lpips=True).to(self.device)
        super(AnomalyMap, self).__init__()

    def dilate_masks(self, masks):
        """
        :param masks: masks to dilate
        :return: dilated masks
        """
        kernel = np.ones((3, 3), np.uint8)

        dilated_masks = torch.zeros_like(masks)
        for i in range(masks.shape[0]):
            mask = masks[i][0].detach().cpu().numpy()
            if np.sum(mask) < 1:
                dilated_masks[i] = masks[i]
                continue
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            dilated_mask = torch.from_numpy(dilated_mask).to(masks.device).unsqueeze(dim=0)
            dilated_masks[i] = dilated_mask

        return dilated_masks

    def compute_residual(self, x_rec, x, hist_eq=False):
        """
        :param x_rec: reconstructed image
        :param x: original image
        :param hist_eq: whether to perform histogram equalization
        :return: residual image
        """
        # if hist_eq:
        #     x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
        #     x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
        #     x_res = np.abs(x_rec_rescale - x_rescale)
        # else:
        x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())

        return x_res

    def lpips_loss_old(self, anomaly_img, ph_img, retPerLayer=False):
        """
        :param anomaly_img: anomaly image
        :param ph_img: pseudo-healthy image
        :param retPerLayer: whether to return the loss per layer
        :return: LPIPS loss
        """
        if len(ph_img.shape) < 2:
            print('Image should have 2 dimensions at lease (LPIPS)')
            return
        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        if len(ph_img.shape) == 3:
            ph_img = torch.unsqueeze(ph_img, 0)
            anomaly_img = torch.unsqueeze(anomaly_img, 0)

        saliency_maps = []
        for batch_id in range(anomaly_img.size(0)):
            lpips = self.lpips(2*anomaly_img[batch_id:batch_id + 1, :, :, :]-1, 2*ph_img[batch_id:batch_id + 1, :, :, :]-1,
                                   normalize=True, retPerLayer=retPerLayer)
            if retPerLayer:
                lpips = lpips[1][0]
            saliency_maps.append(lpips[0,:,:,:].cpu().detach().numpy())
        return np.asarray(saliency_maps)
    
    def lpips_loss(self, anomaly_img, ph_img, retPerLayer=False):
        """
        :param anomaly_img: anomaly image
        :param ph_img: pseudo-healthy image
        :param retPerLayer: whether to return the loss per layer
        :return: LPIPS loss
        """
        # b, c, h, w = anomaly_img.shape

        if len(ph_img.shape) < 2:
            print('Image should have 2 dimensions at lease (LPIPS)')
            return
        if len(ph_img.shape) == 2:  # assuming [H, W]
            ph_img = rearrange(ph_img, "h w -> 1 1 h w")
            anomaly_img = rearrange(anomaly_img, "h w -> 1 1 h w")
        if len(ph_img.shape) == 3:  # assuming [C, H, W]
            ph_img = rearrange(ph_img, "c h w -> 1 c h w")
            anomaly_img = rearrange(anomaly_img, "c h w -> 1 c h w")

        saliency_maps = self.lpips(anomaly_img, ph_img, normalize=True, retPerLayer=retPerLayer)

        return saliency_maps.detach().cpu().numpy()
    

def main():
    ano_map = AnomalyMap()

    im0 = torch.randn(10, 1, 64, 64)
    im1 = torch.randn(10, 1, 64, 64)

    loss_fn = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True,
                                     spatial=True, lpips=True)
    # d1 = loss_fn.forward(im0,im1).detach().cpu().numpy()
    
    d1 = ano_map.lpips_loss(im0, im1)

    d2 = ano_map.lpips_loss_old(im0, im1)
    diff = d1 - d2
    eps = 1e-2
    

    print(d1.shape)
    




if __name__ == "__main__":
    main()