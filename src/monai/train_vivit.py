import torch
from src.monai.vivit import ViT

v = ViT(
    image_size = 64,          # image size
    frames = 9,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 1,      # frame patch size
    # num_classes = 1000,
    dim = 512,
    spatial_depth = 6,         # depth of the spatial transformer
    temporal_depth = 6,        # depth of the temporal transformer
    heads = 8,
    mlp_dim = 512,
    variant = 'factorized_encoder', # or 'factorized_self_attention'
    reduce_dim = False  # dont perform global pooling or exercise cls token. 
)

video = torch.randn(4, 3, 9, 64, 64) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)
print(preds.shape)