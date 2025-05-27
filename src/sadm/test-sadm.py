# %% [markdown]
# # Dataset
# 
# 

# %%
import os

train_infos = "/home/kdang/projects/spartDM/data/starmen/output_random_noacc/starmen_train.csv"
os.chdir("/home/kdang/projects/spartDM/")
print("Current working directory:", os.getcwd())

# %%
from torch.utils.data import DataLoader

from src.data import StarmenDataset

starmen = StarmenDataset(train_infos)

train_loader = DataLoader(starmen, batch_size=2)

x, x_prev = next(iter(train_loader))

# %%
import numpy as np

from src.sadm.vivit import ViViT

vivit = ViViT(
    image_size = (64, 64),
    patch_size = (16, 16),
    num_frames = 9,
    depth = 3,
    heads = 3,
    dim_head=8
)

# %%
from einops.layers.torch import Rearrange
from torch import nn

patch_size = (16, 16) 
p1, p2 = patch_size
h, w = (4, 4)
patch_dim = int(np.prod(patch_size))


x_prev_emb = vivit.to_patch_embedding(x_prev)
x_prev_emb.shape

vivit.pos_embedding.shape

c = vivit(x_prev)


