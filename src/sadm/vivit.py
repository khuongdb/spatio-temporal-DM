# ------------------------------------------------------------------------
# Modified from SADM (https://github.com/ubc-tea/SADM-Longitudinal-Medical-Image-Generation)
# ------------------------------------------------------------------------


import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

from src.sadm.vivit_module import Attention, PreNorm, FeedForward, Upsample


class Transformer(nn.Module):
    def __init__(self, dims, depth, heads, dim_head, dropout=0.0, direction="down"):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dims[-1])

        if direction == "down":
            code_func = FeedForward
        elif direction == "up":
            code_func = Upsample

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dims[i],
                            Attention(
                                dims[i], heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(
                            dims[i], FeedForward(dims[i], dims[i], dropout=dropout)
                        ),
                        code_func(dims[i], dims[i + 1], dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff1, ff2 in self.layers:
            x = attn(x) + x
            x = ff1(x) + x
            x = ff2(x)
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_frames,
        depth=3,
        heads=3,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        assert all(
            np.mod(image_size, patch_size) == 0
        ), "Image dimensions must be divisible by the patch size."

        assert len(image_size) in (
            2,
            3,
        ), "Image size need to be 2 (h, w) or 3 dimension (d, h, w)"

        # d = image_size[0] // patch_size[0]
        # h = image_size[0] // patch_size[0]
        # w = image_size[1] // patch_size[1]
        # the notation is difference from the actual paper
        # here: d, h, w refers to the number of patch per axis (not the size of patch)
        # equivalently: d = nd, h = nh, w = nw.
        dhw = np.divmod(image_size, patch_size)[0]
        if len(image_size) == 2:
            h, w = map(int, dhw)
            p1, p2 = patch_size
            reshape_fnc = Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=p1,
                p2=p2,
                h=h,
                w=w,
            )
        else:
            d, h, w = map(int, dhw)
            p1, p2, p3 = patch_size
            reshape_fnc = Rearrange(
                "b t c (d p1) (h p2) (w p3) -> b t (d h w) (p1 p2 p3 c)",
                p1=p1,
                p2=p2,
                p3=p3,
                d=d,
                h=h,
                w=w,
            )

        num_patches = int(np.prod(dhw))
        patch_dim = int(np.prod(patch_size))

        # 1-Image patch embedding - downscale the dimension of input
        # !TODO: make the down_scale the argument of model - more dynamic.
        self.to_patch_embedding = nn.Sequential(
            reshape_fnc,
            nn.Linear(patch_dim, patch_dim // 4),
        )

        # 2-Multihead spartial-temporal attention layers. 
        self.temporal_dims = [patch_dim // (4 ** (i + 1)) for i in range(depth + 1)]
        self.space_dims = [num_patches * num_frames for i in range(depth + 2)]

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, num_patches, patch_dim // 4)
        )
        self.temporal_transformer = Transformer(
            self.temporal_dims, depth, heads, dim_head, dropout, direction="down"
        )
        self.space_transformer = Transformer(
            self.space_dims, depth + 1, heads, dim_head, dropout, direction="up"
        )

        self.dropout = nn.Dropout(emb_dropout)

        # 3-Upscale using convolution network. 
        if len(image_size) == 3:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=d * h * w, mode="nearest"),
                Rearrange(
                    "b (p1 p2 p3) (d h w c) -> b c (d p1) (h p2) (w p3)",
                    p1=p1,
                    p2=p2,
                    p3=p3,
                    d=d,
                    h=h,
                    w=w,
                ),
                nn.Conv3d(
                    in_channels=self.space_dims[-1],
                    out_channels=1,
                    kernel_size=3,
                    padding="same",
                ),
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=h * w, mode="nearest"),
                Rearrange(
                    "b (p1 p2) (h w c) -> b c (h p1) (w p2)",
                    p1=p1,
                    p2=p2,
                    h=h,
                    w=w,
                ),
                nn.Conv2d(
                    in_channels=self.space_dims[-1],
                    out_channels=1,
                    kernel_size=3,
                    padding="same",
                ),
            )


    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, d = x.shape
        x += self.pos_embedding
        x = self.dropout(x)
        x = rearrange(x, "b t n d -> b (t n) d")
        x = self.temporal_transformer(x)
        x = rearrange(x, "b (t n) d -> b d (t n)", b=b, t=t, n=n)
        x = self.space_transformer(x)
        return self.conv(x)
