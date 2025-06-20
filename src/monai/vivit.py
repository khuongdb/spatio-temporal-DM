# ------------------------------------------------------------------------
# Modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vivit.py



import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn
import time
import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler
import shutil
from src.sadm.utils import convert_sec_hms
import os
from src.data import StarmenDataset
from monai.data import DataLoader
from monai.utils import set_determinism

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class FactorizedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        b, f, n, _ = x.shape
        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
            x = temporal_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

        return self.norm(x)

class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        num_classes=None,
        pool = 'mean',
        channels = 3,
        out_channels = 1, 
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        variant = 'factorized_encoder',
        reduce_dim = False,
        out_upsample = False
    ):
        super().__init__()

        assert not (reduce_dim and out_upsample), 'reduce_dim and upsample cannnot be True at the same time. Please only choose 1.'
        self.reduce_dim = reduce_dim
        self.out_upsample = out_upsample

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        h = image_height // patch_height
        w = image_width // patch_width
        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'factorized_encoder':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        elif variant == 'factorized_self_attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.num_classes = num_classes
        if num_classes: 
            self.mlp_head = nn.Linear(dim, num_classes)

        self.variant = variant

        if out_upsample:
        # Upsample to match the original input images
            self.upsample = nn.Sequential(
                # Rearrange('b (h w) d -> b d h w', h=h, w=w),
                nn.Upsample(scale_factor=patch_height * patch_width, mode="nearest"),
                Rearrange(
                        "b d (h w p1 p2) -> b d (h p1) (w p2)",
                        p1=patch_height,
                        p2=patch_width,
                        h=h,
                        w=w,
                    ),
                nn.Conv2d(
                        in_channels=dim,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding="same",
                    ),

            )


    def forward(self, video):
        """
        Transformer to get the (latent) representation of multiple frames. 
        There are 2 possible outcomes, depend on how we want to use the context (as input to condition ddpm model)
        
        - Cross Attention context: the context need to be of shape [B, 1, D]
            with D: dimension of cross-attention (covariate condition).
        
        - Concaternation context: the context need to be of same shape as x [B, C, H, W]. 
            In this case, ViVit will upsample the output to match dimensions of a frame. 
            Return [B, C, H, W]
        """
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape

        x = x + self.pos_embedding[:, :f, :n]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        if self.variant == 'factorized_encoder':
            x = rearrange(x, 'b f n d -> (b f) n d')

            # attend across space

            x = self.spatial_transformer(x)
            x = rearrange(x, '(b f) n d -> b f n d', b = b)

            # In the original implementation, we use global pooling for exercise out the cls token. 

            if self.reduce_dim:  
            # excise out the spatial cls tokens or average pool for temporal attention
                x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')
                # append temporal CLS tokens
                if exists(self.temporal_cls_token):
                    temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)
                    x = torch.cat((temporal_cls_tokens, x), dim = 1)
                
                # attend across time
                x = self.temporal_transformer(x)

                # excise out temporal cls token or average pool (across all frames)
                x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

            elif self.out_upsample: 
                # Here we want to keep the same number of dimension [b f n d] after both spartial and temporal attention layer.
                # and then upsampling to match the original input [b c (f df) (h dh) (w dw)]
                x = rearrange(x, 'b f n d -> (b n) f d') 

                # attend across time
                x = self.temporal_transformer(x)

                x = rearrange(x, '(b n) f d -> b n f d', b=b)
                x - reduce(x, 'b n f d -> b n d', 'mean')

                # upsample
                x = rearrange(x, 'b n d -> b d n')
                x = self.upsample(x)  
            
            else:
                NotImplementedError


        elif self.variant == 'factorized_self_attention':
            x = self.factorized_transformer(x)
            x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

        x = self.to_latent(x)
        if self.num_classes:
            x = self.mlp_head(x)

        return x
    

def vivit_train(
    workdir=None,
    model: nn.Module=None, 
    train_loader=None,
    val_loader=None,
    device="cpu",
    logger=None,
    num_train_timesteps=1000,
    learning_rate=2.5e-5,
    n_epochs=100,
    val_interval=10,
    sample_interval=50,
    autoresume=False,
    checkpoint_path=None,
):
    """
    Train Vivit model 
    Takes a sequence of input x_prev = [B, C, T, H, W], the model output prediction of next frame x_pred.
    There is 2 possible outcomes: 
        - Output representation of x_pred = [B, 1, dim]
        - Output x_pred directly (upsample) x_pred = [B, C, H, W]
    """
    import os

    from src.sadm.utils import get_logger
    from src.utils.networks import load_checkpoint

    if not logger:
        logger = get_logger("train", workdir=workdir, mode="a")


    # make workdir folder
    ckpt_dir = f"{workdir}/ckpt"
    sample_dir = f"{workdir}/train_samples"
    for dir in [ckpt_dir, sample_dir]:
        os.makedirs(dir, exist_ok=True)

    model.to(device)
    logger.info("Model:")
    logger.info(model)

    # Autoresume from checkpoint
    if autoresume:
        # Load model checkpoint
        # First, the model looks at checkpoint_path
        # if not found then try to find latest.pth at workdir/ckpt/
        if checkpoint_path:
            ckpt_path = checkpoint_path
        else: 
            ckpt_path = os.path.join(workdir, "ckpt/latest.pth")
        if not os.path.exists(ckpt_path):
            raise NameError(f"Checkpoint does not exist at {ckpt_path}")

        model, optimizer, info = load_checkpoint(ckpt_path, model)

        if info.get("epoch", None):
            start_ep = info["epoch"] + 1
            print(f"Resume training from epoch {start_ep}")
        else: 
            start_ep = 0

        if info.get("best_val_loss", None):
            best_val_loss = info["best_val_loss"]
            best_val_epoch = info["best_val_epoch"]
        else:
            best_val_loss = float("inf")
            best_val_epoch = -1
    else: 
        start_ep = 0
        best_val_loss = float('inf')
        best_val_epoch = -1

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Optimizer: {optimizer}")

    ep_loss_list = []
    val_ep_loss_list = []

    scaler = GradScaler(device=device)
    total_start = time.time()

    for ep in range(start_ep, n_epochs):
        model.trains()
        ep_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {ep}")
        for step, batch in progress_bar:
            x_prev = batch["x_prev"].to(device)
            x = batch["x"].to(device)
            optimizer.zero_grad(set_to_none=True)

            x_pred = model(x_prev)
            loss = F.mse_loss(x.float(), x_pred.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()

            progress_bar.set_postfix({"loss": ep_loss / (step + 1)})

        logger.info(f"Epoch: {ep} - loss: {ep_loss / (step + 1)}")
        ep_loss_list.append(ep_loss / (step + 1))

        # Evaluation during training
        if (ep + 1) % val_interval == 0:
            model.eval()
            val_ep_loss = 0
            for step, batch in enumerate(val_loader):
                x_prev = batch["x_prev"].to(device)
                x = batch["x"].to(device)
                with torch.no_grad():
                    x_pred = model(x_prev)
                    val_loss = F.mse_loss(x_pred.float(), x.float())

                val_ep_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_ep_loss / (step + 1)})


            avg_val_loss = val_ep_loss / (step + 1)
            logger.info(f"Epoch: {ep} - val_loss: {avg_val_loss}")
            val_ep_loss_list.append(avg_val_loss)

            # Update best validation metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_epoch = ep

        # Save latest.pth model after each epoch.
        # if the current model has the best val metrics - also save to best.pth
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "info": {
                    "epoch": ep,
                    "best_val_loss": best_val_loss,
                    "best_val_epoch": best_val_epoch
                }
            },
            os.path.join(ckpt_dir, "latest.pth"),
        )

        if best_val_epoch == ep:
            shutil.copyfile(
                os.path.join(ckpt_dir, "latest.pth"),
                os.path.join(ckpt_dir, "best.pth")
            )
            print(f"Save new best model at epoch: {ep:03} - avg_val_loss: {avg_val_loss}")
    
    total_time = time.time() - total_start
    h, m, s = convert_sec_hms(total_time)
    logger.info(f"Train completed, total time: {h:02}:{m:02}:{s}.")



def main(args=None):

    DATA_DIR = "data/starmen/output_random_noacc"

    workdir = args.workdir
    experiment = args.experiment
    if workdir is None:
        if experiment is not None: 
            workdir = f"workdir/{experiment}"
        else: 
            workdir = "workdir"

    os.makedirs(workdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.job_type =="train":
        # Load train and val dataset
        train_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="train",
            nb_subject=None,
            save_data=False,
            workdir=workdir,
        )
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1)

        val_ds = StarmenDataset(
            data_dir=DATA_DIR,
            split="val",
            nb_subject=None,
            save_data=False,
            workdir=workdir,
        )
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=True, num_workers=1)

        # define model
        model = ViT(
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


        vivit_train(
            workdir=workdir,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            # logger=logger,
            num_train_timesteps=1000,
            learning_rate=1.5e-5,
            n_epochs=1000,
            val_interval=10,
            sample_interval=20,
            autoresume=args.autoresume,
            checkpoint_path=args.checkpoint
        )

    else:
        NotImplementedError


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job_type",
        type=str,
        default="train",
        help="Type of job: 'train' or 'inference'",
    )

    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="working directory to store checkpoints, samples, logs.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="sadm",
        help="Name of experience. If workdir is not defined, new directory workdir/<experiment> will be created. ",
    )

    parser.add_argument(
        "--autoresume",
        action="store_true",
        help="Autoresume training from checkpoint. Checkpoint will be located at <workdir>/ckpt/latest.pt",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to load checkpoint if --autoresume is True. If not set, the model will look for latest.pth or best.pth at <workdir>/ckpt/",
    )


    args = parser.parse_args()

    set_determinism(42)
    main(args=args)
