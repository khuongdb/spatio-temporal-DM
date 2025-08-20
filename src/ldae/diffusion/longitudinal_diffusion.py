from leaspy.algo import AlgorithmSettings
from leaspy.io.data import Data, Dataset
from leaspy.models import LinearModel, LogisticModel
import torch
import pandas as pd
import numpy as np
import os
from einops import rearrange

class LongitudinalDiffusion:
    """
    Main class implementing the longitudinal diffusion model. 

    This class handles various operation related to diffusion model in conjuntion with leaspy
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

    @torch.no_grad()
    def create_leaspy_dataset(self, ids, ages, feats, limit=None):
        """
        Create dataFrame for leaspy
        Args: 
            ids: np.ndarray(b * t) | list[np.ndarray or torch.Tensor]
            ages: np.ndarray(b * t) np.ndarray(b * t) | list[np.ndarray or torch.Tensor]
            ysems: np.ndrayy(b*t, dim) np.ndarray(b * t) | list[np.ndarray or torch.Tensor]
        """

        if isinstance(feats, np.ndarray) or isinstance(feats, torch.Tensor):
            _, d = feats.shape
            feature_cols = [f'dim_{i}' for i in range(d)]
            data = feats

        elif isinstance(feats, list): 
            *_, d = feats[0].shape
            rows = [t.float().numpy() for t in feats]
            data = np.vstack(rows)
            feature_cols=[f"dim_{i}" for i in range(d)]

        else:
            raise TypeError(f"Wrong input type of features: {type(feats)}")
        
        # Sanity check


        ids_flat = np.hstack(ids)
        ages_flat = np.hstack(ages)

        df = pd.DataFrame(data, columns=feature_cols)
        df.index = pd.MultiIndex.from_arrays([ids_flat, ages_flat], names=["ID", "TIME"])

        if limit is not None:
            ids_to_keep = df.index.get_level_values(0).unique()[:limit]
            # Filter the DataFrame
            df = df[df.index.get_level_values(0).isin(ids_to_keep)]
        else:
            df = df

        data = Data.from_dataframe(df)
        dataset = Dataset(data)
        return df, data, dataset
    
    def init_leaspy_model(self, enc_args, leaspy_path=None):
        model_leaspy = LogisticModel(name="logistic", source_dimension=enc_args["emb_chans"] - 1)
        if leaspy_path is not None: 
            model_leaspy = self.load_leaspy_model(model_leaspy, leaspy_path)
        model_leaspy.move_to_device(self.device)
        return model_leaspy

    def load_leaspy_model(self, model_leaspy, leaspy_path):
        """
        Load pretrained leaspy model
        """
        if not model_leaspy.is_initialized:
            if not os.path.exists(leaspy_path):
                raise ValueError(f"No pretrained leaspy model found at {leaspy_path}. Initialize new leaspy model.")
            else: 
                with torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=False):
                    model_leaspy = model_leaspy.load(leaspy_path)
        return model_leaspy
    
    def mask_input_data(self, 
                        x, 
                        age, 
                        perc_seen=0.2, 
                        miss_past=2, 
                        miss_future=3):
        """
        Mask original input data
        Args:
            x (torch.Tensor(B, T, C, H, W)): original sequence of images.
            age: (torch.Tensor(B, T)): orginal ages
            pers_seen (float): persentage of seen images for each sequence. 
            miss_past (int): number of missing images from the part. 
            miss_future (int): number of missing images into the future. 
        Return:
            out (dict) contains 
                x_seen (b, t, c, h, w), 
                age(b, t),
                idx_seen (b, t)
        """
        b, t, c, h, w = x.shape

        # perc_seen = 0.25

        miss_past = 2
        miss_future = 3

        seen_mask = torch.full((b, t), 0.)
        for i in range(b):
            seen_mask_i = seen_mask[i][miss_past:(t-miss_future)]
            while torch.all(seen_mask_i == 0.):
                seen_mask_i = torch.bernoulli(torch.full((t - miss_past - miss_future, ), perc_seen))
            
            seen_mask[i][miss_past:(t-miss_future)] = seen_mask_i

        x_seen = x * rearrange(seen_mask, "b t-> b t 1 1 1")
        age_seen = age * (seen_mask)
        return {
            "x_seen": x_seen,
            "age_seen": age_seen,
            "idx_seen": seen_mask
        }
    
    @torch.no_grad()
    def extract_semantic_encoder_one_batch(self, ids, x, age, mask, encoder, use_age_cond=False):
        """
        Extract semantic representation from trained encoder. 
        """
        b, t, *_ = x.shape

        ages_all = []
        ids_all = []
        ysem_all = []

        if mask is None: 
            mask = torch.full((b, t), 1.)

        for i in range(b):

            ## TIME (age)
            age_i = age[i][mask[i].bool()].detach().cpu()
            ages_all.append(age_i)

            ## ID
            p_ids = ids[i].cpu().numpy()
            p_ids = np.repeat(p_ids, len(age_i))
            ids_all.append(p_ids.astype("str"))

            ## semantic encoder
            x_i = x[i][mask[i].bool()].to(self.device)
            # x_flat = rearrange(x_i, "t ... -> (b t) ...")
            age_flat = rearrange(age_i, "b -> b 1").to(self.device)
            if use_age_cond:
                ysem = encoder(x_i, age_flat)
            else:
                ysem = encoder(x_i)
            ysem_all.append(ysem.detach().cpu())

        return {
            "ids": ids_all,
            "ages": ages_all,
            "features": ysem_all
        }

    @torch.no_grad()
    def extract_semantic_encoder_dataloader(self, dataloader, encoder, limit=None, use_age_cond=False):
        """
        Extract semantic representation from the whole dataloader.
        """
        if limit is None:
            limit = len(dataloader)

        for i, batch in enumerate(dataloader):
            out_ysem = self.extract_semantic_encoder_one_batch(
                ids=batch["id"],
                x=batch["x_origin"],
                age=batch["age"],
                mask=None,
                encoder=encoder,
                use_age_cond=use_age_cond
            )

            df_batch, _, _ = (
                self.create_leaspy_dataset(
                    ids=out_ysem["ids"],
                    ages=out_ysem["ages"],
                    feats=out_ysem["features"],
                )
            )

            if i == 0: 
                df = df_batch
            else: 
                df = pd.concat((df, df_batch))

            if i == limit - 1: 
                break

        data = Data.from_dataframe(df)
        dataset = Dataset(data)
        return df, data, dataset
    
    def impute_missing_data(self):
        """
        Impute missing data from a sequence of timepoints.
        """
