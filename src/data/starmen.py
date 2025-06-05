import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from einops import rearrange
import torch
import logging

class StarmenDataset(Dataset):
    def __init__(self, data_dir, split="train", time=10, nb_subject=None, **options):
        self.csv_path = os.path.join(data_dir, f"starmen_{split}.csv")

        self.datas = pd.read_csv(self.csv_path)
        self.ids = self.datas["id"].unique()
        self.time = time

        if nb_subject:
            selected_ids = np.random.choice(self.ids, size=nb_subject, replace=False)
            self.datas = self.datas[self.datas["id"].isin(selected_ids)]
            self.ids = selected_ids

            logging.info(f"{split} dataset - only load {nb_subject} subjects.")
            logging.info(f"Subject id: {self.ids}")

    def prepare_mask(self):

        target_idx = np.random.randint(1, self.time)
        missing_mask = [1]
        if target_idx > 1:
            if target_idx > 2:
                missing_mask = np.append(
                    missing_mask, np.random.randint(0, 2, size=(target_idx - 2,))
                )
            missing_mask = np.append(missing_mask, [1])
        missing_mask = np.append(
            missing_mask, np.zeros(self.time - len(missing_mask))
        ).astype(np.float32)
        return target_idx, missing_mask

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Generates one sample of data: concatenate all data of 1 patient into 1 tensor.
        Return:
        x: target image, shape [C, H, W]
        x_prev: previous images, shape [T-1, C, H, W]
        """

        index_id = self.ids[index]
        img_t = self.get_images_by_id(id=index_id)

        # Random choose a target index and create missing mask
        target_idx, mask = self.prepare_mask()
        x_prev = np.clip(img_t[:-1] * mask[:-1, None, None, None], 0.0, 1.0)
        x = img_t[target_idx]

        item = {
            "id": index_id,
            "x": torch.from_numpy(x).float(),
            "x_prev": torch.from_numpy(x_prev).float(),
            "x_origin": torch.from_numpy(img_t).float(),
            "target_idx": target_idx,
            "mask": mask,
        }
        return item
        # return torch.from_numpy(x).float(), torch.from_numpy(x_prev).float(), torch.from_numpy(img_t).float(), target_idx, mask

    def get_images_by_id(self, id):
        """
        Get a sequence of 10 images for a given id in Dataset.
        Return: np.array (t, c, h, w)
        """
        df = self.datas[self.datas["id"] == id]
        df.sort_values(by="age")

        img_ls = [np.load(i) for i in df["img_path"]]
        img_t = np.stack(img_ls, axis=0)
        # Add channel dimension c
        img_t = rearrange(img_t, "t h w -> t 1 h w")
        return img_t

    def plot_data(self, index=0, save=False, show_info=False):
        """
        Plot a sequence (10 items) images of subject for given index(s)
        :param index: int or list[int]
        """

        if isinstance(index, int):
            index = [index]

        fig, axes = plt.subplots(len(index), 10, figsize=(20, 2 * len(index)))
        plt.subplots_adjust(wspace=0, hspace=0)
        for e, i in enumerate(index):
            for j in range(10):
                img_np = np.load(self.datas["img_path"][10 * i + j])
                axes[e][j].matshow(255 * img_np)
                # axes[2 * i + 1][j].matshow(255 * indiv_tra[10 * i + j][0].cpu().detach().numpy())

                # Additional infos
                if show_info:
                    subject_id = self.datas["id"].iloc[10 * i]
                    age = self.datas["age"].iloc[10 * i + j]
                    axes[e][0].set_ylabel(
                        f"Subject {subject_id}",
                        rotation=0,
                        labelpad=50,
                        fontsize=10,
                        va="center",
                    )
                    axes[e][j].set_title(f"age: {age}")

        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        # if not os.path.exists('visualization'):
        #     os.mkdir('visualization')
        # plt.savefig('visualization/Z.png', dpi=300, bbox_inches='tight')


def main():
    pass


if __name__ == "__main__":
    main()
