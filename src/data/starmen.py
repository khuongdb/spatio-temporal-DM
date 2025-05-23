import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class StarmenDataset(Dataset):
    def __init__(self, csv_path, nb_subject=None, data_mask=None):
        self.csv_path = csv_path

        if nb_subject:
            self.nb_subject = nb_subject
        else:
            self.nb_subject = 1_000
        self.datas = pd.read_csv(self.csv_path)
        self.ids = self.datas["id"].unique()

        if data_mask: 
            self.data_mask = data_mask
            self.datas['masked'] = False
            self.datas = self.datas.groupby('id', group_keys=False).apply(self.mask_rows)
            self.datas_unmasked = self.datas[~self.datas["masked"]]

    def mask_rows(self, group):
        random_probs = np.random.rand(len(group))
        mask = random_probs < self.data_mask
        group.loc[mask, 'masked'] = True
        return group

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        if self.data_mask:
            df = self.datas_unmasked
        else: 
            df = self.datas
        row = df.iloc[index]
        img_np = np.load(row["img_path"])
        id = row["id"]
        age = row["age"]
        return img_np, id, age
    

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
                    age = self.datas["age"].iloc[10*i+j]
                    axes[e][0].set_ylabel(f"Subject {subject_id}", rotation=0, labelpad=50, fontsize=10, va='center')
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
