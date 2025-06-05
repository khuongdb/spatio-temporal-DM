
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_multi_imgs(imgs, labels, opt=None):
    if not isinstance(imgs, list):
        imgs = [imgs]


    processed_imgs = []
    for img in imgs:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        processed_imgs.append(img)

    imgs = processed_imgs


    fig, axes = plt.subplots(len(imgs), 11, figsize=(22, 2 * len(imgs)))
    ws = opt.get("wspace", 0)
    hs = opt.get("hspace", 0) 
    
    plt.subplots_adjust(wspace=ws, hspace=hs)

    color = opt.get("cmap", "gray")
    for i, i_imgs in enumerate(imgs):
        axes[i][0].axis('off')  
        axes[i][0].text(0.5, 0.5, labels[i],
                    fontsize=12, ha='center', va='center')
        for j, img in enumerate(i_imgs):
            if not np.all(img == 0): 
                axes[i][j + 1].matshow(img, aspect="equal", cmap=color)

    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])