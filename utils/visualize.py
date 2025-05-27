import torch
from torch.utils import data
from dataset import Data_preprocess_starmen, Dataset_starmen
from torch.autograd import Variable
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import logging
import time
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # load model
    autoencoder = torch.load('model/0_fold_UOMETM', map_location=device)
    autoencoder.eval()

    # load train and test data
    data_generator = Data_preprocess_starmen()
    test_data = data_generator.generate_all()
    test_data.requires_grad = False

    Dataset = Dataset_starmen
    test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                   test_data['timepoint'], test_data['first_age'], test_data['alpha'])

    test_loader = torch.utils.data.DataLoader(test, batch_size=50, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=True)

    # get Z, ZU, ZV
    with torch.no_grad():
        Z, ZU, ZV = None, None, None
        for data in test_loader:
            image = torch.tensor([[np.load(path)] for path in data[0]], device=device).float()
            input_ = Variable(image).to(device)
            reconstructed, z, zu, zv = autoencoder.forward(input_)
            self_reconstruction_loss = autoencoder.loss(input_, reconstructed)

            # store Z, ZU, ZV
            if Z is None:
                Z, ZU, ZV = z, zu, zv
            else:
                Z = torch.cat((Z, z), 0)
                ZU = torch.cat((ZU, zu), 0)
                ZV = torch.cat((ZV, zv), 0)

    # calculate psi
    psi = test_data['alpha'] * (test_data['age'] - test_data['baseline_age'])
    psi_array = np.linspace(min(psi), max(psi), num=5)
    index = [np.nonzero(np.abs(np.array(psi) - p) < 0.05)[0][:2] for p in psi_array]
    index = [j for i in index for j in i]

    # individual trajectory
    subject = [i // 10 for i in index]
    subject_img = []
    for s in subject:
        subject_img += list(np.arange(s * 10, (s + 1) * 10))
    path = test_data.iloc[subject_img, 0]
    image = torch.tensor([[np.load(p)] for p in path], device=device).float()
    indiv_tra, _, _, _ = autoencoder.forward(image)

    fig, axes = plt.subplots(2 * len(index), 10, figsize=(20, 2 * 2 * len(index)))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(index)):
        for j in range(10):
            axes[2 * i][j].matshow(255 * image[10 * i + j][0].cpu().detach().numpy())
            axes[2 * i + 1][j].matshow(255 * indiv_tra[10 * i + j][0].cpu().detach().numpy())
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    if not os.path.exists('visualization'):
        os.mkdir('visualization')
    plt.savefig('visualization/Z.png', dpi=300, bbox_inches='tight')

    # ZU
    zu = ZU[subject_img]
    global_tra = autoencoder.decoder(zu)

    fig, axes = plt.subplots(2 * len(index), 10, figsize=(20, 2 * 2 * len(index)))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(index)):
        for j in range(10):
            if j == 0:
                last = global_tra[10 * i + j]
                axes[2 * i][j].matshow(255 * global_tra[10 * i + j][0].cpu().detach().numpy())
            else:
                axes[2 * i][j].matshow(255 * global_tra[10 * i + j][0].cpu().detach().numpy())
                axes[2 * i + 1][j].matshow((global_tra[10 * i + j] - last)[0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                            norm=mcolors.Normalize(vmin=-1, vmax=1))
                last = global_tra[10 * i + j]
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('visualization/ZU.png', dpi=300, bbox_inches='tight')

    # individual heterogeneity
    random.seed(int(time.time()))
    random.shuffle(index)
    path = test_data.iloc[index, 0]
    image = torch.tensor([[np.load(p)] for p in path], device=device).float()
    zv = ZV[index]
    indiv_hetero = autoencoder.decoder(zv)

    fig, axes = plt.subplots(3, len(index), figsize=(2 * len(index), 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(index)):
        if i == 0:
            last = indiv_hetero[i]
            axes[0][i].matshow(255 * image[i][0].cpu().detach().numpy())
            axes[1][i].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
        else:
            axes[0][i].matshow(255 * image[i][0].cpu().detach().numpy())
            axes[1][i].matshow(255 * indiv_hetero[i][0].cpu().detach().numpy())
            axes[2][i].matshow((indiv_hetero[i] - last)[0].cpu().detach().numpy(),
                                cmap=matplotlib.cm.get_cmap('bwr'),
                                norm=mcolors.Normalize(vmin=-1, vmax=1))
            last = indiv_hetero[i]
    for axe in axes:
        for ax in axe:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig('visualization/ZV.png', dpi=300, bbox_inches='tight')

    plt.close()