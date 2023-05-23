import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch


def decompose_codes(codes, n_components=2):
    # codes []
    pca = PCA(n_components=n_components)
    pca.fit(codes)

    return pca.components_


def decompose_emb(emb_org, emb_trg, emb_avg, n_components=2):
    # emb [1, 256]
    emb_org, emb_trg, emb_avg = emb_org.squeeze(0), emb_trg.squeeze(0), emb_avg.squeeze(0)
    embs = torch.stack([emb_org, emb_trg, emb_avg], dim=0)
    print(embs.shape)
    pca = PCA(n_components=n_components)
    embs_2d = pca.fit_transform(embs)

    return embs_2d[0], embs_2d[1], embs_2d[2]


def plot_2d_codes(codes):
    pass


def plot_2d_emb(emb_org, emb_trg, emb_avg):
    fig, ax = plt.subplots()
    ax.scatter(emb_org[0], emb_org[1], color='red', label='source speaker')
    ax.scatter(emb_trg[0], emb_trg[1], color='blue', label='targe speaker')
    ax.scatter(emb_avg[0], emb_avg[1], color='orange', label='average')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    pass