import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import time
from tqdm import tqdm

from speaker_encoder import load_speaker_encoder, get_embedding_from_audio

def get_speaker_list(wav_dir):
    wav_files = os.listdir(wav_dir)
    speaker_list = []
    i = 0
    for wav_file in tqdm(wav_files):
        if i < 100 and i % 3 == 0:
            speaker_num = int(wav_file.split('-')[0])
            speaker_list.append(speaker_num)
        i += 1

    return speaker_list

def get_embeddings(wav_dir, encoder):
    wav_files = os.listdir(wav_dir)
    embs = None
    speaker_list = []
    i = 0
    for wav_file in tqdm(wav_files):
        if i < 100 and i % 3 == 0:
            speaker_num = int(wav_file.split('-')[0])
            speaker_list.append(speaker_num)
            wav_path = os.path.join(wav_dir, wav_file)
            emb = get_embedding_from_audio(wav_path, encoder)  # [1, 256]
            emb = emb.squeeze(0)    # [256]

            if embs is None:
                embs = emb
            else:
                embs = np.concatenate((embs, emb))

        i += 1

    np.save(f'embeddings_{len(embs)}.npy', embs)

    return embs, speaker_list


def get_2d_embeddings(embs):
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(embs)

    np.save(f'embeddings_2d_{len(embs)}.npy', embs_2d)

    return embs_2d

def plot_embeddings(embs, speaker_list):
    fig, ax = plt.subplots()
    ax.scatter(embs[:, 0], embs[:, 1], c=speaker_list)

    plt.show()


if __name__ == '__main__':
    encoder_path = 'speaker_encoder/3000000-BL.ckpt'
    wav_dir = 'wavs/test_data'

    encoder = load_speaker_encoder(encoder_path)

    wav_files = os.listdir(wav_dir)
    speaker_list = get_speaker_list(wav_dir)
    # embs, speaker_list = get_embeddings(wav_dir, encoder)
    # embs_2d = get_2d_embeddings(embs)
    embs_2d = np.load('embeddings_2d_34.npy')
    # embs = np.load('embeddings_262.npy')
    plot_embeddings(embs_2d, speaker_list)
    print(speaker_list)
    print(len(speaker_list))
