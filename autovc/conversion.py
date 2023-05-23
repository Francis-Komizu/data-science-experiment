import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad


# TODO:从mel到code，再从code到mel，再生成wav的pipline

def load_generator(checkpoint_path):
    generator = Generator(32, 256, 512, 32).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model'])

    return generator


def get_codes_from_mel(uttr_org, emb_org, emb_trg, G):
    with torch.no_grad():
        encoder_outputs, codes = G.encode(uttr_org, emb_org, emb_trg)

    return encoder_outputs, codes


def get_mel_from_codes(encoder_outputs, len_pad, G):
    with torch.no_grad():
        x_outputs, x_outputs_postnet = G.decode(encoder_outputs)

    if len_pad == 0:
        uttr_trg = x_outputs_postnet[0, 0, :, :].cpu().numpy()
    else:
        uttr_trg = x_outputs_postnet[0, 0, :-len_pad, :].cpu().numpy()

    return uttr_trg


def get_average_emb(emb_org, emb_trg, weight=0.5, scale=1.0):
    emb_avg = scale * (emb_org * weight + emb_trg * (1 - weight))

    return emb_avg



if __name__ == '__main__':
    G = Generator(32, 256, 512, 32).eval().to(device)
    g_checkpoint = torch.load('autovc.ckpt', map_location=torch.device('cpu'))
    G.load_state_dict(g_checkpoint['model'])

    metadata = pickle.load(open('metadata.pkl', "rb"))

    spect_vc = []

    for sbmt_i in metadata:

        x_org = sbmt_i[2]
        x_org, len_pad = pad_seq(x_org)
        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
        emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)

        for sbmt_j in metadata:

            emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)

            with torch.no_grad():
                _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)

            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

            spect_vc.append(('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg))

    with open('results.pkl', 'wb') as handle:
        pickle.dump(spect_vc, handle)
