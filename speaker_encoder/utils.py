import torch
from collections import OrderedDict
import sys
sys.path.append('speaker_encoder')

from model_bl import D_VECTOR
from make_spect import get_spect_from_audio

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_speaker_encoder(model_path):
    speaker_encoder = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval()
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for key, val in checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    speaker_encoder.load_state_dict(new_state_dict)

    return speaker_encoder


def get_embedding_from_audio(wav_path, encoder):
    spect = get_spect_from_audio(wav_path)  # [N, 80]
    spect = torch.from_numpy(spect)
    spect = spect.unsqueeze(0)  # [1, N, 80]

    emb = encoder(spect)    # [1, 256]

    emb = emb.detach()

    return emb


if __name__ == '__main__':
    pass


