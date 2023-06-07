import glob
import os
import json
import torch
from scipy.io.wavfile import write

from autovc.hifigan.models import Generator as HiFiGAN

MAX_WAV_VALUE = 32768.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def load_vocoder(config_path, checkpoint_path):
    with open(config_path) as f:
        data = f.read()

    config = json.loads(data)
    h = AttrDict(config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vocoder = HiFiGAN(h).to(device)

    state_dict_g = load_checkpoint(checkpoint_path, device)
    vocoder.load_state_dict(state_dict_g['generator'])

    vocoder.eval()
    vocoder.remove_weight_norm()

    return vocoder, h


def vocode(input_spect, output_path, vocoder):
    with torch.no_grad():
        x = torch.FloatTensor(input_spect).to(device)
        if x.shape[1] == 80:
            x = torch.transpose(x, 0, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        assert x.shape[1] == 80
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        write(output_path, 16000, audio)
        print('Successfully saved audio to {}'.format(output_path))


def vocode_without_saving(input_spect, vocoder):
    with torch.no_grad():
        x = torch.FloatTensor(input_spect).to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if x.shape[2] == 80:
            x = torch.transpose(x, 1, -1)
        assert x.shape[1] == 80
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

    return audio


if __name__ == '__main__':
    pass
