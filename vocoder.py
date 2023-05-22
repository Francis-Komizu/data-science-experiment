from hifigan.utils import load_vocoder, vocode
import torch
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    hifigan_config_path = 'hifigan/config_hifigan_v1.json'
    hifigan_model_path = 'hifigan/g_03280000'
    vocoder, h = load_vocoder(hifigan_config_path, hifigan_model_path)

    spect_vc = pickle.load(open('results.pkl', 'rb'))

    for spect in spect_vc:
        name = spect[0]
        c = spect[1]
        wav_path = f'wavs/{name}.wav'
        vocode(c, wav_path, vocoder, h)



