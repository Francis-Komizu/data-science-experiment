import torch
import numpy as np
from scipy.io.wavfile import write

from autovc.model_vc import Generator
from autovc.make_spect import get_spect_from_audio
from autovc.speaker_encoder import get_embedding_from_audio
from autovc.hifigan import load_hifigan, vocode, vocode_without_saving
from autovc.speaker_encoder import load_speaker_encoder, get_embedding_from_audio
from conversion import pad_seq

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_generator(checkpoint_path):
    G = Generator(32, 256, 512, 32).eval().to(device)
    g_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    G.load_state_dict(g_checkpoint['model'])

    return G


if __name__ == '__main__':
    # 参数配置
    generator_path = ''  # autoencoder模型
    hifigan_path = ''  # hifigan模型
    hifigan_config_path = ''  # hifigan配置路径
    speaker_encoder_path = ''  # speaker encoder路径
    src_wav_path = ''  # 源语音路径，注意采样率为16000，单声道
    trg_wav_path = ''  # 目标语者的语音路径，注意采样率为16000，单声道
    save_path = ''  # 生成语音的保存路径

    # 加载模型
    generator = load_generator(generator_path)
    hifigan = load_hifigan(hifigan_config_path, hifigan_path)
    speaker_encoder = load_speaker_encoder(speaker_encoder_path)

    # 获取语音梅尔普和speaker embedding
    x_org = get_spect_from_audio(src_wav_path)
    emb_org = get_embedding_from_audio(src_wav_path, speaker_encoder)
    emb_trg = get_embedding_from_audio(trg_wav_path, speaker_encoder)

    x_org, len_pad = pad_seq(x_org)

    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(emb_org[np.newaxis, :]).to(device)

    # 推理
    with torch.no_grad():
        _, x_identic_psnt, _ = generator(uttr_org, emb_org, emb_trg)

    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    # mel to wave
    audio = vocode_without_saving(uttr_trg, hifigan)

    # 保存生成语音
    write(save_path, 16000, audio)
