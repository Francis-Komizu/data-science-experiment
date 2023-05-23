from conversion import *
from hifigan.utils import *
from visual_aids import *
import pickle
from scipy.io.wavfile import write
from speaker_encoder import get_embedding_from_audio, load_speaker_encoder


if __name__ == '__main__':
    # hparams
    weight = 0.7
    scale = 1.0
    encoder_path = 'speaker_encoder/3000000-BL.ckpt'
    generator_path = 'autovc.ckpt'
    vocoder_path = 'hifigan/g_03280000'
    vocoder_config_path = 'hifigan/config_hifigan_v1.json'

    # wav_path_rnd = 'wavs/random.wav'

    metadata_path = 'metadata.pkl'
    metadata = pickle.load(open(metadata_path, "rb"))
    # f-225, f-228, m-256, m-270
    sbmt_org = metadata[0]
    sbmt_trg = metadata[3]

    # wav_path_org = f'wavs/exp1/{sbmt_org[0]}_{sbmt_trg[0]}_org.wav'
    # wav_path_trg = f'wavs/exp1/{sbmt_org[0]}x{sbmt_trg[0]}_trg.wav'
    wav_path_avg = f'wavs/exp1/{sbmt_org[0]}x{sbmt_trg[0]}_{weight}_{scale}.wav'

    # encoder = load_speaker_encoder(encoder_path)
    generator = load_generator(generator_path)
    vocoder, h = load_vocoder(vocoder_config_path, vocoder_path)

    # 获取input
    print('Convert from {} to {}.'.format(sbmt_org[0], sbmt_trg[0]))
    x_org = sbmt_org[2]  # 输入的mel
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

    # speaker embeddings
    emb_org = torch.from_numpy(sbmt_org[1][np.newaxis, :]).to(device)
    # emb_trg = get_embedding_from_audio('wavs/test_data/10-300600004.wav', encoder)
    emb_trg = torch.from_numpy(sbmt_trg[1][np.newaxis, :]).to(device)

    # 修改speaker embeddings
    # print(emb_trg.shape)    # [1, 256]
    emb_avg = get_average_emb(emb_org, emb_trg, weight, scale)

    # emb_diff = emb_trg - emb_org

    # 可视化speaker embeddings
    # emb_org_2d, emb_trg_2d, emb_avg_2d = decompose_emb(emb_org, emb_trg, emb_avg)

    # plot_2d_emb(emb_org_2d, emb_trg_2d, emb_avg_2d)

    # 获取codes
    # encoder_outputs, codes = get_codes_from_mel(uttr_org, emb_org, emb_trg, generator)
    encoder_outputs_avg, codes_avg = get_codes_from_mel(uttr_org, emb_org, emb_avg, generator)
    # encoder_outputs_rnd, codes_rnd = get_codes_from_mel(uttr_org, emb_org, emb_diff, generator)
    # 可视化codes


    # 修改codes


    # 获取mel
     #uttr_trg = get_mel_from_codes(encoder_outputs, len_pad, generator)
    uttr_avg = get_mel_from_codes(encoder_outputs_avg, len_pad, generator)
    # uttr_rnd = get_mel_from_codes(encoder_outputs_rnd, len_pad, generator)

    # 获取waveform
    print(uttr_org.shape)
    # wav_org = vocode_without_saving(uttr_org, vocoder)
    # wav_trg = vocode_without_saving(uttr_trg, vocoder)
    wav_avg = vocode_without_saving(uttr_avg, vocoder)


    # write(wav_path_org, 16000, wav_org)
    # write(wav_path_trg, 16000, wav_trg)
    write(wav_path_avg, 16000, wav_avg)


