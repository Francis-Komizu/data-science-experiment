import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw


def plot_waveform(wav_path, sampling_rate=16000):
    # 加载语音文件
    y, sr = librosa.load(wav_path, sr=sampling_rate)

    # 绘制波形图
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()


def plot_linear_spectrogram(wav_path, sampling_rate=16000):
    # 加载语音文件
    y, sr = librosa.load(wav_path, sr=sampling_rate)

    # 计算短时傅里叶变换（STFT）
    stft = librosa.stft(y, n_fft=1024, hop_length=256, window='hamming')

    # 将线性谱转换为分贝刻度
    spectrogram = librosa.amplitude_to_db(abs(stft))

    # 绘制线性谱
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()


def plot_mel_spectrogram(wav_path, sampling_rate=16000):
    # 加载语音文件
    y, sr = librosa.load(wav_path, sr=sampling_rate)

    # 计算短时傅里叶变换（STFT）
    stft = librosa.stft(y, n_fft=1024, hop_length=256, window='hamming')

    # 将线性谱转换为梅尔谱
    mel_spec = librosa.feature.melspectrogram(S=abs(stft) ** 2,
                                              sr=sr,
                                              n_mels=128,
                                              n_fft=1024,
                                              hop_length=256)

    # 将梅尔谱转换为分贝刻度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 绘制梅尔频谱图
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()


def plot_fourier_transform_example():
    t = np.linspace(0, 1, 44100, endpoint=False)
    signal = 0.5 * np.cos(2 * np.pi * 220 * t + np.pi / 2) + \
             0.3 * np.cos(2 * np.pi * 440 * t + np.pi / 4) + \
             0.2 * np.cos(2 * np.pi * 880 * t + np.pi) + \
             0.1 * np.cos(2 * np.pi * 1760 * t - np.pi / 2) + \
             0.05 * np.cos(2 * np.pi * 3520 * t + np.pi / 3) + \
             0.03 * np.cos(2 * np.pi * 7040 * t + np.pi / 6)

    # 计算信号的傅里叶变换
    freqs = np.fft.fftfreq(len(signal), 1 / 44100)
    fft = np.fft.fft(signal)

    # 绘制信号的时域波形
    plt.figure(figsize=(14, 5))
    # plt.plot(t, signal)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Time-domain signal')
    # plt.show()
    # # 绘制信号的频域谱图
    # plt.subplot(2, 1, 2)
    plt.plot(freqs[:len(freqs) // 2], abs(fft[:len(freqs) // 2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency-domain spectrum')
    plt.tight_layout()
    plt.show()


from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt


def compute_cosine_similarity(x, y):
    numerator = np.dot(x, y.T)  # 分子
    denominator = np.linalg.norm(x) * np.linalg.norm(y)  # 分母
    similarity = numerator / denominator  # 余弦相似度
    return similarity


def compute_cosine_similarity_by_index(seq1, seq2, indexes1, indexes2):
    similarities = []
    assert len(indexes1) == len(indexes2)

    for i in range(len(indexes1)):
        vector1 = seq1[indexes1[i]]
        vector2 = seq2[indexes2[i]]
        similarity = compute_cosine_similarity(vector1, vector2)
        similarities.append(similarity)

    return similarities


def plot_similarities(similarities):
    length = len(similarities)
    max = np.ones(length)
    plt.plot(similarities)
    plt.plot(max, linestyle='--')
    plt.title('Similarities')
    plt.ylabel('Similarity')
    plt.xlabel('Time')
    plt.show()


if __name__ == '__main__':
    # wav_path = 'resources/audios/welcome_converted.wav'
    # plot_waveform(wav_path)
    # # plot_linear_spectrogram(wav_path)
    # plot_mel_spectrogram(wav_path)

    seq1 = np.load('samples/units/libri_units.npy', allow_pickle=True)
    seq2 = np.load('samples/units/user_units.npy', allow_pickle=True)
    print(seq1.shape)
    print(seq2.shape)
    d, C, D1, path = dtw(seq1, seq2, dist=lambda x, y: np.linalg.norm(x - y))

    indexes1, indexes2 = path[0], path[1]
    print(len(path[0]))
    similarities = compute_cosine_similarity_by_index(seq1, seq2, indexes1, indexes2)
    print(len(similarities))
    print(similarities)
    plot_similarities(similarities)