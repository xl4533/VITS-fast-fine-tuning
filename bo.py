# -*- coding: gb2312 -*-
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

plt.figure(dpi=600) # ����ʾ������ͼ�ֱ��ʵ���
matplotlib.rc("font",family='SimHei') # ��ʾ����
matplotlib.rcParams['axes.unicode_minus']=False # ��ʾ����


def displayWaveform(): # ��ʾ����ʱ����
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load('C:\\Users\\xiaolu\\Desktop\\VITS-fast-fine-tuning-main\\custom_character_voice-ori\\ccc.wav', sr=16000)
    # samples = samples[6000:16000]

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.plot(time, samples)
    plt.title("�����ź�ʱ����")
    plt.xlabel("ʱ�����룩")
    plt.ylabel("���")
    # plt.savefig("your dir\�����ź�ʱ����ͼ", dpi=600)
    plt.show()

def displaySpectrum(): # ��ʾ����Ƶ������
    x, sr = librosa.load(r'your wav file path', sr=16000)
    print(len(x))
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # ��fft�Ľ��ֱ��ȡģ��ȡ����ֵ�����õ�����magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)

    ft = fft(x)
    print(len(ft), type(ft), np.max(ft), np.min(ft))
    magnitude = np.absolute(ft)  # ��fft�Ľ��ֱ��ȡģ��ȡ����ֵ�����õ�����magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)

    print(len(magnitude), type(magnitude), np.max(magnitude), np.min(magnitude))
    print(len(frequency), type(frequency), np.max(frequency), np.min(frequency))

    # plot spectrum���޶�[:40000]
    # plt.figure(figsize=(18, 8))
    plt.plot(frequency[:40000], magnitude[:40000])  # magnitude spectrum
    plt.title("�����ź�Ƶ������")
    plt.xlabel("Ƶ�ʣ����ȣ�")
    plt.ylabel("����")
    plt.savefig("your dir\�����ź�Ƶ��ͼ", dpi=600)
    plt.show()

    # # plot spectrum�����޶� [�Գ�]
    # plt.figure(figsize=(18, 8))
    # plt.plot(frequency, magnitude)  # magnitude spectrum
    # plt.title("�����ź�Ƶ������")
    # plt.xlabel("Ƶ�ʣ����ȣ�")
    # plt.ylabel("����")
    # plt.show()


def displaySpectrogram():
    x, sr = librosa.load(r'your wav file path', sr=16000)

    # compute power spectrogram with stft(short-time fourier transform):
    # ����stft������power spectrogram
    spectrogram = librosa.amplitude_to_db(librosa.stft(x))

    # show
    librosa.display.specshow(spectrogram, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('�����źŶ�����ͼ')
    plt.xlabel('ʱ�����룩')
    plt.ylabel('Ƶ�ʣ����ȣ�')
    plt.show()


if __name__ == '__main__':
    displayWaveform()
    displaySpectrum()
    displaySpectrogram()