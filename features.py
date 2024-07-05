import librosa
import numpy as np
import scipy.signal, scipy.fftpack

def LTCC(y, frame_size, hop_size, n_coeff):
    Y = librosa.stft(y=y, n_fft=frame_size, hop_length=hop_size).T
    S = 20 * np.log10(np.abs(Y) + 1e-10)
    ltcc = np.fft.irfft(S)
    # ltcc = scipy.fftpack.dct(S)
    return ltcc[:,1:n_coeff+1]

def LTAS(y, frame_size, hop_size):
    _, ltas = scipy.signal.welch(y, nperseg=frame_size, noverlap=frame_size-hop_size, average='median')
    ltas = 10 * np.log10(ltas)
    return ltas.T

def MFCC(y, sr, frame_size, hop_size, n_coeff):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coeff, n_fft=frame_size, hop_length=hop_size)
    return mfcc.T

def LTAS_third(y, sr, frame_size, hop_size):
    y = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_size).T
    CENTER_FREQ = 1000
    freqs = CENTER_FREQ * np.power(2**(1/3), np.arange(-18, 7))
    # freqs = 7000 * np.power(2**(1/3), np.arange(-16, 1))
    # freqs = CENTER_FREQ * 2.0 ** (np.arange(-1, 9)-5)
    freqs_lower = freqs / 2**(1/6)
    freqs_upper = freqs * 2**(1/6)
    LTAS = np.empty((y.shape[0], freqs.shape[0]))
    for k in range(len(freqs)):
        lower = freqs_lower[k]
        upper = freqs_upper[k]
        # factor = ((sr / 2) / (freqs_upper/2**(1/6)))
        # print(np.round(len(y) / factor))
        # sd = scipy.signal.resample(y, np.round(len(y) / factor))
        sos = scipy.signal.butter(N=3, Wn=np.array([lower, upper])/(sr/2), btype='bandpass', analog=False,output='sos')
        band = scipy.signal.sosfiltfilt(sos, y)
        LTAS[:, k] = 10 * np.log10(np.sum(np.power(band,2), axis=-1))
    # print(LTAS)
    LTAS -= LTAS[:, 0:1]
    return LTAS

# def LTAS_third2(y, sr):
#     freqs = 7000 * np.power(2**(1/3), np.arange(-16, 1))
#     f, Pxx = scipy.signal.periodogram(y, fs=sr)
#     LTAS = np.empty(freqs.shape)
#     for k in range(len(freqs) - 1):
#         ind_min = np.argmax(f > freqs[k]) - 1
#         ind_max = np.argmax(f > freqs[k+1]) - 1
#         LTAS[k] = np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
#     return LTAS

freqs = 7000 * np.power(2**(1/3), np.arange(-16, 1))

def gauss(freq, freqs, noct):
    sigma = (freq / noct) / np.pi
    g = np.exp(- (freqs - freq)**2 / (2 * sigma**2))
    g /= np.sum(g)
    return g

def smooth(Pxx, freqs, noct=12):
    SPxx = np.zeros(Pxx.shape)
    SPxx[:, 0] = Pxx[:, 0]
    for (i, freq) in enumerate(freqs):
        if i != 0:
            g = gauss(freq, freqs, noct)
            SPxx[:, i] = np.sum(g * Pxx, axis=-1)
    SPxx[SPxx < 0] = 0
    return SPxx

def LTAS_third2(y, sr, frame_size, hop_size):
    # f, Pxx = scipy.signal.periodogram(y, fs=sr)
    # (f, Pxx) = scipy.signal.welch(y, fs=sr, window='blackmanharris', nperseg=FRAME_SIZE, noverlap=FRAME_SIZE-HOP_SIZE, average='mean', detrend=False)
    # window = scipy.signal.windows.blackmanharris(FRAME_SIZE, sym=True)
    window = scipy.signal.windows.hamming(frame_size, sym=True)
    window = hop_size * window / window.sum()
    (f, t, Zxx) = scipy.signal.stft(x=y.astype(np.float128), fs=sr, window=window, nperseg=frame_size, noverlap=frame_size - hop_size, scaling='spectrum', padded=False, boundary=None)
    Zxx /= np.sqrt(1.0 / window.sum()**2)
    Pxx = np.mean(np.abs(Zxx/frame_size)**2, axis=-1)
    Pxx = smooth(Pxx, f)
    Pxx = np.mean(Pxx, axis=0)
    # return Pxx

    LTAS = np.empty(len(freqs))
    for k in range(len(freqs) - 1):
        ind_min = np.argmin(np.abs(freqs[k] - f))
        ind_max = np.argmin(np.abs(freqs[k+1] - f))
        # ind_min = np.argmax(f > freqs[k]) - 1
        # ind_max = np.argmax(f > freqs[k+1]) - 1
        # LTAS[k] = np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
        LTAS[k] = np.mean(Pxx[ind_min: ind_max+1])
    LTAS /= LTAS[0]
    LTAS = 10 * np.log10(LTAS)
    return LTAS


# from transformers import Wav2Vec2FeatureExtractor
# from transformers import AutoModel
# import torch
# from torch import nn
# import torchaudio.transforms as T
# from datasets import load_dataset

# model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
# resample_rate = processor.sampling_rate

# def MERT(y):
#     inputs = processor(y, sampling_rate=resample_rate, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)

#     all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
#     time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
#     return time_reduced_hidden_states.numpy().reshape(-1)