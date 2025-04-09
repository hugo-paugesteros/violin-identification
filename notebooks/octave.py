import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import PyOctaveBand
import os

def LTCC(y, frame_size, hop_ratio, n_coeff):
    Y = librosa.stft(y=y, n_fft=frame_size, hop_length=frame_size // hop_ratio).T
    S = 20 * np.log10(np.abs(Y))
    ltcc = np.fft.irfft(S)[:,2:n_coeff+2]
    return ltcc

CENTER_FREQ = 1000
freqs = CENTER_FREQ * np.power(2**(1/3), np.arange(-18, 13))
freqs = 7000 * np.power(2**(1/3), np.arange(-16, 1))
# freqs = CENTER_FREQ * 2.0 ** (np.arange(-1, 9)-5)
freqs_lower = freqs / 2**(1/6)
freqs_upper = freqs * 2**(1/6)
print(freqs)

def LTAS_octave_bands2(y):
    LTAS = np.empty(freqs.shape)
    for k in range(len(freqs)):
        lower = freqs_lower[k]
        upper = freqs_upper[k]
        # factor = ((sr / 2) / (freqs_upper/2**(1/6)))
        # print(np.round(len(y) / factor))
        # sd = scipy.signal.resample(y, np.round(len(y) / factor))
        sos = scipy.signal.butter(N=3, Wn=np.array([lower, upper])/(sr/2), btype='bandpass', analog=False,output='sos')
        band = scipy.signal.sosfiltfilt(sos, y)
        LTAS[k] = 10 * np.log10(np.sum(np.power(band,2)))

    LTAS -= LTAS[0]
    return LTAS

def LTAS_octave_bands(y):
    LTAS, freq, xb = PyOctaveBand.octavefilter(y, fs=sr, fraction=3, order=3, limits=[176, 7000], sigbands=1)
    # print(freq)
    for i in range(len(xb)):
        LTAS[i] = 10 * np.log10(np.sum(np.power(xb[i],2)))
    # LTAS = 10 * np.log10(LTAS)
    LTAS -= LTAS[0]
    return LTAS

def LTAS_third2(y, sr):
    freqs = 7000 * np.power(2**(1/3), np.arange(-16, 1))
    f, Pxx = scipy.signal.periodogram(y, fs=sr)
    LTAS = np.empty(freqs.shape)
    for k in range(len(freqs) - 1):
        ind_min = np.argmax(f > freqs[k]) - 1
        ind_max = np.argmax(f > freqs[k+1]) - 1
        LTAS[k] = np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
    return LTAS


plt.figure()

n_coeff = 40
LTCC_all = np.full((len(freqs), 24), fill_value=np.nan)
LTCC_all2 = np.full((len(freqs), 24), fill_value=np.nan)
LTCC_all = np.full((n_coeff, 24), fill_value=np.nan)
LTCC_all2 = np.full((n_coeff, 24), fill_value=np.nan)
for player in range(24):
    print(f'PLAYER {player}')
    file = f'../Data/SegmentationPerViolin/PLAYER{player+1}/Violin{5}.wav'
    if(os.path.exists(file)):
        y, sr = librosa.load(file, sr=44100)
        # LTCC_all[:, player] = LTAS_octave_bands(y)
        LTCC_all[:, player] = np.mean(LTCC(y, 4096*32, 2, n_coeff), axis=0)
    file = f'../Data/SegmentationPerViolin/PLAYER{player+1}/Violin{2}.wav'
    if(os.path.exists(file)):
        y, sr = librosa.load(file, sr=44100)
        # LTCC_all2[:, player] = LTAS_octave_bands(y)
        LTCC_all2[:, player] = np.mean(LTCC(y, 4096*32, 2, n_coeff), axis=0)

# plt.step(freqs, LTAS_all, where='post')
# plt.step(np.arange(n_coeff), np.nanmean(LTCC_all, axis=1), where='post')
# plt.step(np.arange(n_coeff), np.nanmean(LTCC_all2, axis=1), where='post')
# plt.legend(np.arange(24)+1)
# plt.xscale('log')
# plt.xlim([176, 7000])

data1 = {
    'x': np.arange(n_coeff),
    'y': np.nanmean(LTCC_all, axis=1),
    'yerr': np.nanstd(LTCC_all, axis=1)
}
data2 = {
    'x': np.arange(n_coeff),
    'y': np.nanmean(LTCC_all2, axis=1),
    'yerr': np.nanstd(LTCC_all2, axis=1)
}
for data in [data1, data2]:
    plt.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]
    }
    plt.fill_between(**data, alpha=.25)
plt.show()
