import scipy.signal
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

FRAME_SIZE = 4096
HOP_SIZE   = FRAME_SIZE // 2

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

def LTAS_third2(y, sr):
    # f, Pxx = scipy.signal.periodogram(y, fs=sr)
    # (f, Pxx) = scipy.signal.welch(y, fs=sr, window='blackmanharris', nperseg=FRAME_SIZE, noverlap=FRAME_SIZE-HOP_SIZE, average='mean', detrend=False)
    # window = scipy.signal.windows.blackmanharris(FRAME_SIZE, sym=True)
    window = scipy.signal.windows.hamming(FRAME_SIZE, sym=True)
    window = HOP_SIZE * window / window.sum()
    (f, t, Zxx) = scipy.signal.stft(x=y.astype(np.float128), fs=sr, window=window, nperseg=FRAME_SIZE, noverlap=FRAME_SIZE - HOP_SIZE, scaling='spectrum', padded=False, boundary=None)
    Zxx /= np.sqrt(1.0 / window.sum()**2)
    Pxx = np.mean(np.abs(Zxx/FRAME_SIZE)**2, axis=-1)
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

players = [3,4,5,6,7,8,11,12,14,16,17,18,19,21,22,24]
players = range(1, 17)
# Erreur code Victor : violonistes mal indexés !
feature = np.full((len(freqs), len(players)), fill_value=np.nan)
# feature = np.full((2049, 24), fill_value=np.nan)
# for player in range(26):
for (i, player) in enumerate(players):
    print(f'PLAYER {player}')
    file = f'../Data/SegmentationPerViolin/PLAYER{player}/Violin{5}.wav'
    if(os.path.exists(file)):
        y, sr = librosa.load(file, sr=44100, mono=False)
        feature[:, i] = LTAS_third2(y, sr)


plt.figure()
plt.step(freqs, feature, where='post')
plt.legend(np.arange(26)+1)
plt.xscale('log')
plt.xlim([176, 7000])
plt.ylim([-20, 40])
plt.show()