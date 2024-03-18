import librosa
import numpy as np

def LTCC(y, frame_size, hop_size, n_coeff):
    Y = librosa.stft(y=y, n_fft=frame_size, hop_length=hop_size).T
    S = 20 * np.log10(np.abs(Y) + 0.00001)
    ltcc = np.fft.irfft(S)[:,:n_coeff]
    return ltcc