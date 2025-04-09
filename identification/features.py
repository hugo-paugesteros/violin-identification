import librosa
import numpy as np
import scipy.signal, scipy.fftpack


def pop(S, **kwargs):
    return S[1:]


def db(S, **kwargs):
    """Concert a spectrogram in dB scale

    Args:
        S (nd.array): Spectrogram in linear scale

    Returns:
        nd.array: Spectrogram in dB scale
    """
    return 10 * np.log10(S)


def mel(S, frame_size, sr, n_mels, **kwargs):
    melfb = librosa.filters.mel(sr=sr, n_fft=frame_size, n_mels=n_mels)
    return melfb.dot(S)


def LTAS_stft(y, frame_size, hop_size, **kwargs):
    ltas = librosa.stft(y=y, n_fft=frame_size, hop_length=hop_size).T
    ltas = np.abs(ltas) ** 2
    ltas = np.median(ltas, axis=0)
    return ltas


def LTAS(y, frame_size, hop_size, **kwargs):
    _, ltas = scipy.signal.welch(
        y, nperseg=frame_size, noverlap=frame_size - hop_size, average="mean"
    )
    return ltas


def IRFFT(S, **kwargs):
    return np.fft.irfft(S)


def LTCC(y, frame_size, hop_size, **kwargs):
    S = LTAS(y, frame_size, hop_size)
    ltcc = np.fft.irfft(S)
    return ltcc


def MFCC(S, frame_size, hop_size, sr, n_coeff, **kwargs):
    S = np.expand_dims(S, axis=0).T
    mfcc = librosa.feature.mfcc(
        S=S, sr=sr, n_mfcc=n_coeff, n_fft=frame_size, hop_length=hop_size
    ).T
    mfcc = np.median(mfcc, axis=0)
    return mfcc


def MFCC_librosa(y, frame_size, hop_size, sr, n_coeff, **kwargs):
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_coeff, n_fft=frame_size, hop_length=hop_size
    ).T
    mfcc = np.mean(mfcc, axis=0)
    return mfcc


# Booooh, pas beau, vilain code
import kymatio
from kymatio.numpy import Scattering1D

# SHAPE = 2**19
# kwargs = {"J": 10, "shape": SHAPE, "Q": (3, 1), "T": "global"}
# sc = Scattering1D(**kwargs)


# def scattering(y, **kwargs):
#     pad = lambda a, i: (
#         a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
#     )
#     y = pad(y, SHAPE)
#     Y = np.squeeze(sc(y))
#     Y = np.log10(np.abs(Y))
#     return Y


pipes = {
    "LTAS_welch": [LTAS],
    "LTAS_stft": [LTAS_stft],
    "LTAS_welch_db": [LTAS, pop, db],
    "LTCC_welch": [LTAS, pop, db, IRFFT],
    "LTCC_stft": [LTAS_stft, db, IRFFT, pop],
    "MFCC_welch": [LTAS, pop, db, MFCC],
    "MFCC_stft": [LTAS_stft, db, MFCC],
    "MFCC_librosa": [MFCC_librosa, pop],
    "MEL_welch": [LTAS, pop, mel, db],
    "MFCC_welch_mel": [LTAS, db, mel, MFCC, pop],
    # "scattering": [scattering],
}
