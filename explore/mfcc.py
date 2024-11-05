import pandas as pd
import librosa
import numpy as np
from features import *
import tqdm
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('styles.mplstyle')

# Params
SR          = 10000
FRAME_SIZE  = 2048
HOP_SIZE    = FRAME_SIZE
N_COEFF     = FRAME_SIZE//2+1
PLAYERS = [3,4,5,6,7,8,11,12,14,16,17,18,19,21,22,24]

# Data
df = pd.read_pickle('data/processed/dataset_bilbao.pkl')
df = df[df['type'] == 'scale']
df = df[df['violin'].isin(PLAYERS)]

y, _ = librosa.load(df['file'].iloc[0], sr=SR)

# Feature
f, ltas = scipy.signal.welch(y, fs=SR, nperseg=FRAME_SIZE, average='median')
ltas = ltas[:, np.newaxis]
S = librosa.feature.melspectrogram(S=ltas, sr=SR, n_mels=100)
S = 10 * np.log10(S)

mfcc = librosa.feature.mfcc(S=S, n_mfcc=2049, n_mels=100)
mfcc[75:] = 0
Sm = scipy.fftpack.idct(mfcc, axis=-2, type=2, norm='ortho')
Sml = np.power(10, Sm/10)
envelope = librosa.feature.inverse.mel_to_stft(Sml, sr=SR, power=1.0)

ltas = 10 * np.log10(ltas)
envelope = 10 * np.log10(envelope)

# Plot
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(f, ltas, label='LTAS')
ax.plot(f, envelope, label='Envelope (MFCCs)')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (dB)')
ax.set_xscale('log')
ax.set_xlim([200, 4000])
ax.legend()
plt.savefig('figures/mfcc.png', bbox_inches='tight', dpi=300)
plt.show()