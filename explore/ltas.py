import pandas as pd
import librosa
import numpy as np
from src.features import *
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import sklearn.feature_selection

plt.style.use('styles.mplstyle')

# Params
SR          = 22050
FRAME_SIZE  = 2048
HOP_SIZE    = FRAME_SIZE
N_COEFF     = FRAME_SIZE//2+1
# N_COEFF     = 75
PLAYERS = [3,4,5,6,7,8,11,12,14,16,17,18,19,21,22,24]
SIZE = 500 * SR

# Data
df = pd.read_pickle('recordings.pkl')
# df = df[df['violin'].isin([13])]
df = df[df['type'] == 'scale']
df = df[df['player'].isin(PLAYERS)]

# Features
data = []
for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    y, _ = librosa.load(str(row['file']), sr=SR)
    for audio in np.split(y, np.arange(SIZE, len(y), SIZE)):
        _, features = 10 * np.log10(scipy.signal.welch(audio, nperseg=FRAME_SIZE, average='mean'))
        # features = scipy.fftpack.dct(features, norm='ortho')
        # features[50:] = 0
        # features = scipy.fftpack.idct(features, norm='ortho')
        # features = 10 * np.log10(LTAS(audio, FRAME_SIZE, HOP_SIZE, N_COEFF))
        # features = np.mean(10 * np.log10(LTAS(audio, FRAME_SIZE, HOP_SIZE, N_COEFF)), axis=0)
        # features = 10 * np.log10(np.mean(LTAS(audio, FRAME_SIZE, HOP_SIZE), axis=0))
        # features = np.mean(LTCC(audio, FRAME_SIZE, HOP_SIZE, N_COEFF), axis=0)
        # features = np.mean(MFCC(audio, SR, FRAME_SIZE, HOP_SIZE, N_COEFF), axis=0)
        # features = np.mean(
        #     librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_COEFF, n_fft=FRAME_SIZE, hop_length=HOP_SIZE),
        #     axis=1,
        #     keepdims=True
        # )
        # features = librosa.feature.inverse.mfcc_to_mel(features)
        # features = librosa.feature.inverse.mel_to_stft(features, sr=SR, n_fft=FRAME_SIZE)
        # features = 10 * np.log10(features[:, 0])
        N_COEFF = len(features)
        
        dic = dict(zip(np.arange(len(features)), features))
        dic.update(violin=row['violin'])
        data.append(dic)

        # for i in range(features.shape[0]):
        #     dic = dict(zip(list(range(features.shape[1])), features[i].T))
        #     dic.update(violin=row['violin'])
        #     data.append(dic)

features = pd.DataFrame(data)
features.to_pickle('features.pkl')
features = pd.read_pickle('features.pkl')

X = features.iloc[:, :N_COEFF]
y = features['violin']

selection = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, k='all')
selection.fit_transform(X, y)
scores = -np.log10(selection.pvalues_)

freqs = np.arange(N_COEFF) / N_COEFF * SR/2
mean = features.groupby(['violin']).mean()
mean = mean + 60
std = features.groupby(['violin']).std()

# Plot
fig, ax = plt.subplots(figsize=(16,9))
ax.bar(freqs, scores, align='center', width=20)
for violin in range(mean.shape[0]):
    # ax.plot(freqs, mean.iloc[violin])
    ax.fill_between(
        freqs,
        mean.iloc[violin] - std.iloc[violin],
        mean.iloc[violin] + std.iloc[violin],
        alpha=0.2
    )
ax.set_xlim([200, 6000])
ax.set_ylim([0, 65])
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('LTAS (dB)')
plt.savefig('figures/ltas.svg', bbox_inches='tight', dpi=300)
plt.show()