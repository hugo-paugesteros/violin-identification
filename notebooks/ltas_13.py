import pandas as pd
import librosa
import numpy as np
from features import *
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import os

plt.style.use('styles.mplstyle')

# Params
SR          = 10000
FRAME_SIZE  = 2048
HOP_SIZE    = FRAME_SIZE
N_COEFF     = FRAME_SIZE//2+1
PLAYERS = [3,4,5,6,7,8,11,12,14,16,17,18,19,21,22,24]

# Data
df = pd.read_pickle('recordings.pkl')
df = df[df['type'] == 'scale']
df = df[df['violin'].isin(PLAYERS)]

# Feature
data = []
for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    y, _ = librosa.load(str(row['file']), sr=SR, mono=False)
    features = LTAS_third2(y, SR, FRAME_SIZE, HOP_SIZE)

    dic = dict(zip(np.arange(len(features)), features))
    dic.update(violin=row['violin'])
    data.append(dic)

features = pd.DataFrame(data)
features.to_pickle('features.pkl')
features = pd.read_pickle('features.pkl')

mean = features.groupby(['violin']).mean()
std = features.groupby(['violin']).std()

# Plot
fig, ax = plt.subplots(figsize=(16,9))
for violin in range(mean.shape[0]):
    # ax.step(freqs, mean.iloc[violin], where='post')
    ax.fill_between(
        freqs,
        mean.iloc[violin] - std.iloc[violin],
        mean.iloc[violin] + std.iloc[violin],
        alpha=0.2,
        step='post'
    )
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (dB)')
ax.set_xscale('log')
ax.set_ylim([-20, 30])
plt.savefig('figures/ltas_13.png', bbox_inches='tight', dpi=300)
plt.show()