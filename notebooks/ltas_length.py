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
SIZES       = [1*SR, 10*SR, 30*SR, 60*SR]

# Data
df = pd.read_pickle('recordings.pkl')
df = df[df['violin'].isin([13])]

# Features
data = []
for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    y, _ = librosa.load(str(row['file']), sr=SR)
    for size in SIZES:
        for audio in np.split(y, np.arange(size, len(y), size)):
            # _, features = 10 * np.log10(scipy.signal.welch(audio, nperseg=FRAME_SIZE, average='mean'))
            _, features = scipy.signal.welch(audio, nperseg=FRAME_SIZE, average='mean')
            dic = dict(zip(np.arange(len(features)), features))
            dic.update(size=size)
            data.append(dic)

features = pd.DataFrame(data)

freqs = np.arange(FRAME_SIZE//2+1) / (FRAME_SIZE//2+1) * SR/2
mean = features.groupby(['size']).mean()
std = features.groupby(['size']).std()

# Plot
fig, ax = plt.subplots(figsize=(16,9))
# ax.bar(freqs, scores, align='center', width=20)
for size in range(mean.shape[0]):
    ax.plot(freqs, mean.iloc[size], label='_nolegend_')
    ax.fill_between(
        freqs,
        np.maximum(mean.iloc[size] - std.iloc[size], 0),
        mean.iloc[size] + std.iloc[size],
        alpha=0.3,
    )
ax.set_xlim([200, 6000])
# ax.set_ylim([-65, -30])
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('LTAS')
ax.legend([f'{size//SR}s' for size in SIZES])
plt.savefig('figures/ltas_length.png', bbox_inches='tight', dpi=300)
plt.show()