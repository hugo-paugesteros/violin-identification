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
config = {
    'sr': 10000,
    'frame_size': 1024*2,
    'hop_size': 1024*2,
    'n_coeff': 80,
    'size': 10
}
PLAYERS = [3,4,5,6,7,8,11,12,14,16,17,18,19,21,22,24]
SIZE = 500 * config['sr']

# Data
df = pd.read_pickle('recordings.pkl')
# df = df[df['violin'].isin([13])]
df = df[df['type'] == 'scale']
df = df[df['player'].isin(PLAYERS)]

# Features
data = []
for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    y, _ = librosa.load(str(row['file']), sr=config['sr'])
    for audio in np.split(y, np.arange(SIZE, len(y), SIZE)):

        features = y
        for step in pipes['MFCC_librosa']:
            features = step(features, **config)
        features = features[:config['n_coeff']]
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

mean = features.groupby(['violin']).mean() / 1
mean = mean - np.min(mean)
std = features.groupby(['violin']).std() / 1

# Plot
fig, ax = plt.subplots(figsize=(16,9))
ax.bar(np.arange(len(scores)), scores, align='center', width=.5)
for violin in range(mean.shape[0]):
    # ax.plot(np.arange(len(scores)), mean.iloc[violin])
    ax.fill_between(
        np.arange(len(scores)),
        mean.iloc[violin] - std.iloc[violin],
        mean.iloc[violin] + std.iloc[violin],
        alpha=0.2
    )
# ax.set_xlim([200, 6000])
# ax.set_ylim([0, 65])
# ax.set_xscale('log')
# ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('MFCC')
plt.savefig('figures/mfcc_select.png', bbox_inches='tight', dpi=300)
plt.show()