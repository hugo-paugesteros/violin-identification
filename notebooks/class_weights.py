import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import librosa
import numpy as np

plt.style.use('/home/hugo/Thèse/styles.mplstyle')
plt.rcParams['image.composite_image'] = False

# Data
# df = pd.read_pickle('data/processed/dataset_bilbao.pkl')
# df = df[df['player'].isin(list(range(30)))]
# df = df[df.player != 15]
# df = df[df['player'].isin([30,31,32,99]) & (df.type == "villefavard")]

df = pd.read_pickle('data/processed/dataset_cnsm.pkl')
df = df[df.violin.isin(['A', 'B', 'C'])]

# Compute duration
def get_row_duration(row):
    # row['duration'] = librosa.get_duration(path=row['file']) / 60
    row['duration'] = (row['end'] - row['start']) / 60
    return row
df = df.apply(get_row_duration, axis=1)

duration_sum = df.groupby(['violin', 'player']).sum().unstack('player', fill_value=0)['duration']

# Plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5,4), sharey='row', sharex='col', gridspec_kw=dict(height_ratios=[1, 3], width_ratios=[3, 1]))

im = ax3.imshow(duration_sum.to_numpy(), aspect='auto')
ax3.set_xticks(np.arange(len(set(df['player']))), labels=np.arange(1, 14))
ax3.set_yticks(np.arange(len(set(df['violin']))), labels=duration_sum.index)
ax3.set_xlabel('Player')
ax3.set_ylabel('Violin')

ax1.bar(np.arange(len(set(df['player']))), duration_sum.sum(), align='center')
ax1.set_ylabel('Rec. time (min)')
ax1.tick_params(bottom=False)

ax4.barh(np.arange(len(set(df['violin']))), duration_sum.sum(axis=1), align='center')
ax4.set_xlabel('Rec. time (min)')
ax4.tick_params(left=False)

ax2.axis('off')
w, h = fig.get_size_inches()
plt.subplots_adjust(wspace=.5/w, hspace=.5/h)
plt.savefig('reports/figures/class_weights_cnsm.svg', bbox_inches='tight', dpi=300)