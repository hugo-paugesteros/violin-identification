import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import librosa
import numpy as np

plt.style.use('styles.mplstyle')
plt.rcParams['image.composite_image'] = False

# Data
df = pd.read_pickle('recordings.pkl')
df = df[df['player'].isin([30,31,32,99])]

# Compute duration
def get_row_duration(row):
    row['duration'] = librosa.get_duration(path=row['file'])
    return row
df = df.apply(get_row_duration, axis=1)

duration_sum = df.groupby(['violin', 'player']).sum().unstack('player', fill_value=0)['duration']

# Plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9), sharey='row', sharex='col', gridspec_kw=dict(height_ratios=[1, 3],width_ratios=[3, 1]))

ax3.imshow(duration_sum.to_numpy(), aspect='auto')
ax3.set_xticks(np.arange(len(set(df['player']))), labels=duration_sum.columns)
ax3.set_yticks(np.arange(len(set(df['violin']))), labels=duration_sum.index)
ax3.set_xlabel('Player')
ax3.set_ylabel('Violin')

ax1.bar(np.arange(len(set(df['player']))), duration_sum.sum()/60, align='center')
ax1.set_ylabel('Recording time (min)')
ax1.tick_params(bottom=False)

ax4.barh(np.arange(len(set(df['violin']))), duration_sum.sum(axis=1)/60, align='center')
ax4.set_xlabel('Recording time (min)')
ax4.tick_params(left=False)

ax2.axis('off')

plt.subplots_adjust(wspace=1/16, hspace=1/9)

plt.savefig('figures/class_weights_2024.png', bbox_inches='tight', dpi=300)
plt.show()