import os
import glob
import pandas as pd
import numpy as np
import datetime

# files = sorted(glob.glob(f'data/raw/Session 2/01-Couple 4006-*_*.txt'))
# files = sorted(glob.glob(f'data/raw/240910_SESSION1/AUDIO FILES/01-Couple 4006-*_*.txt'))

extracts = [
    {
        'name': 'Paul',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240910_1203.txt'
    },
    {
        'name': 'Renato',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240910_1326.txt'
    },
    {
        'name': 'Areski',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240910_1508.txt'
    },
    {
        'name': 'Félix',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240910_1643.txt'
    },
    {
        'name': 'Céleste',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240910_1808.txt'
    },
    {
        'name': 'SMD',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_0915.txt'
    },
    {
        'name': 'SMD',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1017.txt'
    },
    {
        'name': 'SMD',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1017.txt'
    },
    {
        'name': 'Hélène',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1054.txt'
    },
    {
        'name': 'Clara',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1208.txt'
    },
    {
        'name': 'Norimi',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1330.txt'
    },
    {
        'name': 'Fanton',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1454.txt'
    },
    {
        'name': 'Lucie',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1638.txt'
    },
    {
        'name': 'Eugénie',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/240910_SESSION1/AUDIO FILES/01-Couple 4006-240911_1737.txt'
    },
    {
        'name': 'Paul',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240918_1230.txt'
    },
    {
        'name': 'SMD',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240918_1413.txt'
    },
    {
        'name': 'SMD',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240918_1515.txt'
    },
    {
        'name': 'Félix',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240918_1537.txt'
    },
    {
        'name': 'Clara',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240918_1705.txt'
    },
    {
        'name': 'Areski',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240918_1802.txt'
    },
    {
        'name': 'Eugénie',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240919_0923.txt'
    },
    {
        'name': 'Norimi',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240919_1031.txt'
    },
    {
        'name': 'Fanton',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240919_1302.txt'
    },
    {
        'name': 'Renato',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240919_1352-01.txt'
    },
    {
        'name': 'Antonin',
        'session': 1,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240919_1528.txt'
    },
    {
        'name': 'Hélène',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240925_1705.txt'
    },
    {
        'name': 'Lucie',
        'session': 2,
        'path': '/home/hugo/Thèse/Data/Session 2/01-Couple 4006-240925_1813.txt'
    },
]

dfs = []
for extract in extracts:
    # Read each label file
    df = pd.read_csv(extract['path'], sep='\t', header=None, names=['start', 'end', 'annotation'])

    df[['violin', 'condition', 'extract']] = df['annotation'].str.split(';', n=3, expand=True)

    # Replace extract=None with free
    df.extract = df.extract.replace({None: 'free'})
    
    df = df.replace({'undefined': '?'})
    # df = df.replace({'?': 'free'})

    # Set experiment condition
    df.condition = np.where(df.violin.isin(['A', 'B', 'C']), 'aveugle', 'non-aveugle')

    # Rename violin D and E
    df.violin = df.violin.replace({'D': 'A', 'E': 'C'})

    # Add the path of the wav file
    df['file'] = extract['path'][:-4] + '.wav'

    # Add the player and the session
    df['player'] = extract['name']
    df['session'] = extract['session']

    df = df.drop('annotation', axis=1)

    dfs.append(df)

df = pd.concat(dfs, axis=0)

# Save dataset
df.to_pickle('data/processed/dataset_cnsm.pkl')
df.to_csv('data/processed/dataset_cnsm.csv')


## Rires
import librosa, scipy.io
y = []
for index, row in df[df.violin == 'rire'].iterrows():
    offset = row['start']
    duration = row['end'] - offset
    audio, _ = librosa.load(str(row['file']), offset=offset, duration=duration)
    y.append(audio)

y = np.hstack(y)
scipy.io.wavfile.write('data/processed/rires.wav', 22050, y)