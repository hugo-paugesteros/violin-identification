import os
import glob
import pandas as pd
import numpy as np

files = glob.glob(f'data/raw/240910_SESSION1/AUDIO FILES/01-Couple 4006-*_*_labels_corrected.txt')

dfs = []
for file in files:
    # Read each label file
    df = pd.read_csv(file, sep='\t', header=None, names=['start', 'end', 'annotation'])

    # Parse the annotation column violin;extract;condition
    df[['violin', 'extract', 'condition']] = df['annotation'].str.split(';', n=3, expand=True)

    # Replace extract=None with free
    df.extract = df.extract.replace({None: 'free'})

    # Set experiment condition
    df.condition = np.where(df.violin.isin(['A', 'B', 'C']), 'aveugle', 'non-aveugle')

    # Rename violin D and E
    df.violin = df.violin.replace({'D': 'A', 'E': 'C'})

    # Add the path of the wav file
    df['file'] = file.rsplit('_', 2)[0] + '.wav'

    dfs.append(df)

df = pd.concat(dfs, axis=0)

# Save dataset
df.to_pickle('data/processed/dataset_cnsm.pkl')
df.to_csv('data/processed/dataset_cnsm.csv')