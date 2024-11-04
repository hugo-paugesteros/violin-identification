import os
import glob
import pandas as pd
import numpy as np

files = sorted(glob.glob(f'data/raw/240910_SESSION1/AUDIO FILES/01-Couple 4006-*_*.txt'))[:4]

for file in files:
    df = pd.read_csv(file, sep='\t', header=None, names=['start', 'end', 'annotation'])
    df[['violin', 'extract', 'condition']] = df['annotation'].str.split(';', n=3, expand=True)
    df = df.replace({None: '?'})
    df['annotation'] = df[['violin', 'condition', 'extract']].apply(';'.join, axis=1)
    df = df.drop(['violin', 'condition', 'extract'], axis=1)
    df.to_csv(file, header=None, index=False, sep='\t')

