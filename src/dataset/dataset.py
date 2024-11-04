from sklearn.base import BaseEstimator, TransformerMixin
import librosa
import pandas as pd
import numpy as np
from ..features import *
import tqdm
import json

class Dataset(BaseEstimator):
    
    def __init__(
        self, 
        frame_size = 2048, 
        hop_ratio = 1,
        sr = 22050, 
        n_coeff = 50, 
        features = 'MFCC_welch',
        sample_duration = 10
    ):
        super().__init__()
        self.frame_size= frame_size
        self.hop_ratio = hop_ratio
        self.hop_size = frame_size // hop_ratio
        self.sr = sr
        self.n_coeff = n_coeff
        self.features = features
        self.sample_duration = sample_duration

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.fit_transform(X, y)
    
    def fit_transform(self, X, y):

        data = []
        for index, row in tqdm.tqdm(X.iterrows(), total=X.shape[0]):
            offset = row['start'] or None
            duration = (row['end'] - offset) or None
            y, _    = librosa.load(str(row['file']), sr=self.sr, offset=offset, duration=duration)
            
            # for i, audio in  enumerate(np.lib.stride_tricks.sliding_window_view(y, window_shape=self.sample_duration * self.sr)[::self.sample_duration * self.sr]):
            # for audio in np.split(y, np.arange(self.sample_duration * self.sr, len(y), self.sample_duration * self.sr)):
            for audio in [y]:

                features = y
                for step in pipes[self.features]:
                    features = step(features, **{
                        'sr': self.sr, 
                        'frame_size': self.frame_size, 
                        'hop_size': self.hop_size, 
                        'n_coeff': self.n_coeff
                    })

                dic = row.to_dict()
                dic.update(
                    features=features,
                    audio=audio
                )
                data.append(dic)
        dataset = pd.DataFrame(data)
        return (np.vstack(dataset.features), dataset.violin.to_numpy())