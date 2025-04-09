import pandas as pd
import librosa
import tqdm
import numpy as np

from ..features import *

def get_dataset(config, df):
    config['hop_size'] = config['frame_size'] // config['hop_ratio']
    data = []
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        try:
            offset      = row['start']
            duration    = row['end'] - offset or None
        except KeyError:
            offset = None
            duration = None
        y, _        = librosa.load(str(row['file']), sr=config['sr'], offset=offset, duration=duration)

        # for audio in np.lib.stride_tricks.sliding_window_view(y, window_shape=config['sample_duration']*config['sr'])[::config['sample_duration']*config['sr']]:
        for audio in np.array_split(y, np.arange(config['sr'] * config['sample_duration'], len(y), config['sr'] * config['sample_duration'])):
        # for audio in np.array_split(y, config['sr'] * config['sample_duration']):
        # for audio in [y]:
            if len(audio) < config['sr'] * 4 :
                continue
            features = audio
            for step in pipes[config['feature']]:
                features = step(features, **config)

            dic = row.to_dict()
            dic.update(
                features=features,
                # audio=audio
            )
            data.append(dic)

    features_df = pd.DataFrame(data)
    return features_df