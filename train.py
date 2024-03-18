import pandas as pd
import librosa
import numpy as np
import sklearn.neighbors, sklearn.preprocessing
from features import LTCC

SR          = 10000
FRAME_SIZE  = 2048
HOP_SIZE    = FRAME_SIZE
N_COEFF     = 45

df = pd.read_pickle('recordings.pkl')

# df = df[(df['type'] == 'scale') & (df['player'] != 3)]

data = []
for index, row in df.iterrows():
    audio, _ = librosa.load(str(row['file']), sr=SR)
    features = LTCC(audio, FRAME_SIZE, HOP_SIZE, N_COEFF)
    for i in range(features.shape[0]):
        dic = dict(zip(list(range(features.shape[0])), features[i].T))
        dic.update(player=row['player'], violin=row['violin'], type=row['type'])
        data.append(dic)

features_df = pd.DataFrame(data)

# Standardization
scaler = sklearn.preprocessing.StandardScaler()
# scaler = sklearn.preprocessing.MinMaxScaler()
features_df.iloc[:, :N_COEFF] = scaler.fit_transform(features_df.iloc[:, :N_COEFF])


train_cdt = (features_df['type'] == 'scale') & (features_df['player'].any() not in [3, 4])
x_train = features_df[train_cdt].iloc[:, :N_COEFF]
y_train = features_df[train_cdt]['violin']
x_test = features_df[~train_cdt].iloc[:, :N_COEFF]
y_test = features_df[~train_cdt]['violin']

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 13, weights='uniform', p=1)
knn.fit(x_train, y_train)
print(f'Train score : {knn.score(x_train, y_train)}')
print(f'Test score : {knn.score(x_test, y_test)}')

