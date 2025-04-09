import numpy as np
import sklearn.model_selection, sklearn.neighbors
import librosa
from KNNEstimator import KNNEstimator

def LTCC(y, frame_size, hop_ratio, n_coeff):
    Y = librosa.stft(y=y, n_fft=frame_size, hop_length=frame_size // hop_ratio).T
    S = 20 * np.log10(np.abs(Y))
    ltcc = np.fft.irfft(S)[:,:n_coeff]
    return ltcc

# recordings
recordings = [np.load(f'violin{i}.npy') for i in range(1,14)]

# Parameter Grid
param_grid = {
    'frame_size': [2**n for n in range(11, 16)],
    'hop_ratio': [2,4,8],
    'n_coeff': [10*i for i in range(1,5)]
}
grid = sklearn.model_selection.ParameterGrid(param_grid)

# Grid Search
for params in grid:
    print(params)
    x = []
    y = []
    for i in range(13):
        features = LTCC(recordings[i], **params)
        x.append(features)
        y.append(np.ones(features.shape[0])*i)
    x = np.concatenate(x)
    y = np.concatenate(y)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 13)
    knn.fit(x_train, y_train)
    print(f'\t Train : \t{round(knn.score(x_train,y_train), 2)} \n\t Test : \t{round(knn.score(x_test,y_test),2)}')