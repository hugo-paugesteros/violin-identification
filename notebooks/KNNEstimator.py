import sklearn, sklearn.neighbors
import numpy as np
import librosa

class KNNEstimator(sklearn.base.BaseEstimator):

    def __init__(self, frame_size=2048, hop_ratio=2, n_coeff=18):
        self.frame_size = frame_size
        self.hop_ratio = hop_ratio
        self.n_coeff = n_coeff
        self.knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 13)

    def LTCC(self, y):
        Y = librosa.stft(y=y, n_fft=self.frame_size, hop_length=self.frame_size // self.hop_ratio).T
        S = 20 * np.log10(np.abs(Y))
        ltcc = np.fft.irfft(S)[:,:self.n_coeff]
        return ltcc
        
    def fit(self, recordings):
        X = []
        y = []
        for j in range(13):
            features = self.LTCC(recordings[j])
            X.append(features)
            y.append(np.ones(features.shape[0])*j)
        self.X = np.concatenate(X)
        self.y = np.concatenate(y)
        print(self.X.shape)
        return self.knn.fit(self.X, self.y)

    def predict(self, X):
        return self.knn.predict(self.X)

    def score(self, X, y):
        return self.knn.score(self.X, self.y)