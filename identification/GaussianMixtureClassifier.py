import sklearn.base, sklearn.mixture
import numpy as np

class GaussianMixtureClassifier(sklearn.base.BaseEstimator):

    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X, y):
        means_init = np.array(
            [X[y == i].mean(axis=0) for i in range(self.n_components)]
        )
        self.gmm = sklearn.mixture.GaussianMixture(n_components=self.n_components, means_init=means_init, verbose=5, tol=1e-200, max_iter=100)
        self.gmm.fit(X)
        return self

    def predict(self, x):
        return self.gmm.predict(x)

    def score(self, X, y):
        y_pred = self.gmm.predict(X)
        score = np.mean(y.ravel() == y_pred.ravel())
        return score
