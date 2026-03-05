import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import welch

class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        n_trials, n_channels, n_times = X.shape
        features = np.zeros((n_trials, n_channels))
        for i in range(n_trials):
            for j in range(n_channels):
                coeffs = pywt.wavedec(X[i, j, :], self.wavelet, level=self.level)
                features[i, j] = np.sum(np.square(coeffs[-1]))
        return np.log1p(features)
