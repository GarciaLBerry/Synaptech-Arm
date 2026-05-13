import pywt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=4, mode="symmetric"):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        n_trials, n_channels, _ = X.shape
        
        # pywt.wavedec returns [cA_L, cD_L, cD_{L-1}, ..., cD_1]
        features = np.empty((n_trials, n_channels * self.level), dtype=float)
        
        for i in range(n_trials):
            out = []
            for ch in range(n_channels):
                coeffs = pywt.wavedec(X[i, ch, :], self.wavelet, level=self.level, mode=self.mode)
                
                # energy per component
                out.extend([np.sum(c * c) for c in coeffs[1:]])

            features[i, :] = np.log1p(out)
        return features
