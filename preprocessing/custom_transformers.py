import pywt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        wavelet='db4',
        level=4,
        mode="symmetric",
        drop_first_detail=False,
        include_coefficients=False,
        include_approximation=True,
    ):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.drop_first_detail = drop_first_detail
        self.include_coefficients = include_coefficients
        self.include_approximation = include_approximation

    def __setstate__(self, state):
        # Pipelines saved before coefficient output was introduced do not
        # contain these attributes.
        state.setdefault("include_coefficients", False)
        state.setdefault("include_approximation", True)
        self.__dict__.update(state)

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                "WaveletTransformer expects X with shape "
                "(n_trials, n_channels, n_samples)."
            )
        self.n_channels_in_ = X.shape[1]
        self.n_samples_in_ = X.shape[2]
        self.n_features_out_ = self._transform_trial(X[0]).size
        return self

    @property
    def n_details(self):
        return max(1, self.level - 1) if self.drop_first_detail else self.level
    
    def detail_levels(self):
        return range(self.level - self.n_details, self.level)

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                "WaveletTransformer expects X with shape "
                "(n_trials, n_channels, n_samples)."
            )

        features = np.empty(
            (X.shape[0], self._transform_trial(X[0]).size),
            dtype=float,
        )

        for trial_index in range(X.shape[0]):
            features[trial_index] = self._transform_trial(X[trial_index])
        return features

    def _transform_trial(self, trial):
        out = []
        for channel in trial:
            coeffs = pywt.wavedec(
                channel,
                self.wavelet,
                level=self.level,
                mode=self.mode,
            )
            detail_coeffs = coeffs[-self.n_details:]

            # Retain the compact, shift-tolerant energy summary first.
            out.extend(np.log1p([np.sum(coeff * coeff) for coeff in detail_coeffs]))

            if not self.include_coefficients:
                if self.include_approximation:
                    out.extend(coeffs[0])
                for detail_coeff in detail_coeffs:
                    out.extend(detail_coeff)

        return np.asarray(out, dtype=float)



class ChannelDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_channel: int = -1):
        self.drop_channel = drop_channel

    def fit(self, X, y=None):
        n_channels = X.shape[1]

        if self.drop_channel < 0:
            self.keep_channels_ = list(range(n_channels))
        else:
            self.keep_channels_ = [
                i for i in range(n_channels)
                if i != self.drop_channel
            ]

        return self

    def transform(self, X):
        return X[:, self.keep_channels_, :]
