import numpy as np
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FactorAnalysis


class FactorAnalysisTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        tol=0.01,
        copy=True,
        max_iter=1000,
        noise_variance_init=None,
        svd_method: Literal['lapack', 'randomized'] = "randomized",
        iterated_power=3,
        rotation=None,
        random_state=None,
        append_latents=False,
        n_energy_details=-1,
    ):
        self.n_components = n_components
        self.tol = tol
        self.copy = copy
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.svd_method: Literal['lapack', 'randomized'] = svd_method
        self.iterated_power = iterated_power
        self.rotation = rotation
        self.random_state = random_state
        self.append_latents = append_latents
        self.n_energy_details = n_energy_details

    def fit(self, X, y=None):
        self.model_ = FactorAnalysis(
            n_components=self.n_components,
            tol=self.tol,
            copy=self.copy,
            max_iter=self.max_iter,
            noise_variance_init=self.noise_variance_init,
            svd_method=self.svd_method,
            iterated_power=self.iterated_power,
            rotation=self.rotation,
            random_state=self.random_state,
        )
        target, _ = self._split_features(X)
        self.model_.fit(target, y)

        self.components_ = self.model_.components_
        self.mean_ = self.model_.mean_
        self.noise_variance_ = self.model_.noise_variance_
        self.loglike_ = self.model_.loglike_
        self.n_iter_ = self.model_.n_iter_
        return self

    def transform(self, X):
        target, energy = self._split_features(X)
            
        z = self.model_.transform(target)
        
        if self.append_latents:
            return np.hstack([energy, z])

        return energy

    def transform_latent(self, X):
        _, target_features = self._split_features(X)
            
        return self.model_.transform(target_features)
    
    def _split_features(self, X):
        if self.n_energy_details <= 0:
            return X, X
        energy_features = X[:,:self.n_energy_details]
        target_features = X[:,self.n_energy_details:]
        
        if X.shape[1] <= self.n_energy_details + 10:
            target_features = energy_features
        
        return energy_features, target_features
    
    
class SensorTiedFactorAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    Factor Analysis with diagonal noise tied by sensor group.

    Feature ordering assumption:
        Features are ordered by sensor, then wavelet level, matching the current
        wavelet naming pattern:
            sensor 0: cDlevel, ..., cD1
            sensor 1: cDlevel, ..., cD1
            ...

    Available attributes:
        components_: shape (n_components, n_features), sklearn-style
        mean_: shape (n_features,)
        noise_variance_: expanded per-feature noise, shape (n_features,)
        sensor_noise_variance_: one tied value per sensor group
        loglike_: list of EM log-likelihoods
        n_iter_: number of EM iterations
    """

    def __init__(
        self,
        n_components=2,
        tol=0.01,
        copy=True,
        max_iter=1000,
        noise_variance_init=None,
        svd_method: Literal["lapack", "randomized"] = "randomized",
        iterated_power=3,
        rotation=None,
        random_state=None,
        append_latents=False,
        sensor_groups=None,
        n_sensors=None,
        n_features_per_sensor=None,
        min_noise_variance=1e-6,
        init_with_sklearn_fa=True,
    ):
        self.n_components = n_components
        self.tol = tol
        self.copy = copy
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.svd_method: Literal["lapack", "randomized"] = svd_method
        self.iterated_power = iterated_power
        self.rotation = rotation
        self.random_state = random_state
        self.append_latents = append_latents
        self.sensor_groups = sensor_groups
        self.n_sensors = n_sensors
        self.n_features_per_sensor = n_features_per_sensor
        self.min_noise_variance = min_noise_variance
        self.init_with_sklearn_fa = init_with_sklearn_fa

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if self.copy:
            X = X.copy()

        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D, got shape {X.shape}")

        n_samples, n_features = X.shape
        if self.n_components >= n_features:
            raise ValueError(
                f"n_components must be smaller than n_features. "
                f"Got n_components={self.n_components}, n_features={n_features}."
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        self.sensor_groups_ = self._build_sensor_groups(n_features)
        self.unique_sensor_groups_ = np.unique(self.sensor_groups_)
        self.n_sensor_groups_ = len(self.unique_sensor_groups_)

        W, sensor_noise = self._initialize_parameters(X_centered)

        self.loglike_ = []
        previous_loglike = -np.inf

        for iteration in range(self.max_iter):
            expected_z, expected_zz_mean = self._e_step(X_centered, W, sensor_noise)

            W = self._m_step_loadings(X_centered, expected_z, expected_zz_mean)
            residual_variance = self._expected_residual_variance(
                X_centered,
                W,
                expected_z,
                expected_zz_mean,
            )
            sensor_noise = self._tie_noise_by_sensor(residual_variance)

            loglike = self._log_likelihood(X_centered, W, sensor_noise)
            self.loglike_.append(loglike)

            if iteration > 0 and abs(loglike - previous_loglike) < self.tol:
                break

            previous_loglike = loglike

        self.n_iter_ = len(self.loglike_)

        self.components_ = W.T
        self.sensor_noise_variance_ = sensor_noise
        self.noise_variance_ = self._expand_sensor_noise(sensor_noise)

        return self

    def transform(self, X):
        z = self.transform_latent(X)

        if self.append_latents:
            return np.hstack([X, z])

        return z

    def transform_latent(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_

        W = self.components_.T
        inv_noise = 1.0 / self.noise_variance_

        posterior_cov = np.linalg.inv(
            np.eye(self.n_components) + W.T @ (inv_noise[:, None] * W)
        )

        return (X_centered * inv_noise) @ W @ posterior_cov

    def _build_sensor_groups(self, n_features):
        if self.sensor_groups is not None:
            sensor_groups = np.asarray(self.sensor_groups)
            if len(sensor_groups) != n_features:
                raise ValueError(
                    f"sensor_groups must have length n_features={n_features}, "
                    f"got length {len(sensor_groups)}."
                )
            return sensor_groups

        if self.n_features_per_sensor is not None:
            if n_features % self.n_features_per_sensor != 0:
                raise ValueError(
                    f"n_features={n_features} is not divisible by "
                    f"n_features_per_sensor={self.n_features_per_sensor}."
                )

            n_sensors = n_features // self.n_features_per_sensor
            return np.repeat(np.arange(n_sensors), self.n_features_per_sensor)

        if self.n_sensors is not None:
            if n_features % self.n_sensors != 0:
                raise ValueError(
                    f"n_features={n_features} is not divisible by "
                    f"n_sensors={self.n_sensors}."
                )

            n_features_per_sensor = n_features // self.n_sensors
            return np.repeat(np.arange(self.n_sensors), n_features_per_sensor)

        raise ValueError(
            "SensorTiedFactorAnalysisTransformer needs one of: "
            "sensor_groups, n_sensors, or n_features_per_sensor."
        )

    def _initialize_parameters(self, X_centered):
        n_samples, n_features = X_centered.shape

        if self.init_with_sklearn_fa:
            init_model = FactorAnalysis(
                n_components=self.n_components,
                tol=self.tol,
                copy=True,
                max_iter=min(self.max_iter, 100),
                noise_variance_init=self.noise_variance_init,
                svd_method=self.svd_method,
                iterated_power=self.iterated_power,
                rotation=self.rotation,
                random_state=self.random_state,
            )
            init_model.fit(X_centered)

            W = init_model.components_.T
            feature_noise = np.asarray(init_model.noise_variance_, dtype=np.float64)
            sensor_noise = self._tie_noise_by_sensor(feature_noise)
            return W, sensor_noise

        rng = np.random.default_rng(self.random_state)
        feature_variance = np.var(X_centered, axis=0, ddof=1)

        W = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(n_features, self.n_components),
        )

        if self.noise_variance_init is None:
            feature_noise = np.maximum(feature_variance, self.min_noise_variance)
        else:
            feature_noise = self._coerce_noise_init(n_features)

        sensor_noise = self._tie_noise_by_sensor(feature_noise)
        return W, sensor_noise

    def _coerce_noise_init(self, n_features):
        init = np.asarray(self.noise_variance_init, dtype=np.float64)

        if init.ndim == 0:
            return np.full(n_features, float(init))

        if len(init) == n_features:
            return init

        if len(init) == self.n_sensor_groups_:
            return self._expand_sensor_noise(init)

        raise ValueError(
            "noise_variance_init must be scalar, feature-length, or sensor-group-length. "
            f"Got shape {init.shape}."
        )

    def _e_step(self, X_centered, W, sensor_noise):
        noise_variance = self._expand_sensor_noise(sensor_noise)
        inv_noise = 1.0 / noise_variance

        posterior_cov = np.linalg.inv(
            np.eye(self.n_components) + W.T @ (inv_noise[:, None] * W)
        )

        expected_z = (X_centered * inv_noise) @ W @ posterior_cov

        expected_zz_mean = posterior_cov + (expected_z.T @ expected_z) / X_centered.shape[0]

        return expected_z, expected_zz_mean

    def _m_step_loadings(self, X_centered, expected_z, expected_zz_mean):
        cross_cov = (X_centered.T @ expected_z) / X_centered.shape[0]
        return cross_cov @ np.linalg.inv(expected_zz_mean)

    def _expected_residual_variance(self, X_centered, W, expected_z, expected_zz_mean):
        data_variance = np.mean(X_centered * X_centered, axis=0)
        cross_cov = (X_centered.T @ expected_z) / X_centered.shape[0]

        # For each feature i:
        # E[(x_i - w_i z)^2]
        residual_variance = (
            data_variance
            - 2.0 * np.sum(W * cross_cov, axis=1)
            + np.einsum("ij,jk,ik->i", W, expected_zz_mean, W)
        )

        return np.maximum(residual_variance, self.min_noise_variance)

    def _tie_noise_by_sensor(self, feature_noise):
        feature_noise = np.asarray(feature_noise, dtype=np.float64)
        sensor_noise = np.empty(self.n_sensor_groups_, dtype=np.float64)

        for i, sensor_group in enumerate(self.unique_sensor_groups_):
            mask = self.sensor_groups_ == sensor_group
            sensor_noise[i] = np.mean(feature_noise[mask])

        return np.maximum(sensor_noise, self.min_noise_variance)

    def _expand_sensor_noise(self, sensor_noise):
        sensor_noise = np.asarray(sensor_noise, dtype=np.float64)
        expanded = np.empty(len(self.sensor_groups_), dtype=np.float64)

        for i, sensor_group in enumerate(self.unique_sensor_groups_):
            mask = self.sensor_groups_ == sensor_group
            expanded[mask] = sensor_noise[i]

        return np.maximum(expanded, self.min_noise_variance)

    def _log_likelihood(self, X_centered, W, sensor_noise):
        n_samples, n_features = X_centered.shape
        noise_variance = self._expand_sensor_noise(sensor_noise)

        covariance = W @ W.T + np.diag(noise_variance)
        sign, logdet = np.linalg.slogdet(covariance)

        if sign <= 0:
            return -np.inf

        solved = np.linalg.solve(covariance, X_centered.T).T
        quadratic = np.sum(X_centered * solved)

        return -0.5 * (
            n_samples * (n_features * np.log(2.0 * np.pi) + logdet)
            + quadratic
        )