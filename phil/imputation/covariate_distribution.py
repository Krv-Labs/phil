"""
Covariate-conditional distribution imputation strategy.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist


class CovariateDistributionImputer(BaseEstimator):
    """
    Imputer that samples from the conditional distribution P(x_j | x_{-j})
    approximated via k-nearest neighbors in the observed covariate space.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        missing_values=np.nan,
        random_state=None,
        threshold: float = 1.0,
        covariance_matrix=None,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        self.missing_values = missing_values
        self.random_state = random_state
        self.threshold = threshold
        self.covariance_matrix = covariance_matrix

    def fit(self, X, y) -> "CovariateDistributionImputer":
        X = np.asarray(X)
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("CovariateDistributionImputer only supports 1D y.")

        self.dtype_ = y.dtype
        self.is_categorical_ = y.dtype.kind in "OSU"

        if not self.is_categorical_:
            y = y.astype(float, copy=True)
        else:
            y = y.astype(object, copy=True)

        if self.covariance_matrix is not None:
            cov = np.asarray(self.covariance_matrix)
            if cov.shape[0] != X.shape[1] or cov.shape[1] != X.shape[1]:
                raise ValueError("covariance_matrix must be square with dimensions matching X.")
            self.VI_ = np.linalg.inv(cov)
        else:
            self.VI_ = None

        missing_mask = (y == self.missing_values) | pd.isnull(y)
        fraction_missing = missing_mask.sum() / y.size

        n_features = X.shape[1] if X.ndim > 1 else 1
        if fraction_missing == 1.0:
            self.skip_imputation_ = True
            self.X_obs_ = np.empty((0, n_features))
            self.y_obs_ = np.array([], dtype=self.dtype_)
        else:
            self.skip_imputation_ = fraction_missing > self.threshold
            if not self.skip_imputation_:
                self.X_obs_ = X[~missing_mask]
                self.y_obs_ = y[~missing_mask]
            else:
                self.X_obs_ = np.empty((0, n_features))
                self.y_obs_ = np.array([], dtype=self.dtype_)

        if isinstance(self.random_state, np.random.RandomState):
            self.rng_ = self.random_state
        else:
            self.rng_ = np.random.RandomState(self.random_state)

        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "y_obs_"):
            raise RuntimeError("Call fit before predict")

        X = np.asarray(X)
        n_samples = X.shape[0]

        if self.skip_imputation_ or self.y_obs_.size == 0:
            if self.is_categorical_:
                return np.full(n_samples, None, dtype=object)
            return np.full(n_samples, np.nan, dtype=float)

        k = min(self.n_neighbors, len(self.y_obs_))
        
        if self.VI_ is not None:
            dists = cdist(X, self.X_obs_, metric='mahalanobis', VI=self.VI_)
        else:
            dists = euclidean_distances(X, self.X_obs_)
            
        neighbor_idxs = np.argpartition(dists, k - 1, axis=1)[:, :k]

        predictions = np.empty(n_samples, dtype=object if self.is_categorical_ else float)
        for i, neighbors in enumerate(neighbor_idxs):
            pool = self.y_obs_[neighbors]
            predictions[i] = self.rng_.choice(pool)

        return predictions.astype(self.dtype_)
