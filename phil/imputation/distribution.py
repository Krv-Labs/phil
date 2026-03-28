"""
Distribution-preserving imputation strategies.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DistributionImputer(BaseEstimator):
    """Imputer that samples from empirical observed values."""

    def __init__(self, missing_values=np.nan, random_state=None, threshold=1.0):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.missing_values = missing_values
        self.random_state = random_state
        self.threshold = threshold

    def fit(self, X, y):
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("DistributionImputer only supports 1D y.")

        self.dtype_ = y.dtype
        self.is_categorical_ = y.dtype.kind in "OSU"

        if not self.is_categorical_:
            y = y.astype(float, copy=True)
        else:
            y = y.astype(object, copy=True)

        missing_mask = (y == self.missing_values) | pd.isnull(y)
        fraction_missing = missing_mask.sum() / y.size

        if fraction_missing == 1.0:
            self.skip_imputation_ = True
            self.distribution_ = np.array([], dtype=self.dtype_)
        else:
            self.skip_imputation_ = fraction_missing > self.threshold
            if not self.skip_imputation_:
                self.distribution_ = y[~missing_mask]
            else:
                self.distribution_ = np.array([], dtype=self.dtype_)

        if isinstance(self.random_state, np.random.RandomState):
            self.rng_ = self.random_state
        else:
            self.rng_ = np.random.RandomState(self.random_state)

        return self

    def predict(self, X):
        if not hasattr(self, "distribution_"):
            raise RuntimeError("Call fit before predict")

        n_samples = X.shape[0]

        if self.skip_imputation_ or self.distribution_.size == 0:
            if self.is_categorical_:
                return np.full(n_samples, None, dtype=object)
            return np.full(n_samples, np.nan, dtype=float)

        predictions = self.rng_.choice(self.distribution_, size=n_samples, replace=True)
        return predictions.astype(self.dtype_)
