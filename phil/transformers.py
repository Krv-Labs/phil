"""
Scikit-learn compatible transformers for Phil.
"""

from typing import Any, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from phil.imputation import ImputationConfig
from phil.phil import Phil


class PhilTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        samples: int = 30,
        param_grid: Union[str, ImputationConfig] = "default",
        magic: str = "ECT",
        config: Optional[dict] = None,
        random_state: Optional[int] = None,
        max_iter: int = 5,
    ):
        self.samples = samples
        self.param_grid = param_grid
        self.magic = magic
        self.config = config
        self.random_state = random_state
        self.max_iter = max_iter
        self.phil = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PhilTransformer":
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        self.phil = Phil(
            samples=self.samples,
            param_grid=self.param_grid,
            magic=self.magic,
            config=self.config,
            random_state=self.random_state,
        )
        self.phil.fit(X, max_iter=self.max_iter)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.phil is None:
            raise RuntimeError(
                "This PhilTransformer instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
        return self.phil.transform(X, max_iter=self.max_iter)
