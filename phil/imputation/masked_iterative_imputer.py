import importlib
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.impute import IterativeImputer

from phil.imputation.config import ESTIMATOR_MAP
from phil.imputation.covariate_distribution import CovariateDistributionImputer


class MaskedIterativeImputer(IterativeImputer):
    """
    Extension of IterativeImputer that respects a covariate_subsets mapping,
    per-feature estimator overrides, and per-feature bounds.

    This class allows users to restrict which predictors and estimators are
    used to impute specific target variables based on domain knowledge.
    """

    def __init__(
        self,
        estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        initial_strategy="mean",
        fill_value=None,
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
        keep_empty_features=False,
        covariate_subsets=None,
        feature_names=None,
    ):
        super().__init__(
            estimator=estimator,
            missing_values=missing_values,
            sample_posterior=sample_posterior,
            max_iter=max_iter,
            tol=tol,
            n_nearest_features=n_nearest_features,
            initial_strategy=initial_strategy,
            fill_value=fill_value,
            imputation_order=imputation_order,
            skip_complete=skip_complete,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose,
            random_state=random_state,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        self.covariate_subsets = covariate_subsets
        self.feature_names = feature_names

    @staticmethod
    def _instantiate_named_estimator(est_name: str, random_state=None):
        if est_name not in ESTIMATOR_MAP:
            return None
        mod_name, cls_name = ESTIMATOR_MAP[est_name]
        cls_obj = getattr(importlib.import_module(mod_name), cls_name)
        kwargs = {}
        if (
            random_state is not None
            and "random_state" in cls_obj.__init__.__code__.co_varnames
        ):
            kwargs["random_state"] = random_state
        return cls_obj(**kwargs)

    def fit_transform(self, X, y=None):
        """Fit the imputer on X and return the transformed array."""
        if self.covariate_subsets and not self.feature_names:
            warnings.warn(
                "covariate_subsets provided but feature_names is missing. "
                "Masking will not be applied.",
                UserWarning,
            )

        self._subset_indices = None
        self._subset_estimators = {}
        if self.covariate_subsets and self.feature_names:
            self._subset_indices = {}
            for target_name, config in self.covariate_subsets.items():
                if target_name in self.feature_names:
                    target_idx = self.feature_names.index(target_name)
                    if isinstance(config, dict):
                        predictor_names = config.get("predictors", [])
                        estimators_list = config.get("estimators") or []
                    else:
                        predictor_names = getattr(config, "predictors", [])
                        estimators_list = getattr(config, "estimators", None) or []
                    predictor_indices = [
                        self.feature_names.index(p)
                        for p in predictor_names
                        if p in self.feature_names
                    ]
                    self._subset_indices[target_idx] = set(predictor_indices)
                    if estimators_list and self.estimator is None:
                        est_obj = self._instantiate_named_estimator(
                            str(estimators_list[0]), random_state=self.random_state
                        )
                        if est_obj is not None:
                            self._subset_estimators[target_idx] = est_obj

        return super().fit_transform(X, y)

    def fit(self, X, y=None):
        """Fit the imputer on X and return self."""
        self.fit_transform(X, y)
        return self

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True,
        params=None,
    ):
        if (
            estimator is None
            and hasattr(self, "_subset_estimators")
            and feat_idx in self._subset_estimators
            and not self.sample_posterior
        ):
            estimator = clone(self._subset_estimators[feat_idx])

        extra_kwargs = {"params": params} if params is not None else {}

        if getattr(self, "_subset_indices", None) and feat_idx in self._subset_indices:
            allowed_neighbors = self._subset_indices[feat_idx]
            original_neighbors = set(neighbor_feat_idx)
            restricted_neighbors = list(
                original_neighbors.intersection(allowed_neighbors)
            )

            if restricted_neighbors:
                neighbor_feat_idx = np.array(sorted(restricted_neighbors), dtype=int)

        # Each regression sees a different ordered subset of predictors. Slice the
        # full covariance before fitting and retain that estimator for transform.
        if estimator is None and isinstance(
            self._estimator, CovariateDistributionImputer
        ):
            covariance = self._estimator.covariance_matrix
            if covariance is not None:
                covariance = np.asarray(covariance, dtype=float)
                if covariance.shape != (X_filled.shape[1], X_filled.shape[1]):
                    raise ValueError(
                        "Covariance matrix must match all imputer features."
                    )
                estimator = clone(self._estimator)
                estimator.set_params(
                    covariance_matrix=covariance[
                        np.ix_(neighbor_feat_idx, neighbor_feat_idx)
                    ]
                )

        try:
            return super()._impute_one_feature(
                X_filled,
                mask_missing_values,
                feat_idx,
                neighbor_feat_idx,
                estimator=estimator,
                fit_mode=fit_mode,
                **extra_kwargs,
            )
        except TypeError:
            return super()._impute_one_feature(
                X_filled,
                mask_missing_values,
                feat_idx,
                neighbor_feat_idx,
                estimator=estimator,
                fit_mode=fit_mode,
            )
