import warnings

import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.utils.validation import check_is_fitted

class MaskedIterativeImputer(IterativeImputer):
    """
    Extension of IterativeImputer that respects a covariate_subsets mapping.
    
    This class allows users to restrict which predictors are used to impute
    specific target variables based on domain knowledge.
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

    def fit(self, X, y=None):
        """Fit the imputer on X and return self."""
        if self.covariate_subsets and not self.feature_names:
            warnings.warn(
                "covariate_subsets provided but feature_names is missing. "
                "Masking will not be applied.",
                UserWarning
            )
            
        if self.covariate_subsets and self.feature_names:
            self._subset_indices = {}
            for target_name, config in self.covariate_subsets.items():
                if target_name in self.feature_names:
                    target_idx = self.feature_names.index(target_name)
                    predictor_names = config.get("predictors", [])
                    predictor_indices = [
                        self.feature_names.index(p)
                        for p in predictor_names
                        if p in self.feature_names
                    ]
                    self._subset_indices[target_idx] = set(predictor_indices)
        else:
            self._subset_indices = None
            
        return super().fit(X, y)

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True,
    ):
        if self._subset_indices and feat_idx in self._subset_indices:
            allowed_neighbors = self._subset_indices[feat_idx]
            original_neighbors = set(neighbor_feat_idx)
            restricted_neighbors = list(original_neighbors.intersection(allowed_neighbors))
            
            if not restricted_neighbors:
                restricted_neighbors = neighbor_feat_idx 
            else:
                restricted_neighbors = np.array(sorted(restricted_neighbors), dtype=int)
            
            return super()._impute_one_feature(
                X_filled,
                mask_missing_values,
                feat_idx,
                restricted_neighbors,
                estimator=estimator,
                fit_mode=fit_mode,
            )
            
        return super()._impute_one_feature(
            X_filled,
            mask_missing_values,
            feat_idx,
            neighbor_feat_idx,
            estimator=estimator,
            fit_mode=fit_mode,
        )
