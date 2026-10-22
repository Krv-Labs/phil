import importlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline

import phil.magic as METHODS
from phil.gallery import GridGallery, MagicGallery, ProcessingGallery
from phil.imputation import (
    ESTIMATOR_MAP,
    CovariateDistributionImputer,
    DomainKnowledge,
    ImputationConfig,
    MaskedIterativeImputer,
)
from phil.magic import ECTCollapseResult, Magic, collapse_ect_ensemble

# Methods that are already imputers — do not wrap again in IterativeImputer.
_STANDALONE_IMPUTERS = {
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "MaskedIterativeImputer",
}

# String estimator names used in gallery ParameterGrids for IterativeImputer.
_NAMED_ESTIMATORS = ESTIMATOR_MAP


class Phil:
    def __init__(
        self,
        samples: int = 30,
        param_grid: str = "default",
        magic: str = "ECT",
        config=None,
        random_state=None,
        collapse_strategy: str = "euclidean_medoid",
        temperature: float | None = 1.0,
        trim_quantile: float = 0.2,
    ):
        self.config, self.magic = self._configure_magic_method(
            magic=magic, config=config
        )
        self.samples = samples
        self.param_grid = self._configure_param_grid(param_grid)
        self.random_state = random_state
        self.collapse_strategy = collapse_strategy
        self.temperature = 1.0 if temperature is None else float(temperature)
        self.trim_quantile = float(trim_quantile)
        self.representations = []
        self.magic_descriptors = []
        self.standardized_candidates_ = []
        self.imputed_candidates_ = []
        self.candidate_metadata_ = []
        self.ect_collapse_result_ = None
        self.collapse_result_ = None

    def impute(self, df: pd.DataFrame, max_iter: int = 10) -> list[np.ndarray]:
        if df.isnull().sum().sum() == 0:
            raise ValueError("No missing values found in the input DataFrame.")
        self._input_df_ = df.copy()
        categorical_columns, numerical_columns = self._identify_column_types(df)
        self._categorical_columns_ = categorical_columns
        self._numerical_columns_ = numerical_columns
        preprocessor = self._configure_preprocessor(
            "default", categorical_columns, numerical_columns
        )
        preprocessor.fit(df)
        f_names = preprocessor.get_feature_names_out()
        self.feature_names_out_ = (
            f_names.tolist() if hasattr(f_names, "tolist") else list(f_names)
        )

        imputers = self._create_imputers(preprocessor, max_iter)
        self.selected_imputers = self._select_imputations(imputers)
        applied = self._apply_imputations(df, self.selected_imputers)
        self.standardized_candidates_ = list(applied)
        self.candidate_metadata_ = self._extract_candidate_metadata(
            self.selected_imputers
        )
        self.imputed_candidates_ = [
            self._inverse_transform_candidate(
                arr, pipe, df, categorical_columns, numerical_columns
            )
            for arr, pipe in zip(applied, self.selected_imputers)
        ]
        return applied

    @staticmethod
    def _identify_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
        categorical_columns = df.select_dtypes(
            include=["object", "category", "str"]
        ).columns.tolist()
        numerical_columns = df.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        return categorical_columns, numerical_columns

    def _create_imputers(
        self, preprocessor: ColumnTransformer, max_iter: int
    ) -> list[Pipeline]:
        imputers = []
        domain_knowledge = getattr(self.param_grid, "domain_knowledge", None)

        for method, module, params in zip(
            self.param_grid.methods,
            self.param_grid.modules,
            self.param_grid.grids,
        ):
            model = self._import_model(module, method)
            for param_vals in params:
                compatible_params = {
                    k: v
                    for k, v in param_vals.items()
                    if k in model.__init__.__code__.co_varnames
                }
                compatible_params = self._resolve_estimator_params(compatible_params)

                if (
                    domain_knowledge
                    and domain_knowledge.covariance_matrix
                    and "covariance_matrix" in model.__init__.__code__.co_varnames
                ):
                    compatible_params["covariance_matrix"] = (
                        self._align_covariance_for_cdi(
                            np.asarray(domain_knowledge.covariance_matrix.matrix),
                            domain_knowledge.covariance_matrix.variables,
                            preprocessor,
                        )
                    )

                estimator = model(**compatible_params)
                imputers.append(
                    self._build_pipeline(
                        preprocessor,
                        estimator,
                        max_iter,
                        domain_knowledge,
                        method=method,
                    )
                )

        # If domain_knowledge provides a covariance_matrix and CovariateDistributionImputer
        # was not already in the grid, inject CovariateDistributionImputer candidates
        if (
            domain_knowledge
            and domain_knowledge.covariance_matrix
            and "CovariateDistributionImputer" not in self.param_grid.methods
        ):
            cov_vars = domain_knowledge.covariance_matrix.variables or []
            cov_mat = np.asarray(domain_knowledge.covariance_matrix.matrix, dtype=float)
            for k_neighbors in (3, 5):
                sub_cov = self._align_covariance_for_cdi(
                    cov_mat, cov_vars, preprocessor
                )
                cdi = CovariateDistributionImputer(
                    n_neighbors=k_neighbors,
                    random_state=self.random_state,
                    covariance_matrix=sub_cov,
                )
                imputers.append(
                    self._build_pipeline(
                        preprocessor,
                        cdi,
                        max_iter,
                        domain_knowledge,
                        method="CovariateDistributionImputer",
                    )
                )

        return imputers

    def _align_covariance_for_cdi(
        self, cov_mat: np.ndarray, cov_vars: list[str], preprocessor: ColumnTransformer
    ) -> np.ndarray:
        """Align an original-unit numeric covariance to the scaled feature order."""
        features = self.feature_names_out_
        if any(not name.startswith("num__") for name in features):
            raise ValueError("Covariance priors require numeric features only.")
        columns = [name.removeprefix("num__") for name in features]
        variables = cov_vars or list(self._input_df_.columns)
        if (
            cov_mat.shape != (len(variables), len(variables))
            or len(set(variables)) != len(variables)
            or not set(columns).issubset(variables)
        ):
            raise ValueError("Covariance matrix must cover all numeric feature names.")
        indices = [variables.index(col) for col in columns]
        aligned = np.asarray(cov_mat, dtype=float)[np.ix_(indices, indices)]
        scale = preprocessor.named_transformers_["num"].scale_
        return aligned / np.outer(scale, scale)

    @classmethod
    def _resolve_estimator_params(cls, params: dict) -> dict:
        """Materialize string estimator names used by gallery grids."""
        resolved = dict(params)
        est = resolved.get("estimator")
        if isinstance(est, str):
            if est not in _NAMED_ESTIMATORS:
                raise ValueError(f"Unknown estimator name in grid: {est!r}")
            module_name, class_name = _NAMED_ESTIMATORS[est]
            resolved["estimator"] = cls._import_model(module_name, class_name)()
        return resolved

    @staticmethod
    def _import_model(module: str, method: str):
        imported_module = importlib.import_module(module)
        return getattr(imported_module, method)

    def _map_covariate_subsets(self, domain_knowledge) -> dict:
        mapped_subsets = {}
        for target, subset_config in domain_knowledge.covariate_subsets.items():
            target_mapped = (
                f"num__{target}"
                if f"num__{target}" in self.feature_names_out_
                else (
                    f"cat__{target}"
                    if f"cat__{target}" in self.feature_names_out_
                    else target
                )
            )
            predictors = (
                subset_config.get("predictors", [])
                if isinstance(subset_config, dict)
                else subset_config.predictors
            )
            estimators = (
                subset_config.get("estimators")
                if isinstance(subset_config, dict)
                else getattr(subset_config, "estimators", None)
            )
            sample_posterior = (
                subset_config.get("sample_posterior")
                if isinstance(subset_config, dict)
                else getattr(subset_config, "sample_posterior", None)
            )
            preds_mapped = []
            for p in predictors:
                if f"num__{p}" in self.feature_names_out_:
                    preds_mapped.append(f"num__{p}")
                elif f"cat__{p}" in self.feature_names_out_:
                    preds_mapped.append(f"cat__{p}")
                else:
                    preds_mapped.append(p)

            mapped_subsets[target_mapped] = {
                "predictors": preds_mapped,
                "estimators": estimators,
                "sample_posterior": sample_posterior,
            }
        return mapped_subsets

    def _map_variable_bounds(
        self, preprocessor: ColumnTransformer, domain_knowledge
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Translate original-space variable_bounds into standardized preprocessor space."""
        if (
            not domain_knowledge
            or not getattr(domain_knowledge, "variable_bounds", None)
            or not getattr(self, "feature_names_out_", None)
        ):
            return -np.inf, np.inf

        n_feats = len(self.feature_names_out_)
        min_vec = np.full(n_feats, -np.inf, dtype=float)
        max_vec = np.full(n_feats, np.inf, dtype=float)

        num_transformer = None
        num_cols = []
        if (
            hasattr(preprocessor, "named_transformers_")
            and "num" in preprocessor.named_transformers_
        ):
            num_transformer = preprocessor.named_transformers_["num"]
            for name, _, cols in getattr(preprocessor, "transformers_", []):
                if name == "num":
                    num_cols = list(cols)
                    break

        for col, bounds in domain_knowledge.variable_bounds.items():
            lo = (
                bounds.get("min_value")
                if isinstance(bounds, dict)
                else getattr(bounds, "min_value", None)
            )
            hi = (
                bounds.get("max_value")
                if isinstance(bounds, dict)
                else getattr(bounds, "max_value", None)
            )
            if lo is None and hi is None:
                continue

            feat_name = (
                f"num__{col}"
                if f"num__{col}" in self.feature_names_out_
                else (col if col in self.feature_names_out_ else None)
            )
            if feat_name is None:
                continue
            f_idx = self.feature_names_out_.index(feat_name)

            # Transform bounds through numeric scaler if fitted
            scale = 1.0
            center = 0.0
            if num_transformer is not None and col in num_cols:
                c_idx = num_cols.index(col)
                if (
                    hasattr(num_transformer, "mean_")
                    and num_transformer.mean_ is not None
                ):
                    center = float(num_transformer.mean_[c_idx])
                elif (
                    hasattr(num_transformer, "center_")
                    and num_transformer.center_ is not None
                ):
                    center = float(num_transformer.center_[c_idx])
                if (
                    hasattr(num_transformer, "scale_")
                    and num_transformer.scale_ is not None
                ):
                    s = float(num_transformer.scale_[c_idx])
                    scale = s if s > 0 else 1.0

            if lo is not None:
                min_vec[f_idx] = (float(lo) - center) / scale
            if hi is not None:
                max_vec[f_idx] = (float(hi) - center) / scale

        if np.all(np.isneginf(min_vec)) and np.all(np.isposinf(max_vec)):
            return -np.inf, np.inf
        return min_vec, max_vec

    def _build_pipeline(
        self,
        preprocessor: ColumnTransformer,
        estimator,
        max_iter: int,
        domain_knowledge=None,
        method: str | None = None,
    ) -> Pipeline:
        min_val, max_val = self._map_variable_bounds(preprocessor, domain_knowledge)
        want_posterior = False
        if domain_knowledge and getattr(domain_knowledge, "covariate_subsets", None):
            for sub in domain_knowledge.covariate_subsets.values():
                sp = (
                    sub.get("sample_posterior")
                    if isinstance(sub, dict)
                    else getattr(sub, "sample_posterior", None)
                )
                if sp:
                    want_posterior = True
                    break

        # Gallery domain grids list sklearn imputers as methods.
        if method in _STANDALONE_IMPUTERS:
            imputer = estimator
            # Automatically promote standalone IterativeImputer to MaskedIterativeImputer
            # when domain_knowledge.covariate_subsets is populated!
            if (
                method == "IterativeImputer"
                and domain_knowledge
                and getattr(domain_knowledge, "covariate_subsets", None)
            ):
                mapped_subsets = self._map_covariate_subsets(domain_knowledge)
                inner_est = getattr(imputer, "estimator", None)
                use_post = bool(
                    want_posterior
                    and (
                        inner_est is None or type(inner_est).__name__ == "BayesianRidge"
                    )
                )
                imputer = MaskedIterativeImputer(
                    estimator=inner_est,
                    random_state=self.random_state,
                    max_iter=max_iter,
                    sample_posterior=use_post,
                    min_value=min_val,
                    max_value=max_val,
                    covariate_subsets=mapped_subsets,
                    feature_names=getattr(self, "feature_names_out_", None),
                )
            elif hasattr(imputer, "get_params") and hasattr(imputer, "set_params"):
                params = imputer.get_params()
                updates = {}
                if "random_state" in params:
                    updates["random_state"] = self.random_state
                if "max_iter" in params:
                    updates["max_iter"] = max_iter
                if not np.isscalar(min_val) and "min_value" in params:
                    updates["min_value"] = min_val
                    updates["max_value"] = max_val
                if (
                    method == "MaskedIterativeImputer"
                    and domain_knowledge
                    and domain_knowledge.covariate_subsets
                    and "covariate_subsets" in params
                ):
                    updates["covariate_subsets"] = self._map_covariate_subsets(
                        domain_knowledge
                    )
                    if "feature_names" in params:
                        updates["feature_names"] = self.feature_names_out_
                if updates:
                    imputer.set_params(**updates)
            return Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("imputer", imputer),
                ]
            )

        if (
            domain_knowledge and getattr(domain_knowledge, "covariate_subsets", None)
        ) or isinstance(estimator, CovariateDistributionImputer):
            mapped_subsets = (
                self._map_covariate_subsets(domain_knowledge)
                if domain_knowledge and domain_knowledge.covariate_subsets
                else None
            )
            use_post = bool(
                want_posterior
                and (estimator is None or type(estimator).__name__ == "BayesianRidge")
            )
            imputer = MaskedIterativeImputer(
                estimator=estimator,
                random_state=self.random_state,
                max_iter=max_iter,
                sample_posterior=use_post,
                min_value=min_val,
                max_value=max_val,
                covariate_subsets=mapped_subsets,
                feature_names=getattr(self, "feature_names_out_", None),
            )
        else:
            imputer = IterativeImputer(
                estimator=estimator,
                random_state=self.random_state,
                max_iter=max_iter,
                min_value=min_val,
                max_value=max_val,
            )

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("imputer", imputer),
            ]
        )

    def _select_imputations(self, imputers: list[Pipeline]) -> list[Pipeline]:
        np.random.seed(self.random_state)
        selected_idxs = np.random.choice(
            range(len(imputers)),
            min(self.samples, len(imputers)),
            replace=False,
        )
        return [imputers[idx] for idx in selected_idxs]

    def _apply_imputations(
        self, df: pd.DataFrame, imputers: list[Pipeline]
    ) -> list[np.ndarray]:
        imputations = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for imputer in imputers:
                imputer.fit(df)
                imputations.append(imputer.transform(df))
            return imputations

    @staticmethod
    def _extract_candidate_metadata(imputers: list[Pipeline]) -> list[dict[str, Any]]:
        metadata = []
        for idx, pipe in enumerate(imputers):
            try:
                imp = pipe.named_steps.get("imputer") or pipe["imputer"]
                imp_name = type(imp).__name__
                est = getattr(imp, "estimator", None)
                est_name = type(est).__name__ if est is not None else imp_name
            except (AttributeError, KeyError, TypeError):
                imp_name = "Unknown"
                est_name = "Unknown"
            metadata.append(
                {
                    "index": idx,
                    "method": imp_name,
                    "estimator": est_name,
                }
            )
        return metadata

    def _inverse_transform_candidate(
        self,
        arr: np.ndarray,
        pipeline: Pipeline,
        original_df: pd.DataFrame,
        categorical_columns: list[str],
        numerical_columns: list[str],
    ) -> pd.DataFrame:
        """Inverse-transform a standardized candidate array back to original units and dtypes."""
        domain_knowledge = getattr(self.param_grid, "domain_knowledge", None)
        var_bounds = (
            getattr(domain_knowledge, "variable_bounds", None)
            if domain_knowledge
            else None
        )

        try:
            preprocessor = (
                pipeline.named_steps.get("preprocessor") or pipeline["preprocessor"]
            )
            if not hasattr(preprocessor, "named_transformers_"):
                raise AttributeError("Preprocessor is not fitted.")

            f_names = [str(f) for f in preprocessor.get_feature_names_out()]
            reconstructed = {}

            if numerical_columns and "num" in preprocessor.named_transformers_:
                num_trans = preprocessor.named_transformers_["num"]
                num_idxs = [
                    f_names.index(f"num__{c}")
                    for c in numerical_columns
                    if f"num__{c}" in f_names
                ]
                if len(num_idxs) == len(numerical_columns):
                    X_num_scaled = np.asarray(arr[:, num_idxs], dtype=float)
                    if hasattr(num_trans, "inverse_transform"):
                        X_num_orig = num_trans.inverse_transform(X_num_scaled)
                    else:
                        X_num_orig = X_num_scaled
                    for c_pos, col_name in enumerate(numerical_columns):
                        col_vals = X_num_orig[:, c_pos].copy()
                        # Preserve exact observed values
                        obs_mask = original_df[col_name].notna().to_numpy()
                        if np.any(obs_mask):
                            col_vals[obs_mask] = original_df[col_name].to_numpy()[
                                obs_mask
                            ]
                        # Apply variable_bounds if present
                        if var_bounds and col_name in var_bounds:
                            b = var_bounds[col_name]
                            lo = (
                                b.get("min_value")
                                if isinstance(b, dict)
                                else getattr(b, "min_value", None)
                            )
                            hi = (
                                b.get("max_value")
                                if isinstance(b, dict)
                                else getattr(b, "max_value", None)
                            )
                            if lo is not None or hi is not None:
                                col_vals[~obs_mask] = np.clip(
                                    col_vals[~obs_mask],
                                    -np.inf if lo is None else float(lo),
                                    np.inf if hi is None else float(hi),
                                )
                        orig_dtype = original_df[col_name].dtype
                        if pd.api.types.is_bool_dtype(orig_dtype):
                            reconstructed[col_name] = col_vals >= 0.5
                        elif pd.api.types.is_integer_dtype(orig_dtype):
                            reconstructed[col_name] = np.rint(col_vals).astype(
                                orig_dtype
                            )
                        else:
                            reconstructed[col_name] = col_vals

            if categorical_columns and "cat" in preprocessor.named_transformers_:
                cat_trans = preprocessor.named_transformers_["cat"]
                cat_idxs = [
                    i for i, name in enumerate(f_names) if name.startswith("cat__")
                ]
                if cat_idxs and hasattr(cat_trans, "inverse_transform"):
                    X_cat_encoded = np.asarray(arr[:, cat_idxs], dtype=float)
                    X_cat_orig = cat_trans.inverse_transform(X_cat_encoded)
                    for c_pos, col_name in enumerate(categorical_columns):
                        col_vals = pd.Series(
                            X_cat_orig[:, c_pos], index=original_df.index
                        )
                        obs_mask = original_df[col_name].notna()
                        col_vals.loc[obs_mask] = original_df.loc[obs_mask, col_name]
                        reconstructed[col_name] = col_vals

            if set(reconstructed.keys()) == set(original_df.columns):
                ordered = {col: reconstructed[col] for col in original_df.columns}
                return pd.DataFrame(ordered, index=original_df.index)
        except (AttributeError, KeyError, TypeError, ValueError):
            reconstructed = {}

        # Fallback when preprocessor is mocked or non-standard
        cols = (
            list(original_df.columns)
            if arr.shape[1] == original_df.shape[1]
            else [f"feat_{i}" for i in range(arr.shape[1])]
        )
        idx = original_df.index if arr.shape[0] == len(original_df.index) else None
        return pd.DataFrame(arr, columns=cols, index=idx)

    def generate_descriptors(self) -> list[np.ndarray]:
        return self.magic.generate(self.representations)

    def collapse(
        self,
        strategy: str = "auto",
        *,
        temperature: float | None = 1.0,
        trim_quantile: float = 0.2,
        cluster_id: int | None = None,
        compare_all: bool = False,
    ) -> ECTCollapseResult:
        """
        Agentically navigate and collapse the fitted ECT signature distribution
        without re-fitting the imputation ensemble.
        """
        if not self.magic_descriptors:
            raise RuntimeError("No ECT descriptors available. Call `fit` first.")

        temp_val = (
            getattr(self, "temperature", 1.0)
            if temperature is None
            else float(temperature)
        )
        original_df = getattr(self, "_input_df_", None)
        candidates = getattr(self, "imputed_candidates_", None)
        if not candidates:
            # Construct DataFrames from representations if impute() was mocked
            cols = (
                list(original_df.columns)
                if original_df is not None
                and len(self.representations) > 0
                and self.representations[0].shape[1] == original_df.shape[1]
                else [
                    f"feat_{i}"
                    for i in range(
                        self.representations[0].shape[1] if self.representations else 0
                    )
                ]
            )
            candidates = [pd.DataFrame(r, columns=cols) for r in self.representations]
            if len(candidates) == 1 and len(self.magic_descriptors) > 1:
                candidates = [candidates[0].copy() for _ in self.magic_descriptors]

        domain_knowledge = getattr(self.param_grid, "domain_knowledge", None)
        var_bounds = (
            getattr(domain_knowledge, "variable_bounds", None)
            if domain_knowledge
            else None
        )

        result = collapse_ect_ensemble(
            candidates=candidates,
            descriptors=self.magic_descriptors,
            strategy=strategy,
            temperature=temp_val,
            trim_quantile=trim_quantile,
            cluster_id=cluster_id,
            compare_all=compare_all,
            original_df=original_df,
            standardized_candidates=getattr(
                self, "standardized_candidates_", self.representations
            ),
            candidate_metadata=getattr(self, "candidate_metadata_", None),
            magic_estimator=self.magic,
            preprocessor=(
                self.selected_imputers[0].named_steps.get("preprocessor")
                if getattr(self, "imputed_candidates_", None)
                else None
            ),
            variable_bounds=var_bounds,
        )
        self.ect_collapse_result_ = result
        self.collapse_result_ = result
        self.closest_index = result.selected_index
        if hasattr(self, "selected_imputers") and 0 <= self.closest_index < len(
            self.selected_imputers
        ):
            self.pipeline = self.selected_imputers[self.closest_index]
        return result

    def fit(self, df: pd.DataFrame, max_iter: int = 5) -> pd.DataFrame:
        self._input_df_ = df.copy()
        self.representations = self.impute(df, max_iter)
        self.magic_descriptors = self.generate_descriptors()

        assert len(self.representations) == len(self.magic_descriptors)
        self.closest_index = self._select_representative(self.magic_descriptors)
        self.pipeline = self.selected_imputers[self.closest_index]
        imputed_columns = self._get_imputed_columns(
            transformer=self.pipeline["preprocessor"]
        )

        if getattr(self, "imputed_candidates_", None) and len(
            self.imputed_candidates_
        ) == len(self.representations):
            return self.collapse(
                strategy=self.collapse_strategy,
                temperature=self.temperature,
                trim_quantile=self.trim_quantile,
            ).collapsed_df.copy()

        X = self.representations[self.closest_index]
        return pd.DataFrame(X, columns=imputed_columns)

    def transform(self, df: pd.DataFrame, max_iter: int = 5) -> pd.DataFrame:
        if not hasattr(self, "pipeline"):
            raise RuntimeError("Pipeline not fitted. Call `fit` first.")

        arr = self.pipeline.transform(df)
        categorical_columns, numerical_columns = self._identify_column_types(df)
        inv_df = self._inverse_transform_candidate(
            arr, self.pipeline, df, categorical_columns, numerical_columns
        )
        imputed_columns = self._get_imputed_columns(
            transformer=self.pipeline["preprocessor"]
        )
        if list(inv_df.columns) == list(df.columns):
            return inv_df
        if len(imputed_columns) == inv_df.shape[1]:
            inv_df.columns = imputed_columns
            return inv_df
        return pd.DataFrame(arr, columns=imputed_columns)

    @staticmethod
    def _get_imputed_columns(transformer: ColumnTransformer) -> list[str]:
        return transformer.get_feature_names_out()

    @staticmethod
    def _select_representative(descriptors: list[np.ndarray]) -> int:
        stacked = np.stack(descriptors)
        avg_descriptor = stacked.mean(axis=0)
        norms = np.linalg.norm(
            (stacked - avg_descriptor).reshape(len(descriptors), -1), axis=1
        )
        return int(np.argmin(norms))

    @staticmethod
    def _configure_magic_method(magic: str, config) -> tuple[BaseModel, Magic]:
        magic_method = getattr(METHODS, magic, None)
        if magic_method is None:
            raise ValueError(f"Magic method '{magic}' not found.")
        if not isinstance(config, BaseModel):
            config = MagicGallery.get(magic)
        return config, magic_method(config=config)

    @staticmethod
    def _configure_param_grid(param_grid) -> ImputationConfig:
        if isinstance(param_grid, str):
            grid = GridGallery.get(param_grid)
            return ImputationConfig(
                methods=list(grid.methods),
                modules=list(grid.modules),
                grids=list(grid.grids),
                domain_knowledge=grid.domain_knowledge,
            )
        if isinstance(param_grid, ImputationConfig):
            return param_grid
        if isinstance(param_grid, BaseModel):
            if (
                not hasattr(param_grid, "methods")
                or not hasattr(param_grid, "modules")
                or not hasattr(param_grid, "grids")
            ):
                raise ValueError("Invalid parameter grid configuration.")
            return ImputationConfig(
                methods=param_grid.methods,
                modules=param_grid.modules,
                grids=param_grid.grids,
                domain_knowledge=getattr(param_grid, "domain_knowledge", None),
            )
        if isinstance(param_grid, dict):
            if not all(key in param_grid for key in ["methods", "modules", "grids"]):
                raise ValueError("Invalid parameter grid configuration.")
            dk = param_grid.get("domain_knowledge")
            if isinstance(dk, dict):
                dk = DomainKnowledge(**dk)
            return ImputationConfig(
                methods=param_grid["methods"],
                modules=param_grid["modules"],
                grids=param_grid["grids"],
                domain_knowledge=dk,
            )
        raise ValueError("Invalid parameter grid type.")

    def plot_mds(self, **kwargs):
        """Visualize the ECT descriptor space via MDS after ``fit()``."""
        from phil.visualization import plot_mds as _plot_mds

        if not self.magic_descriptors:
            raise RuntimeError("Call fit() before plot_mds().")
        return _plot_mds(self.magic_descriptors, self.closest_index, **kwargs)

    @staticmethod
    def _configure_preprocessor(
        strategy: str,
        categorical_columns: list[str],
        numerical_columns: list[str],
    ) -> ColumnTransformer:
        strategy = ProcessingGallery.get(strategy)
        transformers: list[tuple[str, Any, list[str]]] = []

        for key, preprocessing_config in strategy.items():
            try:
                model = Phil._import_model(
                    preprocessing_config.module, preprocessing_config.method
                )
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    f"Failed to import model {preprocessing_config.method} from module {preprocessing_config.module}: {e}"
                )

            transformer = model(**preprocessing_config.params)

            if key == "num" and len(numerical_columns) > 0:
                transformers.append((key, transformer, numerical_columns))
            elif key == "cat" and len(categorical_columns) > 0:
                transformers.append((key, transformer, categorical_columns))

        return ColumnTransformer(transformers, verbose_feature_names_out=True)
