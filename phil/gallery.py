"""Collection of predefined configurations for Phil."""

from typing import Dict

import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import ParameterGrid

from phil.imputation import ImputationConfig, PreprocessingConfig
from phil.magic import ECTConfig


class GridGallery:
    """
    Collection of imputation grids optimized for specific domains.

    Citations:
    - Sampling/Multiverse: Wayland et al. (2025) - https://www.nature.com/articles/s41560-025-01871-0
    - Finance: Gu, Kelly, & Xiu (2020) on ML for asset pricing and robust ML portfolios.
    - Healthcare: Stekhoven & Bühlmann (2011) on MissForest and Chen et al. (2023) on clinical imputation.
    - Marketing: Anand & Mamidi (2020) / Zhang et al. (2025) on ML for consumer analytics.
    - Engineering: Thomas & Rajabi (2021) and Idri et al. (2016) on systematic reviews of engineering data.
    """

    _grids = {
        "default": ImputationConfig(
            methods=[
                "BayesianRidge",
                "DecisionTreeRegressor",
                "RandomForestRegressor",
                "GradientBoostingRegressor",
            ],
            modules=[
                "sklearn.linear_model",
                "sklearn.tree",
                "sklearn.ensemble",
                "sklearn.ensemble",
            ],
            grids=[
                ParameterGrid({"alpha": [1.0, 0.1, 0.01]}),
                ParameterGrid(
                    {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
                ),
                ParameterGrid({"n_estimators": [10, 50], "max_depth": [None, 5]}),
                ParameterGrid(
                    {"learning_rate": [0.1, 0.01], "n_estimators": [50, 100]}
                ),
            ],
        ),
        "sampling": ImputationConfig(
            methods=["DistributionImputer"],
            modules=["phil.imputation"],
            grids=[ParameterGrid({"random_state": np.arange(0, 100, 1)})],
        ),
        "covariate_sampling": ImputationConfig(
            methods=["CovariateDistributionImputer"],
            modules=["phil.imputation"],
            grids=[
                ParameterGrid(
                    {
                        "n_neighbors": [3, 5, 10],
                        "random_state": list(range(10)),
                    }
                )
            ],
        ),
        "finance": ImputationConfig(
            methods=["IterativeImputer", "KNNImputer", "SimpleImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid(
                    {
                        "estimator": [
                            "BayesianRidge",
                            "RandomForestRegressor",
                            "KNeighborsRegressor",
                        ],
                        "max_iter": [10],
                    }
                ),
                ParameterGrid(
                    {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]}
                ),
                ParameterGrid({"strategy": ["mean", "median"]}),
            ],
        ),
        "healthcare": ImputationConfig(
            methods=["KNNImputer", "SimpleImputer", "IterativeImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid({"n_neighbors": [5, 10], "weights": ["distance"]}),
                ParameterGrid({"strategy": ["median", "most_frequent"]}),
                ParameterGrid(
                    {
                        "estimator": [
                            "BayesianRidge",
                            "RandomForestRegressor",
                            "ExtraTreesRegressor",
                        ],
                        "max_iter": [10],
                    }
                ),
            ],
        ),
        "marketing": ImputationConfig(
            methods=["SimpleImputer", "KNNImputer", "IterativeImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid(
                    {
                        "strategy": ["most_frequent", "constant"],
                        "fill_value": ["unknown"],
                    }
                ),
                ParameterGrid({"n_neighbors": [3, 5], "weights": ["uniform"]}),
                ParameterGrid(
                    {
                        "estimator": [
                            "BayesianRidge",
                            "GradientBoostingRegressor",
                            "DecisionTreeRegressor",
                        ],
                        "max_iter": [10],
                    }
                ),
            ],
        ),
        "engineering": ImputationConfig(
            methods=["SimpleImputer", "KNNImputer", "IterativeImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid({"strategy": ["mean", "median"]}),
                ParameterGrid({"n_neighbors": [3, 5, 7], "weights": ["distance"]}),
                ParameterGrid(
                    {
                        "estimator": [
                            "BayesianRidge",
                            "DecisionTreeRegressor",
                            "ExtraTreesRegressor",
                        ],
                        "max_iter": [10],
                    }
                ),
            ],
        ),
    }

    @classmethod
    def get(cls, name: str) -> ImputationConfig:
        return cls._grids.get(name, cls._grids["default"])


class ProcessingGallery:
    """
    Collection of preprocessing configurations optimized for specific domains.

    Citations:
    - Finance: RobustScaler for handling outliers in financial time series and asset data.
    - Marketing: TargetEncoder for high-cardinality features (e.g., zip codes, product IDs)
      as discussed in Anand & Mamidi (2020).
    """

    _numeric_methods = {
        "default": PreprocessingConfig(method="StandardScaler"),
        "finance": PreprocessingConfig(method="RobustScaler"),
        "healthcare": PreprocessingConfig(method="RobustScaler"),
        "marketing": PreprocessingConfig(
            method="PowerTransformer", params={"method": ["yeo-johnson"]}
        ),
        "engineering": PreprocessingConfig(method="StandardScaler"),
    }

    _categorical_methods = {
        "default": PreprocessingConfig(method="OneHotEncoder"),
        "finance": PreprocessingConfig(
            method="OneHotEncoder", params={"handle_unknown": ["ignore"]}
        ),
        "healthcare": PreprocessingConfig(
            method="OrdinalEncoder",
            params={"handle_unknown": ["use_encoded_value"]},
        ),
        "marketing": PreprocessingConfig(method="TargetEncoder"),
    }

    @classmethod
    def get(cls, name: str = "default") -> Dict[str, PreprocessingConfig]:
        return {
            "num": cls._numeric_methods.get(name, cls._numeric_methods["default"]),
            "cat": cls._categorical_methods.get(
                name, cls._categorical_methods["default"]
            ),
        }


class MagicGallery:
    @staticmethod
    def get(method: str) -> BaseModel:
        if method == "ECT":
            return ECTConfig(
                num_thetas=64,
                radius=1.0,
                resolution=100,
                scale=500,
                seed=42,
            )
        raise ValueError(f"Unknown magic method: {method}")
