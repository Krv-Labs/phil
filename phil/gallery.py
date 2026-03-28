"""Collection of predefined configurations for Phil."""

from typing import Dict

import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import ParameterGrid

from phil.imputation import ImputationConfig, PreprocessingConfig
from phil.magic import ECTConfig


class GridGallery:
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
        "finance": ImputationConfig(
            methods=["IterativeImputer", "KNNImputer", "SimpleImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid({"estimator": ["BayesianRidge"], "max_iter": [10, 50]}),
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
                    {"estimator": ["RandomForestRegressor"], "max_iter": [10, 20]}
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
                    {"estimator": ["GradientBoostingRegressor"], "max_iter": [10, 30]}
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
                    {"estimator": ["DecisionTreeRegressor"], "max_iter": [10, 20]}
                ),
            ],
        ),
    }

    @classmethod
    def get(cls, name: str) -> ImputationConfig:
        return cls._grids.get(name, cls._grids["default"])


class ProcessingGallery:
    _numeric_methods = {
        "default": PreprocessingConfig(method="StandardScaler"),
        "finance": PreprocessingConfig(
            method="MinMaxScaler", params={"feature_range": [(0, 1)]}
        ),
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
        "marketing": PreprocessingConfig(
            method="OneHotEncoder",
            params={"sparse": [False], "handle_unknown": ["ignore"]},
        ),
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
