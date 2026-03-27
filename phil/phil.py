import importlib
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline

from phil.gallery import GridGallery, MagicGallery, ProcessingGallery
from phil.imputation import ImputationConfig
from phil.magic import Magic
import phil.magic as METHODS


class Phil:
    def __init__(
        self,
        samples: int = 30,
        param_grid: str = "default",
        magic: str = "ECT",
        config=None,
        random_state=None,
    ):
        self.config, self.magic = self._configure_magic_method(
            magic=magic, config=config
        )
        self.samples = samples
        self.param_grid = self._configure_param_grid(param_grid)
        self.random_state = random_state
        self.representations = []
        self.magic_descriptors = []

    def impute(self, df: pd.DataFrame, max_iter: int = 10) -> List[np.ndarray]:
        if df.isnull().sum().sum() == 0:
            raise ValueError("No missing values found in the input DataFrame.")
        categorical_columns, numerical_columns = self._identify_column_types(df)
        preprocessor = self._configure_preprocessor(
            "default", categorical_columns, numerical_columns
        )
        imputers = self._create_imputers(preprocessor, max_iter)
        self.selected_imputers = self._select_imputations(imputers)
        return self._apply_imputations(df, self.selected_imputers)

    @staticmethod
    def _identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical_columns = df.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        return categorical_columns, numerical_columns

    def _create_imputers(
        self, preprocessor: ColumnTransformer, max_iter: int
    ) -> List[Pipeline]:
        imputers = []
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
                estimator = model(**compatible_params)
                imputers.append(self._build_pipeline(preprocessor, estimator, max_iter))
        return imputers

    @staticmethod
    def _import_model(module: str, method: str):
        imported_module = importlib.import_module(module)
        return getattr(imported_module, method)

    def _build_pipeline(
        self, preprocessor: ColumnTransformer, estimator, max_iter: int
    ) -> Pipeline:
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "imputer",
                    IterativeImputer(
                        estimator=estimator,
                        random_state=self.random_state,
                        max_iter=max_iter,
                    ),
                ),
            ]
        )

    def _select_imputations(self, imputers: List[Pipeline]) -> List[Pipeline]:
        np.random.seed(self.random_state)
        selected_idxs = np.random.choice(
            range(len(imputers)),
            min(self.samples, len(imputers)),
            replace=False,
        )
        return [imputers[idx] for idx in selected_idxs]

    def _apply_imputations(
        self, df: pd.DataFrame, imputers: List[Pipeline]
    ) -> List[np.ndarray]:
        imputations = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for imputer in imputers:
                imputer.fit(df)
                imputations.append(imputer.transform(df))
            return imputations

    def generate_descriptors(self) -> List[np.ndarray]:
        return self.magic.generate(self.representations)

    def fit(self, df: pd.DataFrame, max_iter: int = 5) -> pd.DataFrame:
        self.representations = self.impute(df, max_iter)
        self.magic_descriptors = self.generate_descriptors()

        assert len(self.representations) == len(self.magic_descriptors)
        self.closest_index = self._select_representative(self.magic_descriptors)
        X = self.representations[self.closest_index]
        self.pipeline = self.selected_imputers[self.closest_index]
        imputed_columns = self._get_imputed_columns(
            transformer=self.pipeline["preprocessor"]
        )
        return pd.DataFrame(X, columns=imputed_columns)

    def transform(self, df: pd.DataFrame, max_iter: int = 5) -> pd.DataFrame:
        if not hasattr(self, "pipeline"):
            raise RuntimeError("Pipeline not fitted. Call `fit` first.")

        imputed_columns = self._get_imputed_columns(
            transformer=self.pipeline["preprocessor"]
        )
        return pd.DataFrame(self.pipeline.transform(df), columns=imputed_columns)

    @staticmethod
    def _get_imputed_columns(transformer: ColumnTransformer) -> List[str]:
        return transformer.get_feature_names_out()

    @staticmethod
    def _select_representative(descriptors: List[np.ndarray]) -> int:
        stacked = np.stack(descriptors)
        avg_descriptor = stacked.mean(axis=0)
        norms = np.linalg.norm(
            (stacked - avg_descriptor).reshape(len(descriptors), -1), axis=1
        )
        return int(np.argmin(norms))

    @staticmethod
    def _configure_magic_method(magic: str, config) -> Tuple[BaseModel, Magic]:
        magic_method = getattr(METHODS, magic, None)
        if magic_method is None:
            raise ValueError(f"Magic method '{magic}' not found.")
        if not isinstance(config, BaseModel):
            config = MagicGallery.get(magic)
        return config, magic_method(config=config)

    @staticmethod
    def _configure_param_grid(param_grid) -> ImputationConfig:
        if isinstance(param_grid, str):
            return GridGallery.get(param_grid)
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
            )
        if isinstance(param_grid, dict):
            if not all(key in param_grid for key in ["methods", "modules", "grids"]):
                raise ValueError("Invalid parameter grid configuration.")
            return ImputationConfig(
                methods=param_grid["methods"],
                modules=param_grid["modules"],
                grids=param_grid["grids"],
            )
        raise ValueError("Invalid parameter grid type.")

    @staticmethod
    def _configure_preprocessor(
        strategy: str,
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> ColumnTransformer:
        strategy = ProcessingGallery.get(strategy)
        transformers: List[Tuple[str, Any, List[str]]] = []

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
