"""Collection of predefined configurations for Phil."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import ParameterGrid

from phil.imputation import ImputationConfig, PreprocessingConfig
from phil.magic import ECTConfig


@dataclass(frozen=True)
class GridMetadata:
    """Declarative, agent-readable metadata for a built-in imputation grid."""

    name: str
    target_domain: str
    intent: str
    suitability: str
    data_type_affinity: List[str]
    time_complexity: str  # "Low" | "Medium" | "High"
    scale_limits: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GRID_METADATA: dict[str, GridMetadata] = {
    "default": GridMetadata(
        name="default",
        target_domain="general",
        intent=(
            "Mixed regression imputers across BayesianRidge, Decision Tree, "
            "Random Forest, and Gradient Boosting."
        ),
        suitability=(
            "Good starting point for mixed continuous tabular data when no "
            "domain-specific prior applies."
        ),
        data_type_affinity=["continuous", "mixed"],
        time_complexity="Medium",
        scale_limits=(
            "Comfortable under ~100k rows. Tree ensembles dominate cost; "
            "lower samples if sweeps are slow."
        ),
    ),
    "sampling": GridMetadata(
        name="sampling",
        target_domain="distributional",
        intent=(
            "Distribution sampling imputation across 100 seeds "
            "(good for preserving marginals)."
        ),
        suitability=(
            "Prefer when the goal is to preserve marginal distributions "
            "rather than conditional structure (multiverse-style sampling)."
        ),
        data_type_affinity=["continuous"],
        time_complexity="Medium",
        scale_limits=(
            "Scales with sample count and seeds. Safer than KNN on large N; "
            "still reduce samples on very wide tables."
        ),
    ),
    "finance": GridMetadata(
        name="finance",
        target_domain="finance",
        intent=("Iterative + KNN + Simple imputers tuned for tabular financial data."),
        suitability=(
            "Use for asset/returns-style tables with outliers and correlated "
            "continuous metrics (pairs well with RobustScaler preprocessing)."
        ),
        data_type_affinity=["continuous", "correlated"],
        time_complexity="High",
        scale_limits=(
            "KNN and iterative estimators dominate; keep rows <100k or reduce "
            "samples. Avoid blind sweeps on half-million-row tables."
        ),
    ),
    "healthcare": GridMetadata(
        name="healthcare",
        target_domain="healthcare",
        intent="KNN, Simple, and Iterative imputers tuned for clinical tables.",
        suitability=(
            "Required for medical metrics with non-Gaussian distributions "
            "(ordinal scaling + robust bounds)."
        ),
        data_type_affinity=["continuous", "ordinal", "mixed"],
        time_complexity="High",
        scale_limits=(
            "Overhead scales quadratically O(N^2) due to KNN. Keep row count "
            "<100,000 or reduce sample size."
        ),
    ),
    "marketing": GridMetadata(
        name="marketing",
        target_domain="marketing",
        intent="Categorical-friendly Simple, KNN, and Iterative imputers.",
        suitability=(
            "Choose for consumer analytics with high-cardinality categoricals "
            "(zip codes, product IDs); pairs with TargetEncoder preprocessing."
        ),
        data_type_affinity=["categorical", "high-cardinality", "mixed"],
        time_complexity="Medium",
        scale_limits=(
            "KNN still present but lighter neighbor grids. Watch cardinality "
            "explosion; reduce samples on wide categorical tables."
        ),
    ),
    "engineering": GridMetadata(
        name="engineering",
        target_domain="engineering",
        intent=(
            "Robust mean/median + KNN + Decision-Tree iterative imputation "
            "for sensor data."
        ),
        suitability=(
            "Prefer for sensor / industrial measurements where mean/median "
            "baselines plus modest KNN coverage are enough."
        ),
        data_type_affinity=["continuous", "sensor"],
        time_complexity="Medium",
        scale_limits=(
            "Moderate KNN cost. Suitable under ~100k rows; lower samples if "
            "neighbor search dominates runtime."
        ),
    ),
}


def get_grid_metadata(name: str) -> GridMetadata | None:
    """Return metadata for a built-in grid, or ``None`` if unknown."""
    return GRID_METADATA.get(name)


def list_grid_metadata() -> list[GridMetadata]:
    """Return metadata for all agent-facing built-in grids (sorted by name)."""
    return [GRID_METADATA[name] for name in sorted(GRID_METADATA)]


def render_imputation_matrix() -> str:
    """Compile a Markdown comparison matrix from ``GRID_METADATA``."""
    headers = (
        "Grid",
        "Domain",
        "Complexity",
        "Affinity",
        "Scale limits",
        "Suitability",
    )
    lines = [
        "# Phil Imputation Grid Matrix",
        "",
        "Compiled from declarative ``GRID_METADATA`` (single source of truth).",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for meta in list_grid_metadata():
        affinity = ", ".join(meta.data_type_affinity)
        # Escape pipes so Markdown tables stay intact.
        suitability = meta.suitability.replace("|", "\\|")
        scale = meta.scale_limits.replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{meta.name}`",
                    meta.target_domain,
                    meta.time_complexity,
                    affinity,
                    scale,
                    suitability,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Estimator / complexity notes")
    lines.append("")
    lines.append("| Grid | Methods (summary) | Big-O / cost notes |")
    lines.append("| --- | --- | --- |")
    lines.append(
        "| `default` | BayesianRidge, trees, forests, GBM | "
        "Ensemble fits ~O(N log N · trees); Medium |"
    )
    lines.append(
        "| `sampling` | DistributionImputer (many seeds) | "
        "Linear in N · seeds; Medium |"
    )
    lines.append("| `finance` | Iterative + KNN + Simple | KNN ~O(N²); High |")
    lines.append("| `healthcare` | KNN + Simple + Iterative | KNN ~O(N²); High |")
    lines.append(
        "| `marketing` | Simple + KNN + Iterative | Lighter KNN grids; Medium |"
    )
    lines.append("| `engineering` | Simple + KNN + Iterative | Modest KNN; Medium |")
    lines.append("")
    return "\n".join(lines)


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
                        "strategy": ["most_frequent", "mean", "median"],
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
