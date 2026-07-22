"""Rule-based imputation-grid recommender for MCP agents."""

from __future__ import annotations

from typing import Any

import pandas as pd

from phil.gallery import GRID_METADATA, get_grid_metadata

_HIGH_CARDINALITY_UNIQUE = 50
_LARGE_N_ROWS = 100_000
_EXTREME_MISSING_PCT = 50.0


def _is_categorical_series(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _is_high_cardinality_id(series: pd.Series) -> bool:
    """Integer columns with many distinct values behave like categorical IDs."""
    if not pd.api.types.is_integer_dtype(series):
        return False
    return int(series.nunique(dropna=True)) >= _HIGH_CARDINALITY_UNIQUE


def _frame_metrics(df: pd.DataFrame) -> dict[str, Any]:
    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    total_missing = int(df.isna().sum().sum())
    overall_missing_pct = (
        round(total_missing / (n_rows * n_cols) * 100, 3)
        if n_rows and n_cols
        else 0.0
    )

    categorical_columns: list[str] = []
    high_cardinality_columns: list[str] = []
    for col in df.columns:
        series = df[col]
        name = str(col)
        n_unique = int(series.nunique(dropna=True))
        if _is_categorical_series(series):
            categorical_columns.append(name)
            if n_unique >= _HIGH_CARDINALITY_UNIQUE:
                high_cardinality_columns.append(name)
        elif _is_high_cardinality_id(series):
            # ZIP / product-id style ints: categorical for grid choice.
            categorical_columns.append(name)
            high_cardinality_columns.append(name)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_missing": total_missing,
        "overall_missing_pct": overall_missing_pct,
        "n_categorical": len(categorical_columns),
        "categorical_columns": categorical_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "has_high_cardinality": bool(high_cardinality_columns),
    }


def _metadata_fields(grid_name: str) -> dict[str, Any]:
    meta = get_grid_metadata(grid_name)
    if meta is None:
        # Should not happen for built-in recommendations; keep payload stable.
        return {"name": grid_name}
    return {
        "name": meta.name,
        "target_domain": meta.target_domain,
        "intent": meta.intent,
        "suitability": meta.suitability,
        "data_type_affinity": list(meta.data_type_affinity),
        "time_complexity": meta.time_complexity,
        "scale_limits": meta.scale_limits,
    }


def recommend_grid_for_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Choose a built-in grid from dataframe shape / missingness heuristics."""
    metrics = _frame_metrics(df)
    warnings: list[str] = []
    rationale: list[str] = []

    n_rows = metrics["n_rows"]
    overall_missing_pct = metrics["overall_missing_pct"]
    n_categorical = metrics["n_categorical"]
    has_high_cardinality = metrics["has_high_cardinality"]

    if n_rows > _LARGE_N_ROWS:
        recommended = "sampling" if n_categorical == 0 else "default"
        suggested_samples = 10
        warnings.append(
            "N > 100,000: avoid KNN-heavy grids (`healthcare`, `finance`) "
            "unless you reduce samples or subsample rows."
        )
        rationale.append(
            f"Large table ({n_rows:,} rows); prefer a lower-overhead grid "
            f"(`{recommended}`) and keep samples modest."
        )
        if n_categorical == 0:
            rationale.append(
                "No categorical columns detected; `sampling` preserves "
                "marginals without KNN quadratic cost."
            )
        else:
            rationale.append(
                "Categorical columns present; `default` is a safer mixed-type "
                "starting point than KNN-heavy domain grids."
            )
    elif has_high_cardinality:
        recommended = "marketing"
        suggested_samples = 15
        rationale.append(
            "High-cardinality categorical columns detected "
            f"({', '.join(metrics['high_cardinality_columns'][:5])}"
            f"{'…' if len(metrics['high_cardinality_columns']) > 5 else ''}); "
            "`marketing` pairs with TargetEncoder-friendly preprocessing."
        )
    elif overall_missing_pct > _EXTREME_MISSING_PCT:
        recommended = "default"
        suggested_samples = 20
        warnings.append(
            "Overall missingness > 50%: tree-based / iterative regressors may "
            "struggle to converge; monitor diagnose_sweep spread and consider "
            "dropping ultra-sparse columns."
        )
        rationale.append(
            f"Extreme missingness ({overall_missing_pct}%); start with "
            "`default` and a moderate sample budget."
        )
    else:
        recommended = "default"
        suggested_samples = 20
        rationale.append(
            "No scale, cardinality, or extreme-missingness flags; "
            "`default` is the general-purpose starting grid."
        )

    # Defensive: every recommendation must exist in the declarative catalog.
    if recommended not in GRID_METADATA:
        recommended = "default"

    recommendation = (
        f"Recommended Grid: {recommended}, suggested samples: "
        f"{suggested_samples} to preserve performance."
    )

    return {
        "status": "ok",
        "recommended_grid": recommended,
        "suggested_samples": suggested_samples,
        "recommendation": recommendation,
        "warnings": warnings,
        "rationale": rationale,
        "metrics": metrics,
        "grid": _metadata_fields(recommended),
    }
