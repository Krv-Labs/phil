"""Rule-based imputation-grid recommender for MCP agents."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from phil.gallery import GRID_METADATA, get_grid_metadata, grid_scalability

_HIGH_CARDINALITY_UNIQUE = 50
_LARGE_N_ROWS = 100_000
_EXTREME_MISSING_PCT = 50.0

# Floor for Phil.samples when missingness is low.
_M_EFFICIENCY_FLOOR = 5
# Matches the highest non-KNN scalability_cap below.
_M_STABILITY_CAP = 30

# Integer columns only count as IDs when the name looks ID-like.
_ID_NAME_RE = re.compile(r"(?i)(^id$|_id$|^zip(_|$)|_code$|^code$)")


def _is_categorical_series(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _is_high_cardinality_id(series: pd.Series, *, name: str, n_unique: int) -> bool:
    """Integer ID columns (ZIP, product_id, …), not ordinary measurements.

    Requires an ID-like column name plus high cardinality. Raw uniqueness
    alone misroutes ``salary`` / ``age``-style integer measurements.
    """
    if not pd.api.types.is_integer_dtype(series):
        return False
    if n_unique < _HIGH_CARDINALITY_UNIQUE:
        return False
    return bool(_ID_NAME_RE.search(name))


def _frame_metrics(df: pd.DataFrame) -> dict[str, Any]:
    n_rows = len(df)
    n_cols = int(df.shape[1])
    total_missing = int(df.isna().sum().sum())
    overall_missing_pct = (
        round(total_missing / (n_rows * n_cols) * 100, 3) if n_rows and n_cols else 0.0
    )
    if n_rows and n_cols:
        col_missing_pct = df.isna().mean() * 100.0
        max_col_missing_pct = float(round(col_missing_pct.max(), 3))
        n_sparse_cols = int((col_missing_pct > _EXTREME_MISSING_PCT).sum())
    else:
        max_col_missing_pct = 0.0
        n_sparse_cols = 0

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
        elif _is_high_cardinality_id(series, name=name, n_unique=n_unique):
            categorical_columns.append(name)
            high_cardinality_columns.append(name)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_missing": total_missing,
        "overall_missing_pct": overall_missing_pct,
        "max_col_missing_pct": max_col_missing_pct,
        "n_sparse_cols": n_sparse_cols,
        "n_categorical": len(categorical_columns),
        "categorical_columns": categorical_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "has_high_cardinality": bool(high_cardinality_columns),
        "fmi_proxy": round(overall_missing_pct / 100.0, 4),
    }


def _metadata_fields(grid_name: str) -> dict[str, Any]:
    meta = get_grid_metadata(grid_name)
    if meta is None:
        return {"name": grid_name}
    return meta.to_dict()


def _sample_budget(
    *,
    n_rows: int,
    missing_pct: float,
    has_knn: bool,
    grid_name: str,
) -> dict[str, Any]:
    """Heuristic Phil ``samples`` budget from missingness and grid cost.

    Phil ``samples`` is multiverse coverage of the candidate grid, not Rubin's
    multiple-imputation *m*. We still raise the budget with cell-missing rate,
    then cap by scalability (large N, KNN, sampling gallery).
    """
    fmi_proxy = missing_pct / 100.0
    m_from_missing = max(_M_EFFICIENCY_FLOOR, math.ceil(100 * fmi_proxy))
    m_stability_uncapped = m_from_missing
    m_stability = min(m_from_missing, _M_STABILITY_CAP)

    if n_rows > _LARGE_N_ROWS:
        scalability_cap = 8 if has_knn else 12
    elif n_rows > 20_000:
        scalability_cap = 10 if has_knn else 20
    elif has_knn:
        scalability_cap = 15
    else:
        scalability_cap = 30

    # The sampling gallery already expands ~100 seeds; keep Phil.samples modest.
    if grid_name == "sampling":
        scalability_cap = min(scalability_cap, 12)

    suggested = int(min(m_stability, scalability_cap))
    return {
        "fmi_proxy": round(fmi_proxy, 4),
        "m_efficiency_floor": _M_EFFICIENCY_FLOOR,
        "m_stability_uncapped": m_stability_uncapped,
        "scalability_cap": scalability_cap,
        "suggested_samples": suggested,
        "literature_notes": [
            "Heuristic: start near 5 samples when missingness is low.",
            (
                "Raise samples roughly with overall % missing, then cap for "
                "runtime (large N, KNN, or the sampling gallery)."
            ),
            (
                "Phil samples ≈ multiverse coverage of the candidate grid, "
                "not classical MI pooling size m."
            ),
        ],
    }


def _subsample_advice(n_rows: int, has_knn: bool) -> dict[str, Any] | None:
    if n_rows <= _LARGE_N_ROWS and not (has_knn and n_rows > 20_000):
        return None
    target = 50_000 if has_knn else 100_000
    target = min(target, n_rows)
    return {
        "recommended": n_rows > target,
        "suggested_rows": min(n_rows, target),
        "reason": (
            "Subsample rows before KNN/iterative sweeps on large N; "
            "fit on the subset, then optionally refit the chosen method."
            if has_knn
            else "Large N: consider a representative subsample for the sweep, "
            "then apply the selected imputer to the full frame."
        ),
    }


def _scalable_alternatives(recommended: str, n_rows: int) -> list[dict[str, Any]]:
    """Cheaper grids an agent can fall back to when compute is tight."""
    alts: list[str] = []
    if recommended in {"healthcare", "finance", "marketing", "engineering"}:
        alts.extend(["default", "sampling"])
    elif recommended == "default" and n_rows > _LARGE_N_ROWS:
        alts.append("sampling")
    elif recommended == "sampling":
        alts.append("default")

    out: list[dict[str, Any]] = []
    for name in alts:
        if name == recommended or name not in GRID_METADATA:
            continue
        scale = grid_scalability(name)
        meta = _metadata_fields(name)
        out.append(
            {
                "grid": name,
                "cost_tier": scale["cost_tier"],
                "has_knn": scale["has_knn"],
                "grid_candidate_count": scale["grid_candidate_count"],
                "why": meta.get("suitability", ""),
            }
        )
    return out


def recommend_grid_for_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Choose a built-in grid from dataframe shape / missingness heuristics."""
    metrics = _frame_metrics(df)
    warnings: list[str] = []
    rationale: list[str] = []

    n_rows = metrics["n_rows"]
    overall_missing_pct = metrics["overall_missing_pct"]
    max_col_missing_pct = metrics["max_col_missing_pct"]
    n_categorical = metrics["n_categorical"]
    has_high_cardinality = metrics["has_high_cardinality"]

    if n_rows > _LARGE_N_ROWS:
        recommended = "sampling" if n_categorical == 0 else "default"
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
        rationale.append(
            "High-cardinality categorical columns detected "
            f"({', '.join(metrics['high_cardinality_columns'][:5])}"
            f"{'…' if len(metrics['high_cardinality_columns']) > 5 else ''}); "
            "`marketing` targets high-cardinality / mixed consumer tables."
        )
    elif max_col_missing_pct > _EXTREME_MISSING_PCT:
        recommended = "default"
        rationale.append(
            f"Extreme per-column missingness (max {max_col_missing_pct}%); "
            "start with `default` and raise samples with missingness."
        )
    else:
        recommended = "default"
        rationale.append(
            "No scale, cardinality, or extreme-missingness flags; "
            "`default` is the general-purpose starting grid."
        )

    # Warnings accumulate independently of which branch picked the grid.
    if (
        max_col_missing_pct > _EXTREME_MISSING_PCT
        or metrics["n_sparse_cols"] > 0
        or overall_missing_pct > _EXTREME_MISSING_PCT
    ):
        warnings.append(
            "Column(s) with >50% missing: tree-based / iterative regressors may "
            "struggle to converge; monitor diagnose_sweep spread and consider "
            "dropping ultra-sparse columns."
        )

    if recommended not in GRID_METADATA:
        recommended = "default"

    scalability = grid_scalability(recommended)
    sample_budget = _sample_budget(
        n_rows=n_rows,
        missing_pct=overall_missing_pct,
        has_knn=bool(scalability["has_knn"]),
        grid_name=recommended,
    )
    suggested_samples = int(sample_budget["suggested_samples"])
    subsample = _subsample_advice(n_rows, bool(scalability["has_knn"]))
    alternatives = _scalable_alternatives(recommended, n_rows)

    if scalability["has_knn"] and n_rows > 20_000:
        warnings.append(
            "Recommended grid includes KNN (≈O(N²)); prefer subsampling or "
            "switch to a low-cost alternative if the sweep is slow."
        )

    rationale.append(
        "Sample budget heuristic (floor≈5; raise with missingness proxy="
        f"{sample_budget['fmi_proxy']}) then cap for scalability "
        f"(cap={sample_budget['scalability_cap']})."
    )

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
        "scalability": scalability,
        "sample_budget": sample_budget,
        "subsample": subsample,
        "scalable_alternatives": alternatives,
    }
