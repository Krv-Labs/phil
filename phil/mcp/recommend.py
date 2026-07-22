"""Rule-based imputation-grid recommender for MCP agents."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from phil.gallery import GRID_METADATA, GridGallery, get_grid_metadata

_HIGH_CARDINALITY_UNIQUE = 50
_LARGE_N_ROWS = 100_000
_EXTREME_MISSING_PCT = 50.0

# Classical MI: a small m often suffices for point-estimate efficiency (Rubin).
_M_EFFICIENCY_FLOOR = 5
# Practical ceiling before Phil sweeps become an expensive multiverse.
_M_STABILITY_CAP = 40


def _is_categorical_series(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _is_high_cardinality_id(series: pd.Series) -> bool:
    """Integer ID columns with many distinct values (ZIP, product_id, …).

    Only integer / nullable-integer dtypes qualify. Integral *floats*
    (lab values like glucose after CSV+NA promotion) are measurements,
    not categoricals — do not route them to ``marketing``.
    Prefer pandas ``Int64`` for ID columns that may contain NA.
    """
    if not pd.api.types.is_integer_dtype(series):
        return False
    return int(series.nunique(dropna=True)) >= _HIGH_CARDINALITY_UNIQUE


def _frame_metrics(df: pd.DataFrame) -> dict[str, Any]:
    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    total_missing = int(df.isna().sum().sum())
    overall_missing_pct = (
        round(total_missing / (n_rows * n_cols) * 100, 3) if n_rows and n_cols else 0.0
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
        # Proxy for Rubin's fraction of missing information (γ) when FMI
        # is unavailable: overall cell-missing rate.
        "fmi_proxy": round(overall_missing_pct / 100.0, 4),
    }


def _metadata_fields(grid_name: str) -> dict[str, Any]:
    meta = get_grid_metadata(grid_name)
    if meta is None:
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


def _grid_scalability(grid_name: str) -> dict[str, Any]:
    """Expose candidate count / KNN risk so agents can budget compute."""
    grid = GridGallery.get(grid_name)
    methods = list(grid.methods)
    n_candidates = sum(len(list(param_grid)) for param_grid in grid.grids)
    has_knn = any("KNN" in method for method in methods)
    has_iterative = any("Iterative" in method for method in methods)
    has_tree_ensemble = any(
        method
        in {
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "ExtraTreesRegressor",
        }
        for method in methods
    )
    if has_knn:
        big_o = "O(N^2) neighbor search dominates when KNN is in the grid"
        cost_tier = "high" if n_candidates >= 7 else "medium_high"
    elif has_tree_ensemble:
        big_o = "O(N log N · trees) per iterative/ensemble fit"
        cost_tier = "medium"
    elif has_iterative:
        big_o = "O(N · P · iters) chained / iterative updates"
        cost_tier = "medium"
    else:
        big_o = "Near-linear in N for simple / distributional imputers"
        cost_tier = "low"

    return {
        "grid_candidate_count": n_candidates,
        "methods": methods,
        "has_knn": has_knn,
        "has_iterative": has_iterative,
        "has_tree_ensemble": has_tree_ensemble,
        "cost_tier": cost_tier,
        "complexity_note": big_o,
    }


def _literature_sample_budget(
    *,
    n_rows: int,
    missing_pct: float,
    has_knn: bool,
    grid_name: str,
) -> dict[str, Any]:
    """Map MI literature onto Phil's ensemble ``samples`` knob.

    Anchors:
    - Rubin: small m (≈3–10) often enough for point-estimate efficiency.
    - White / Bodner / von Hippel: larger m as FMI grows if you care about
      stable SEs; a common practical rule is m on the order of % missing.
    - Phil ``samples`` covers a multiverse of candidates (not Rubin's rules
      pooling), so we still raise m with missingness, then cap by scalability
      (large N, KNN).
    """
    fmi_proxy = missing_pct / 100.0
    m_stability = max(_M_EFFICIENCY_FLOOR, int(math.ceil(100 * fmi_proxy)))
    m_stability = min(m_stability, _M_STABILITY_CAP)

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
        "m_stability_uncapped": min(
            max(_M_EFFICIENCY_FLOOR, int(math.ceil(100 * fmi_proxy))),
            _M_STABILITY_CAP,
        ),
        "scalability_cap": scalability_cap,
        "suggested_samples": suggested,
        "literature_notes": [
            "Rubin MI: m≈5 often enough for point-estimate efficiency.",
            "White/Bodner/von Hippel: increase m with FMI for stable SEs; "
            "practical rule m ≈ percent missing (capped).",
            "Phil samples ≈ multiverse coverage of the candidate grid; "
            "cap further when N is large or KNN is present.",
        ],
    }


def _subsample_advice(n_rows: int, has_knn: bool) -> dict[str, Any] | None:
    if n_rows <= _LARGE_N_ROWS and not (has_knn and n_rows > 20_000):
        return None
    target = 50_000 if has_knn else 100_000
    target = min(target, n_rows)
    return {
        "recommended": n_rows > target,
        "suggested_rows": target if n_rows > target else n_rows,
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
        scale = _grid_scalability(name)
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
            "`marketing` pairs with TargetEncoder-friendly preprocessing."
        )
    elif overall_missing_pct > _EXTREME_MISSING_PCT:
        recommended = "default"
        warnings.append(
            "Overall missingness > 50%: tree-based / iterative regressors may "
            "struggle to converge; monitor diagnose_sweep spread and consider "
            "dropping ultra-sparse columns."
        )
        rationale.append(
            f"Extreme missingness ({overall_missing_pct}%); start with "
            "`default` and raise samples with FMI (literature-guided)."
        )
    else:
        recommended = "default"
        rationale.append(
            "No scale, cardinality, or extreme-missingness flags; "
            "`default` is the general-purpose starting grid."
        )

    if recommended not in GRID_METADATA:
        recommended = "default"

    scalability = _grid_scalability(recommended)
    sample_budget = _literature_sample_budget(
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
        "Sample budget from MI literature (efficiency floor m≈5; raise with "
        f"FMI proxy={sample_budget['fmi_proxy']}) then cap for scalability "
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
