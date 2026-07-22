"""Unit tests for declarative grid metadata and the rule-based recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phil.gallery import GRID_METADATA, render_imputation_matrix
from phil.mcp.recommend import recommend_grid_for_dataframe


def test_render_imputation_matrix_includes_all_grids() -> None:
    md = render_imputation_matrix()
    assert md.startswith("# Phil Imputation Grid Matrix")
    assert "| Grid | Domain | Complexity |" in md
    for name in GRID_METADATA:
        assert f"`{name}`" in md


def test_recommend_default_for_small_mixed_frame() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": ["x", "y", None, "x"],
        }
    )
    result = recommend_grid_for_dataframe(df)
    assert result["status"] == "ok"
    assert result["recommended_grid"] == "default"
    # 2/8 cells missing → FMI proxy 0.25 → m≈25, no KNN cap 30
    assert result["suggested_samples"] == 25
    assert result["scalability"]["grid_candidate_count"] == 17
    assert result["sample_budget"]["m_efficiency_floor"] == 5
    assert "Recommended Grid: default" in result["recommendation"]
    assert result["grid"]["name"] == "default"
    assert result["metrics"]["n_rows"] == 4


def test_recommend_sampling_for_large_continuous_frame() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=100_001),
            "y": rng.normal(size=100_001),
        }
    )
    df.loc[:10, "x"] = np.nan
    result = recommend_grid_for_dataframe(df)
    assert result["recommended_grid"] == "sampling"
    # Tiny missingness + large N → efficiency floor, scalability/sampling cap
    assert result["suggested_samples"] == 5
    assert result["scalability"]["has_knn"] is False
    assert result["subsample"] is not None
    assert any("100,000" in w or "KNN" in w for w in result["warnings"])


def test_recommend_default_for_large_categorical_frame() -> None:
    rng = np.random.default_rng(1)
    n = 100_001
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    result = recommend_grid_for_dataframe(df)
    assert result["recommended_grid"] == "default"
    assert result["suggested_samples"] == 5
    assert result["warnings"]
    assert result["scalable_alternatives"]


def test_recommend_marketing_for_high_cardinality() -> None:
    df = pd.DataFrame(
        {
            "zip": [f"z{i}" for i in range(60)],
            "spend": list(range(60)),
        }
    )
    df.loc[0, "spend"] = np.nan
    result = recommend_grid_for_dataframe(df)
    assert result["recommended_grid"] == "marketing"
    # ~0.8% missing → m≈5; marketing has KNN so cap 15 → 5
    assert result["suggested_samples"] == 5
    assert result["scalability"]["has_knn"] is True
    assert "zip" in result["metrics"]["high_cardinality_columns"]
    assert any(a["grid"] == "default" for a in result["scalable_alternatives"])


def test_recommend_warns_on_extreme_missingness() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan, np.nan],
            "b": [np.nan, np.nan, np.nan, 2.0],
        }
    )
    result = recommend_grid_for_dataframe(df)
    assert result["recommended_grid"] == "default"
    assert result["metrics"]["overall_missing_pct"] > 50
    # 75% missing → m_stability 40, no-KNN cap 30 → 30
    assert result["suggested_samples"] == 30
    assert any("50%" in w for w in result["warnings"])


def test_recommend_marketing_for_integer_high_cardinality_ids() -> None:
    """ZIP-style integer IDs should still trigger the marketing grid."""
    df = pd.DataFrame(
        {
            "zip_code": list(range(10000, 10080)),
            "spend": list(range(80)),
        }
    )
    df.loc[0, "spend"] = np.nan
    result = recommend_grid_for_dataframe(df)
    assert result["recommended_grid"] == "marketing"
    assert "zip_code" in result["metrics"]["high_cardinality_columns"]
    assert result["sample_budget"]["literature_notes"]


def test_recommend_marketing_for_nullable_integer_ids() -> None:
    """Nullable Int64 IDs with NA still count as high-cardinality categoricals."""
    df = pd.DataFrame(
        {
            "zip_code": pd.array(list(range(10000, 10080)) + [pd.NA], dtype="Int64"),
            "spend": list(range(81)),
        }
    )
    df.loc[1, "spend"] = np.nan
    result = recommend_grid_for_dataframe(df)
    assert result["recommended_grid"] == "marketing"
    assert "zip_code" in result["metrics"]["high_cardinality_columns"]


def test_recommend_default_for_integral_float_lab_values() -> None:
    """Whole-number lab floats (glucose/insulin-like) are not marketing IDs."""
    df = pd.DataFrame(
        {
            "glucose": [float(i) for i in range(80, 200)],
            "bmi": list(range(120)),
        }
    )
    df.loc[0, "glucose"] = np.nan
    df.loc[1, "bmi"] = np.nan
    result = recommend_grid_for_dataframe(df)
    assert result["recommended_grid"] == "default"
    assert "glucose" not in result["metrics"]["high_cardinality_columns"]
