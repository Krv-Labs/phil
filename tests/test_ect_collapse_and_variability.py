"""Tests for ECT signature distribution analysis, 6 collapse strategies, Rubin variability, and confinement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastmcp.client import Client

from phil.magic.collapse import (
    COLLAPSE_STRATEGIES,
    analyze_ect_distribution,
    collapse_ect_ensemble,
    compute_weiszfeld_median,
)
from phil.mcp.registry import DatasetRegistry
from phil.mcp.server import mcp


def _synthetic_ensemble() -> tuple[pd.DataFrame, list[pd.DataFrame], list[np.ndarray]]:
    orig = pd.DataFrame(
        {
            "x": [10.0, 20.0, np.nan, 40.0, np.nan],
            "y": [1.0, 2.0, 3.0, np.nan, 5.0],
            "group": ["a", "b", None, "a", "b"],
        }
    )
    candidates: list[pd.DataFrame] = []
    descriptors: list[np.ndarray] = []

    # 4 tightly clustered candidates + 1 extreme outlier candidate
    imputations_x = [30.0, 30.5, 29.8, 30.2, 95.0]
    imputations_x2 = [50.0, 49.5, 50.2, 50.1, 150.0]
    imputations_y = [4.0, 4.1, 3.9, 4.05, 25.0]
    imputations_g = ["a", "a", "a", "b", "b"]

    ect_vecs = [
        np.array([0.10, 0.20, 0.30, 0.40]),
        np.array([0.11, 0.19, 0.31, 0.39]),
        np.array([0.09, 0.21, 0.29, 0.41]),
        np.array([0.105, 0.205, 0.305, 0.405]),
        np.array([5.0, -4.0, 6.0, -5.0]),  # outlier in ECT space
    ]

    for ix1, ix2, iy, ig, psi in zip(
        imputations_x, imputations_x2, imputations_y, imputations_g, ect_vecs
    ):
        c = orig.copy()
        c.loc[2, "x"] = ix1
        c.loc[4, "x"] = ix2
        c.loc[3, "y"] = iy
        c.loc[2, "group"] = ig
        candidates.append(c)
        descriptors.append(psi)

    return orig, candidates, descriptors


def test_weiszfeld_median_resists_outlier() -> None:
    _, _, descriptors = _synthetic_ensemble()
    V = np.stack(descriptors)
    med = compute_weiszfeld_median(V)
    mean = np.mean(V, axis=0)
    core_center = np.mean(V[:4], axis=0)
    # Geometric median should be much closer to the core 4 points than the arithmetic mean
    assert np.linalg.norm(med - core_center) < np.linalg.norm(mean - core_center)


def test_analyze_ect_distribution_detects_outlier_and_recommends_strategy() -> None:
    _, _, descriptors = _synthetic_ensemble()
    meta = [
        {"index": i, "method": "MaskedIterativeImputer" if i < 4 else "SimpleImputer"}
        for i in range(5)
    ]
    diag = analyze_ect_distribution(descriptors, candidate_metadata=meta)
    assert diag["n_candidates"] == 5
    assert diag["outliers"]["n_outliers"] >= 1
    outlier_idxs = [o["index"] for o in diag["outliers"]["outlier_candidates"]]
    assert 4 in outlier_idxs
    assert diag["recommended_strategy"] in {
        "trimmed_barycenter",
        "cluster_barycenter",
        "frechet_medoid",
    }


@pytest.mark.parametrize("strategy", list(COLLAPSE_STRATEGIES.keys()) + ["auto"])
def test_collapse_ect_ensemble_all_strategies_and_rubin_variability(
    strategy: str,
) -> None:
    orig, candidates, descriptors = _synthetic_ensemble()
    res = collapse_ect_ensemble(
        candidates=candidates,
        descriptors=descriptors,
        strategy=strategy,
        compare_all=True,
        original_df=orig,
        variable_bounds={"x": {"min_value": 0.0, "max_value": 100.0}},
    )
    assert res.strategy in COLLAPSE_STRATEGIES
    assert res.collapsed_df.shape == orig.shape
    assert res.collapsed_df.isna().sum().sum() == 0
    assert len(res.weights) == 5
    assert pytest.approx(sum(res.weights), rel=1e-4) == 1.0
    assert len(res.strategy_comparison) == len(COLLAPSE_STRATEGIES)

    # Verify Rubin's per-column variability metrics
    assert "x" in res.column_variability
    assert "y" in res.column_variability
    assert "group" in res.column_variability
    assert res.column_variability["x"]["n_imputed_cells"] == 2
    assert res.column_variability["x"]["mean_between_imputation_std"] > 0.0
    assert res.column_variability["group"]["type"] == "categorical"


def test_dataset_registry_honors_phil_datasets_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_dir = tmp_path / "confined_phil_mcp"
    monkeypatch.setenv("PHIL_DATASETS_DIR", str(pinned_dir))
    reg = DatasetRegistry()
    assert reg.cache_dir == pinned_dir
    assert pinned_dir.exists()


@pytest.mark.asyncio
async def test_upload_dataset_blocked_when_phil_disable_upload_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHIL_DISABLE_UPLOAD", "1")
    async with Client(mcp) as client:
        res = await client.call_tool(
            "begin_dataset_upload",
            {"filename": "test.csv"},
        )
        payload = json.loads(res.content[0].text)
        assert payload["status"] == "error"
        assert payload["error_code"] == "UPLOAD_DISABLED"


@pytest.mark.asyncio
async def test_mcp_ect_navigation_and_collapse_tools(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0, np.nan, 70.0, 80.0],
            "c": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    csv_file = tmp_path / "ect_nav.csv"
    df.to_csv(csv_file, index=False)

    async with Client(mcp) as client:
        # 1. list_collapse_strategies
        strat_res = await client.call_tool("list_collapse_strategies", {})
        strat_payload = json.loads(strat_res.content[0].text)
        assert strat_payload["status"] == "ok"
        assert set(strat_payload["strategies"].keys()) == set(
            COLLAPSE_STRATEGIES.keys()
        )

        # 2. ingest + create_config + run_imputation_sweep
        ing = json.loads(
            (await client.call_tool("ingest_dataset", {"path": str(csv_file)}))
            .content[0]
            .text
        )
        cfg = json.loads(
            (
                await client.call_tool(
                    "create_config",
                    {"dataset_id": ing["dataset_id"], "grid": "default", "samples": 4},
                )
            )
            .content[0]
            .text
        )
        sweep = json.loads(
            (
                await client.call_tool(
                    "run_imputation_sweep",
                    {
                        "config_yaml": cfg["config_yaml"],
                        "dataset_id": ing["dataset_id"],
                    },
                )
            )
            .content[0]
            .text
        )
        assert sweep["status"] == "ok", sweep
        assert sweep["imputed_cache_path"]
        assert Path(sweep["imputed_cache_path"]).exists()

        # 3. inspect_ect_distribution
        insp = json.loads(
            (await client.call_tool("inspect_ect_distribution", {})).content[0].text
        )
        assert insp["status"] == "ok"
        assert "recommended_strategy" in insp
        assert "center_and_dispersion" in insp

        # 4. collapse_ensemble with compare_all=True
        col_res = json.loads(
            (
                await client.call_tool(
                    "collapse_ensemble",
                    {"strategy": "trimmed_barycenter", "compare_all": True},
                )
            )
            .content[0]
            .text
        )
        assert col_res["status"] == "ok"
        report = col_res["collapse_report"]
        assert report["strategy"] == "trimmed_barycenter"
        assert len(report["strategy_comparison"]) == len(COLLAPSE_STRATEGIES)
        assert "a" in report["column_variability"]


@pytest.mark.asyncio
async def test_sweep_and_collapse_preserve_dropped_identifier_token_columns(
    tmp_path: Path,
) -> None:
    tokens = [f"tok_{i:012x}" for i in range(8)]
    df = pd.DataFrame(
        {
            "mrn": tokens,
            "a": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0, np.nan, 70.0, 80.0],
            "c": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    csv_file = tmp_path / "twin_with_tokens.csv"
    df.to_csv(csv_file, index=False)

    async with Client(mcp) as client:
        ing = json.loads(
            (await client.call_tool("ingest_dataset", {"path": str(csv_file)}))
            .content[0]
            .text
        )
        cfg = json.loads(
            (
                await client.call_tool(
                    "create_config",
                    {"dataset_id": ing["dataset_id"], "grid": "default", "samples": 4},
                )
            )
            .content[0]
            .text
        )
        refined = json.loads(
            (
                await client.call_tool(
                    "refine_config",
                    {
                        "config_yaml": cfg["config_yaml"],
                        "overrides": {"imputation.drop_cols": ["mrn"]},
                    },
                )
            )
            .content[0]
            .text
        )
        sweep = json.loads(
            (
                await client.call_tool(
                    "run_imputation_sweep",
                    {
                        "config_yaml": refined["config_yaml"],
                        "dataset_id": ing["dataset_id"],
                    },
                )
            )
            .content[0]
            .text
        )
        assert sweep["status"] == "ok", sweep
        cached_df = pd.read_csv(sweep["imputed_cache_path"])
        assert list(cached_df.columns) == ["mrn", "a", "b", "c"]
        assert list(cached_df["mrn"]) == tokens
        assert cached_df.isna().sum().sum() == 0

        col_res = json.loads(
            (
                await client.call_tool(
                    "collapse_ensemble",
                    {"strategy": "frechet_medoid"},
                )
            )
            .content[0]
            .text
        )
        assert col_res["status"] == "ok"
        collapsed_df = pd.read_csv(col_res["imputed_cache_path"])
        assert list(collapsed_df.columns) == ["mrn", "a", "b", "c"]
        assert list(collapsed_df["mrn"]) == tokens
