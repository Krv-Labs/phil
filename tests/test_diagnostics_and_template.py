"""Tests for domain-agnostic predictability profiling and research template generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastmcp.client import Client

from phil.diagnostics import build_research_worksheet, profile_predictability
from phil.mcp.config import validate_config_yaml
from phil.mcp.server import mcp


def _make_synthetic_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50.0, 10.0, size=n)
    x2 = 2.5 * x1 + rng.normal(0.0, 2.0, size=n)
    x3 = np.exp(0.03 * x1) + rng.normal(0.0, 0.2, size=n)
    cat = np.where(x1 > 50.0, "high", "low")
    df = pd.DataFrame(
        {
            "age": x1,
            "biomarker_linear": x2,
            "biomarker_skewed": x3,
            "stratum": cat,
        }
    )
    # Introduce MAR missingness on biomarker_linear and biomarker_skewed driven by age
    miss_prob = 1.0 / (1.0 + np.exp(-(x1 - 52.0) / 5.0))
    mask_linear = rng.uniform(size=n) < (0.35 * miss_prob)
    mask_skewed = rng.uniform(size=n) < 0.25
    mask_cat = rng.uniform(size=n) < 0.15
    df.loc[mask_linear, "biomarker_linear"] = np.nan
    df.loc[mask_skewed, "biomarker_skewed"] = np.nan
    df.loc[mask_cat, "stratum"] = None
    return df


def test_profile_predictability_identifies_predictable_columns() -> None:
    df = _make_synthetic_df()
    report = profile_predictability(df, random_state=42)
    assert report["n_rows"] == 80
    assert report["n_cols"] == 4
    targets = report["target_profiles"]
    assert "biomarker_linear" in targets
    assert "biomarker_skewed" in targets
    assert "stratum" in targets

    lin_prof = targets["biomarker_linear"]
    assert lin_prof["predictability_tier"] in {"high", "moderate"}
    assert lin_prof["linear_cv_score"] > 0.5
    assert "age" in lin_prof["top_predictors"]
    assert "distribution_summary" in lin_prof
    assert "missingness_mechanism" in lin_prof


def test_build_research_worksheet_produces_valid_yaml_skeleton() -> None:
    df = _make_synthetic_df()
    worksheet = build_research_worksheet(
        df,
        study_context="Cardiometabolic cohort missingness study",
        run_name="cardio_sweep",
        data_path="/tmp/cohort.csv",
        random_state=42,
    )
    assert "research_questions" in worksheet
    assert len(worksheet["research_questions"]) >= 2
    assert "config_yaml_template" in worksheet
    assert "predictability_profile" in worksheet

    val_report = validate_config_yaml(worksheet["config_yaml_template"])
    assert val_report.ok, val_report.issues
    parsed = yaml.safe_load(worksheet["config_yaml_template"])
    dk = parsed["imputation"]["domain_knowledge"]
    assert "covariate_subsets" in dk
    assert "variable_bounds" in dk
    assert "biomarker_linear" in dk["covariate_subsets"]


@pytest.mark.asyncio
async def test_mcp_profile_predictability_and_generate_research_template(
    tmp_path: Path,
) -> None:
    df = _make_synthetic_df(n=50)
    csv_file = tmp_path / "cohort.csv"
    df.to_csv(csv_file, index=False)

    async with Client(mcp) as client:
        ingest_res = await client.call_tool("ingest_dataset", {"path": str(csv_file)})
        ingest = json.loads(ingest_res.content[0].text)
        dataset_id = ingest["dataset_id"]

        prof_res = await client.call_tool(
            "profile_predictability",
            {"dataset_id": dataset_id},
        )
        prof = json.loads(prof_res.content[0].text)
        assert prof["status"] == "ok"
        assert "biomarker_linear" in prof["target_profiles"]

        tmpl_res = await client.call_tool(
            "generate_research_template",
            {
                "dataset_id": dataset_id,
                "study_context": "Synthetic cardiometabolic trial",
            },
        )
        tmpl = json.loads(tmpl_res.content[0].text)
        assert tmpl["status"] == "ok"
        assert "config_yaml_template" in tmpl
        assert "research_questions" in tmpl

        val_res = await client.call_tool(
            "validate_config",
            {
                "config_yaml": tmpl["config_yaml_template"],
                "dataset_id": dataset_id,
            },
        )
        val = json.loads(val_res.content[0].text)
        assert val["status"] == "ok"


@pytest.mark.asyncio
async def test_phirewall_confinement_preserves_tokens_and_redacts_probe_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_synthetic_df(n=30)
    df.insert(0, "mrn", [f"tok_{i:012x}" for i in range(len(df))])
    df["rare_code"] = ["RARE_1", "RARE_2"] + ["COMMON"] * (len(df) - 2)
    csv_file = tmp_path / "twin.csv"
    df.to_csv(csv_file, index=False)

    monkeypatch.setenv("PHIL_DISABLE_UPLOAD", "1")
    async with Client(mcp) as client:
        ingest_res = await client.call_tool("ingest_dataset", {"path": str(csv_file)})
        dataset_id = json.loads(ingest_res.content[0].text)["dataset_id"]

        probe_res = await client.call_tool(
            "probe_columns",
            {"dataset_id": dataset_id, "columns": ["mrn", "rare_code"]},
        )
        probe = json.loads(probe_res.content[0].text)
        assert probe["status"] == "ok"
        by_name = {p["name"]: p for p in probe["column_profiles"]}
        assert by_name["mrn"]["sample_values"] == []
        assert by_name["mrn"]["top_values"] == {}
        assert by_name["rare_code"]["sample_values"] == []
        assert "RARE_1" not in by_name["rare_code"]["top_values"]
        assert by_name["rare_code"]["top_values"].get("COMMON") == 28

        tmpl_res = await client.call_tool(
            "generate_research_template",
            {"dataset_id": dataset_id, "study_context": "Phirewall twin test"},
        )
        tmpl = json.loads(tmpl_res.content[0].text)
        assert "mrn" in tmpl["predictability_profile"]["identifier_or_token_columns"]
        parsed_cfg = yaml.safe_load(tmpl["config_yaml_template"])
        assert "mrn" in parsed_cfg["imputation"]["drop_cols"]
