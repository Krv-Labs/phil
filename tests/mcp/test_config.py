"""Tests for Phil MCP config schema validation and overrides."""

from __future__ import annotations

import pytest
import yaml

from phil.imputation import ImputationConfig
from phil.magic import ECTConfig
from phil.mcp.config import (
    apply_overrides,
    config_to_phil_kwargs,
    list_builtin_grids,
    render_default_config_yaml,
    validate_config_yaml,
)


def test_default_config_validates_clean() -> None:
    config_yaml = render_default_config_yaml(data="/tmp/data.csv")
    report = validate_config_yaml(config_yaml)
    assert report.ok, report.issues
    assert report.normalized_yaml is not None
    parsed = yaml.safe_load(report.normalized_yaml)
    assert parsed["imputation"]["grid"] == "default"
    assert parsed["magic"]["method"] == "ECT"


def test_unknown_top_level_section_rejected() -> None:
    config_yaml = "run:\n  name: t\nbogus:\n  x: 1\n"
    report = validate_config_yaml(config_yaml)
    assert not report.ok
    assert any(issue.path.startswith("$.bogus") for issue in report.issues)


def test_fenced_markdown_yaml_rejected() -> None:
    config_yaml = "```yaml\nrun:\n  name: t\n```\n"
    report = validate_config_yaml(config_yaml)
    assert not report.ok
    assert report.error_code == "YAML_NOT_RAW"


def test_unknown_grid_rejected() -> None:
    config_yaml = "imputation:\n  grid: bogus\n"
    report = validate_config_yaml(config_yaml)
    assert not report.ok
    assert any("grid" in issue.path for issue in report.issues)


def test_drop_controls_validate_clean() -> None:
    config_yaml = """imputation:
  grid: default
  drop_cols: [a, b]
  missingness_thresh: 0.5
  encode_categoricals: true
"""
    report = validate_config_yaml(config_yaml)
    assert report.ok, report.issues
    parsed = yaml.safe_load(report.normalized_yaml)
    assert parsed["imputation"]["drop_cols"] == ["a", "b"]
    assert parsed["imputation"]["missingness_thresh"] == 0.5
    assert parsed["imputation"]["encode_categoricals"] is True


def test_drop_controls_reject_invalid_threshold() -> None:
    config_yaml = "imputation:\n  missingness_thresh: 2\n"
    report = validate_config_yaml(config_yaml)
    assert not report.ok
    assert any(issue.path == "$.imputation.missingness_thresh" for issue in report.issues)


def test_drop_controls_reject_non_string_drop_cols() -> None:
    config_yaml = "imputation:\n  drop_cols: [1, a]\n"
    report = validate_config_yaml(config_yaml)
    assert not report.ok
    assert any(issue.path == "$.imputation.drop_cols" for issue in report.issues)


def test_apply_overrides_modifies_values() -> None:
    config_yaml = render_default_config_yaml(data="/tmp/data.csv")
    result = apply_overrides(
        config_yaml,
        {
            "imputation.samples": 5,
            "magic.num_thetas": 16,
            "imputation.missingness_thresh": 0.25,
            "imputation.drop_cols": ["x"],
        },
    )
    parsed = yaml.safe_load(result.config_yaml)
    assert parsed["imputation"]["samples"] == 5
    assert parsed["magic"]["num_thetas"] == 16
    assert parsed["imputation"]["missingness_thresh"] == 0.25
    assert parsed["imputation"]["drop_cols"] == ["x"]
    assert {entry["path"] for entry in result.diff} == {
        "imputation.samples",
        "magic.num_thetas",
        "imputation.missingness_thresh",
        "imputation.drop_cols",
    }


def test_apply_overrides_rejects_unknown_key() -> None:
    config_yaml = render_default_config_yaml()
    with pytest.raises(ValueError, match="Unknown override key"):
        apply_overrides(config_yaml, {"imputation.bogus": 1})


def test_config_to_phil_kwargs_builtin_grid() -> None:
    config_yaml = render_default_config_yaml(data="/tmp/data.csv")
    kwargs = config_to_phil_kwargs(config_yaml)
    assert kwargs["param_grid"] == "default"
    assert isinstance(kwargs["config"], ECTConfig)
    assert kwargs["samples"] == 30
    assert kwargs["max_iter"] == 5
    assert kwargs["drop_cols"] == []
    assert kwargs["missingness_thresh"] is None
    assert kwargs["encode_categoricals"] is True


def test_config_to_phil_kwargs_custom_grid() -> None:
    config_yaml = """run:
  name: t
imputation:
  grid: custom
  custom:
    methods: [KNNImputer]
    modules: [sklearn.impute]
    grids:
      - {n_neighbors: [3, 5]}
"""
    kwargs = config_to_phil_kwargs(config_yaml)
    assert isinstance(kwargs["param_grid"], ImputationConfig)
    assert kwargs["param_grid"].methods == ["KNNImputer"]
    assert list(kwargs["param_grid"].grids[0])  # ParameterGrid materializes


def test_list_builtin_grids_includes_default() -> None:
    grids = list_builtin_grids()
    names = {g["name"] for g in grids}
    assert "default" in names
    assert "healthcare" in names
    default_entry = next(g for g in grids if g["name"] == "default")
    assert default_entry["methods"]
