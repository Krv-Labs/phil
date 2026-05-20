"""Fixtures for Phil MCP tests.

Skips the entire MCP test subtree when the optional ``mcp`` dependency
group (``fastmcp`` / ``pyyaml`` / ``pyarrow``) is not installed locally.
CI installs it via ``uv sync --group mcp``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "fastmcp",
    reason=(
        "phil MCP tests require the 'mcp' optional dependency group "
        "(`uv sync --group mcp` or `pip install philler[mcp]`)."
    ),
)
pytest.importorskip("yaml", reason="pyyaml required for MCP tests")
pytest.importorskip("pyarrow", reason="pyarrow required for MCP Parquet tests")


@pytest.fixture
def missing_df() -> pd.DataFrame:
    """Small mixed-type frame with missing values for sweep smoke tests."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, np.nan, 45.0, 22.0, np.nan, 51.0, 33.0, 41.0, 29.0],
            "income": [
                50000,
                np.nan,
                75000,
                80000,
                65000,
                72000,
                np.nan,
                58000,
                90000,
                61000,
            ],
            "category": ["A", "B", "A", None, "B", "A", "B", "B", "A", "B"],
        }
    )


@pytest.fixture
def csv_path(tmp_path, missing_df: pd.DataFrame) -> str:
    target = tmp_path / "missing.csv"
    missing_df.to_csv(target, index=False)
    return str(target)


@pytest.fixture
def parquet_path(tmp_path, missing_df: pd.DataFrame) -> str:
    target = tmp_path / "missing.parquet"
    missing_df.to_parquet(target, index=False)
    return str(target)
