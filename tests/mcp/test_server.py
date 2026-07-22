"""End-to-end smoke tests for the Phil MCP server.

The Phil pipeline is patched to skip the ``trailed`` ECT backend so these
tests run on any developer machine.

Skipped automatically when the optional ``mcp`` dependency group
(``fastmcp``) is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

from fastmcp import Client  # noqa: E402

from phil.mcp.server import mcp  # noqa: E402


def _payload(call_result) -> dict[str, Any]:
    """Extract a JSON dict payload from a FastMCP tool call result."""
    content = call_result.content
    text = content[0].text if content else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


@pytest.fixture
def patched_phil(monkeypatch):
    """Patch ``Phil.fit`` so sweeps run without the ``trailed`` backend."""

    from phil.mcp import server as server_module

    def fake_fit(self, df, max_iter: int = 5):
        n_cols = df.shape[1]
        columns = [f"feat_{i}" for i in range(n_cols)]
        numeric = df.select_dtypes(include="number").fillna(0)
        imputed = df.copy()
        imputed[numeric.columns] = numeric
        imputed = imputed.fillna("missing")
        imputed.columns = columns
        self.representations = [
            numeric.to_numpy(dtype=float)
            if not numeric.empty
            else np.zeros((len(df), 1))
        ]
        self.magic_descriptors = [
            np.array([0.1, 0.2]),
            np.array([0.15, 0.25]),
            np.array([0.12, 0.22]),
        ]
        self.closest_index = 1
        from unittest.mock import MagicMock

        pipe = MagicMock()
        pipe.__getitem__.return_value = MagicMock()
        self.selected_imputers = [pipe, pipe, pipe]
        self.pipeline = pipe
        return imputed

    monkeypatch.setattr(server_module.Phil, "fit", fake_fit)
    yield


@pytest.mark.asyncio
async def test_list_grids_returns_builtins() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("list_grids", {})
        payload = _payload(result)
        assert payload["status"] == "ok"
        names = {g["name"] for g in payload["grids"]}
        assert "default" in names
        assert "healthcare" in names
        healthcare = next(g for g in payload["grids"] if g["name"] == "healthcare")
        assert healthcare["time_complexity"] == "High"
        assert healthcare["suitability"]
        assert "scale_limits" in healthcare


@pytest.mark.asyncio
async def test_recommend_grid_after_ingest(csv_path: str) -> None:
    async with Client(mcp) as client:
        ingest = _payload(await client.call_tool("ingest_dataset", {"path": csv_path}))
        dataset_id = ingest["dataset_id"]
        result = _payload(
            await client.call_tool("recommend_grid", {"dataset_id": dataset_id})
        )
        assert result["status"] == "ok"
        assert result["recommended_grid"] in {
            "default",
            "sampling",
            "finance",
            "healthcare",
            "marketing",
            "engineering",
        }
        assert "suggested_samples" in result
        assert "Recommended Grid:" in result["recommendation"]
        assert result["grid"]["name"] == result["recommended_grid"]


@pytest.mark.asyncio
async def test_imputation_matrix_resource() -> None:
    async with Client(mcp) as client:
        resources = await client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "phil://docs/imputation-matrix" in uris
        contents = await client.read_resource("phil://docs/imputation-matrix")
        text = "".join(
            getattr(part, "text", "") or "" for part in contents
        )
        assert "Phil Imputation Grid Matrix" in text
        assert "`healthcare`" in text

@pytest.mark.asyncio
async def test_end_to_end_csv_sweep(
    csv_path: str, tmp_path: Path, patched_phil
) -> None:
    async with Client(mcp) as client:
        ingest = _payload(await client.call_tool("ingest_dataset", {"path": csv_path}))
        dataset_id = ingest["dataset_id"]
        assert dataset_id.startswith("ds_")

        char = _payload(
            await client.call_tool("characterize_dataset", {"dataset_id": dataset_id})
        )
        assert char["status"] == "ok"
        assert char["n_rows"] == 10
        assert char["total_missing"] > 0

        cfg = _payload(
            await client.call_tool(
                "create_config",
                {"dataset_id": dataset_id, "grid": "default", "samples": 3},
            )
        )
        assert cfg["status"] == "ok"
        assert "config_yaml" in cfg

        validation = _payload(
            await client.call_tool(
                "validate_config",
                {"config_yaml": cfg["config_yaml"], "dataset_id": dataset_id},
            )
        )
        assert validation["status"] == "ok"

        run = _payload(
            await client.call_tool(
                "run_imputation_sweep",
                {"config_yaml": cfg["config_yaml"], "dataset_id": dataset_id},
            )
        )
        assert run["status"] == "ok", run.get("reason", run)
        assert run["n_candidates"] == 3
        assert run["selected_index"] == 1
        run_id = run["run_id"]
        assert run_id.startswith("run_")

        summary = _payload(
            await client.call_tool("get_sweep_summary", {"run_id": run_id})
        )
        assert summary["run_id"] == run_id

        diag = _payload(await client.call_tool("diagnose_sweep", {"run_id": run_id}))
        assert diag["descriptor_stats"]["n_candidates"] == 3

        ranked = _payload(
            await client.call_tool(
                "get_candidate_descriptors", {"run_id": run_id, "top_k": 2}
            )
        )
        assert ranked["status"] == "ok"
        assert len(ranked["ranking"]) == 2

        out_csv = tmp_path / "imputed.csv"
        export = _payload(
            await client.call_tool(
                "export_imputed_data",
                {"output_path": str(out_csv)},
            )
        )
        assert export["status"] == "ok"
        assert out_csv.exists()


@pytest.mark.asyncio
async def test_run_rejects_no_missing(tmp_path: Path, patched_phil) -> None:
    full_csv = tmp_path / "complete.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(full_csv, index=False)

    async with Client(mcp) as client:
        ingest = _payload(
            await client.call_tool("ingest_dataset", {"path": str(full_csv)})
        )
        cfg = _payload(
            await client.call_tool(
                "create_config",
                {"dataset_id": ingest["dataset_id"], "samples": 2},
            )
        )
        run = _payload(
            await client.call_tool(
                "run_imputation_sweep",
                {"config_yaml": cfg["config_yaml"], "dataset_id": ingest["dataset_id"]},
            )
        )
        assert run["status"] == "error"
        assert run["error_code"] == "NO_MISSING_VALUES"


@pytest.mark.asyncio
async def test_parquet_path_ingested_natively(parquet_path: str, patched_phil) -> None:
    async with Client(mcp) as client:
        ingest = _payload(
            await client.call_tool("ingest_dataset", {"path": parquet_path})
        )
        char = _payload(
            await client.call_tool(
                "characterize_dataset", {"dataset_id": ingest["dataset_id"]}
            )
        )
        assert char["status"] == "ok"
        assert char["n_rows"] == 10


@pytest.mark.asyncio
async def test_workflow_guide_and_runtime_context() -> None:
    async with Client(mcp) as client:
        guide = await client.call_tool("get_workflow_guide", {})
        assert "Phil Imputation Sweep Workflow" in guide.content[0].text

        ctx_text = await client.call_tool("get_runtime_context", {})
        payload = json.loads(ctx_text.content[0].text)
        assert "cache_dir" in payload
        assert payload["transport_assumption"] == "stdio-single-client"


@pytest.mark.asyncio
async def test_refine_active_config_round_trip(csv_path: str) -> None:
    async with Client(mcp) as client:
        ingest = _payload(await client.call_tool("ingest_dataset", {"path": csv_path}))
        await client.call_tool(
            "create_config",
            {"dataset_id": ingest["dataset_id"], "samples": 3},
        )
        refined = _payload(
            await client.call_tool(
                "refine_active_config",
                {"overrides": {"imputation.samples": 7}},
            )
        )
        assert refined["status"] == "ok"
        assert any(d["path"] == "imputation.samples" for d in refined["diff"])
        active = _payload(await client.call_tool("get_active_config", {}))
        assert "samples: 7" in active["config_yaml"]


@pytest.mark.asyncio
async def test_run_with_drop_controls(csv_path: str, patched_phil) -> None:
    async with Client(mcp) as client:
        ingest = _payload(await client.call_tool("ingest_dataset", {"path": csv_path}))
        cfg = _payload(
            await client.call_tool(
                "create_config",
                {"dataset_id": ingest["dataset_id"], "samples": 3},
            )
        )
        refined = _payload(
            await client.call_tool(
                "refine_config",
                {
                    "config_yaml": cfg["config_yaml"],
                    "overrides": {
                        "imputation.drop_cols": ["income"],
                        "imputation.missingness_thresh": 0.1,
                    },
                },
            )
        )
        run = _payload(
            await client.call_tool(
                "run_imputation_sweep",
                {
                    "config_yaml": refined["config_yaml"],
                    "dataset_id": ingest["dataset_id"],
                },
            )
        )
        assert run["status"] == "ok", run
        assert "income" in run["dropped_columns"]


@pytest.mark.asyncio
async def test_run_rejects_invalid_drop_cols(csv_path: str, patched_phil) -> None:
    config_yaml = f"""run:
  name: bad_drop
  data: {csv_path}
imputation:
  grid: default
  samples: 3
  drop_cols: [missing_col]
"""
    async with Client(mcp) as client:
        run = _payload(
            await client.call_tool(
                "run_imputation_sweep",
                {"config_yaml": config_yaml},
            )
        )
        assert run["status"] == "error"
        assert run["error_code"] == "INVALID_DROP_COLS"


@pytest.mark.asyncio
async def test_run_rejects_string_columns_when_encoding_disabled(
    csv_path: str, patched_phil
) -> None:
    config_yaml = f"""run:
  name: no_encoding
  data: {csv_path}
imputation:
  grid: default
  samples: 3
  encode_categoricals: false
"""
    async with Client(mcp) as client:
        run = _payload(
            await client.call_tool(
                "run_imputation_sweep",
                {"config_yaml": config_yaml},
            )
        )
        assert run["status"] == "error"
        assert run["error_code"] == "UNSUPPORTED_STRING_COLUMNS"

