"""Tests for the persistent MCP registry."""

from __future__ import annotations

from pathlib import Path

from phil.mcp.registry import MCPRegistry


def test_register_dataset_round_trip(tmp_path: Path) -> None:
    csv = tmp_path / "sample.csv"
    csv.write_text("a,b\n1,2\n3,4\n")

    reg = MCPRegistry()
    record = reg.register_dataset(str(csv))

    assert record.dataset_id.startswith("ds_")
    fetched = reg.get_dataset(record.dataset_id)
    assert fetched is not None
    assert fetched.name == "sample.csv"
    assert fetched.size_bytes == csv.stat().st_size


def test_chunked_upload_finalizes_to_dataset() -> None:
    reg = MCPRegistry()
    upload = reg.begin_upload("uploaded.csv")
    payload = b"a,b\n1,2\n"
    reg.append_upload_chunk(upload.upload_id, payload)
    dataset = reg.finalize_upload(upload.upload_id)
    assert dataset is not None
    assert dataset.source == "upload"
    assert Path(dataset.path).exists()
    assert Path(dataset.path).read_bytes() == payload


def test_save_and_get_run() -> None:
    reg = MCPRegistry()
    record = reg.save_run(
        dataset_id="ds_test",
        config_yaml="run:\n  name: t\n",
        selected_index=3,
        n_candidates=10,
        descriptor_stats={"mean_pairwise_l2": 0.5},
        imputed_columns=["a", "b"],
        output_path=None,
        summary={"grid": "default"},
    )
    fetched = reg.get_run(record.run_id)
    assert fetched is not None
    assert fetched.selected_index == 3
    assert fetched.n_candidates == 10
    assert fetched.summary["grid"] == "default"
