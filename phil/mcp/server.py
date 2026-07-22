"""
FastMCP server for Phil.

Exposes the imputation-sweep pipeline as a set of MCP tools so AI agents
can ingest a pandas/polars dataset, configure a Phil sweep, run it, and
export the chosen imputation without writing Python code.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import gc
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from phil.gallery import render_imputation_matrix
from phil.mcp.config import (
    GRID_INTENTS,
    apply_overrides,
    config_to_phil_kwargs,
    default_config_dict,
    list_builtin_grids,
    render_validation_report,
    validate_config_yaml,
)
from phil.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from phil.mcp.prompts import WORKFLOW_PROMPT
from phil.mcp.recommend import recommend_grid_for_dataframe
from phil.mcp.registry import MCPRegistry
from phil.phil import Phil


logger = logging.getLogger(__name__)
registry = MCPRegistry()


# ---------------------------------------------------------------------------
# Defensive patch: strip unknown kwargs from non-compliant MCP clients.
# Some clients (e.g. Gemini CLI) inject orchestration fields like
# ``wait_for_previous`` into tool calls. FastMCP's Pydantic validation
# rejects these. Patching FunctionTool.run at the class level filters
# known orchestration keys *before* validation — one patch protects every
# tool. Unknown keys outside the allowlist are logged at WARNING.
# ---------------------------------------------------------------------------
from fastmcp.tools.function_tool import FunctionTool  # noqa: E402

_original_function_tool_run = FunctionTool.run
_KNOWN_ORCHESTRATION_KEYS = frozenset({"wait_for_previous"})


async def _lenient_function_tool_run(self, arguments):
    if isinstance(arguments, dict) and arguments:
        valid_keys = set(self.parameters.get("properties", {}).keys())
        unknown_keys = set(arguments.keys()) - valid_keys
        if unknown_keys:
            unexpected = unknown_keys - _KNOWN_ORCHESTRATION_KEYS
            if unexpected:
                logger.warning(
                    "Stripped unexpected argument(s) %s from tool %s",
                    sorted(unexpected),
                    getattr(self, "name", "unknown"),
                )
            arguments = {k: v for k, v in arguments.items() if k in valid_keys}
    return await _original_function_tool_run(self, arguments)


FunctionTool.run = _lenient_function_tool_run


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Phil",
    instructions=(
        "Representation-guided imputation for tabular datasets. Call "
        "`get_workflow_guide` for the recommended end-to-end phase-by-phase "
        "procedure.\n\n"
        "Tool map:\n"
        "- Ingest: `ingest_dataset` (local path) or the chunked "
        "`begin_dataset_upload` / `append_dataset_chunk` / "
        "`finalize_dataset_upload` trio for sandboxed clients. Returns a "
        "`dataset_id` handle. Polars users: write to Parquet, then ingest.\n"
        "- Characterize: `characterize_dataset`, `probe_columns`.\n"
        "- Configure: `recommend_grid`, `list_grids`, `create_config`, "
        "`validate_config`, `refine_config`, `get_active_config`, "
        "`refine_active_config`. Resource: `phil://docs/imputation-matrix`.\n"
        "- Run: `run_imputation_sweep`.\n"
        "- Diagnose: `diagnose_sweep`, `get_candidate_descriptors`, "
        "`compare_sweeps`, `get_experiment_history`.\n"
        "- Export: `get_sweep_summary`, `export_imputed_data`."
    ),
)

@mcp.resource(
    "phil://docs/imputation-matrix",
    mime_type="text/markdown",
    description=(
        "Markdown comparison matrix of built-in imputation grids: domain, "
        "complexity, affinity, scale limits, and estimator cost notes. "
        "Compiled from declarative GRID_METADATA."
    ),
)
def imputation_matrix_resource() -> str:
    """Return the dynamically compiled imputation-grid comparison matrix."""
    return render_imputation_matrix()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
@dataclass
class SweepRecord:
    timestamp: float
    config_yaml: str
    run_id: str
    selected_index: int
    n_candidates: int
    descriptor_stats: dict[str, Any]


@dataclass
class _PhilSession:
    """Per-MCP-client session state."""

    phil: Phil | None = None
    data: pd.DataFrame | None = None
    imputed: pd.DataFrame | None = None
    descriptors: list[np.ndarray] = field(default_factory=list)
    sweep_history: list[SweepRecord] = field(default_factory=list)
    dataset_id: str | None = None
    latest_run_id: str | None = None
    data_dataset_id: str | None = None
    data_path: str | None = None
    active_config_yaml: str | None = None
    active_config_dataset_id: str | None = None
    categorical_codebooks: dict[str, list[str]] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)

    def calculate_memory_mb(self) -> float:
        bytes_total = 0
        if self.data is not None:
            bytes_total += int(self.data.memory_usage(deep=True).sum())
        if self.imputed is not None:
            bytes_total += int(self.imputed.memory_usage(deep=True).sum())
        for arr in self.descriptors:
            if hasattr(arr, "nbytes"):
                bytes_total += int(arr.nbytes)
        return round(bytes_total / (1024 * 1024), 2)


_sessions: OrderedDict[str, _PhilSession] = OrderedDict()
_MAX_SESSIONS = int(os.environ.get("PHIL_MAX_SESSIONS", "3"))


def _session_key(ctx: Context | None) -> str:
    if ctx is None:
        return "default"
    return ctx.session_id or "default"


def _get_session(ctx: Context | None) -> _PhilSession:
    key = _session_key(ctx)
    if key in _sessions:
        _sessions.move_to_end(key)
        return _sessions[key]
    if len(_sessions) >= _MAX_SESSIONS:
        evicted_key, evicted_session = _sessions.popitem(last=False)
        logger.info(
            "Evicting oldest session '%s' to free memory (max=%d)",
            evicted_key,
            _MAX_SESSIONS,
        )
        del evicted_session
        gc.collect()
    session = _PhilSession()
    _sessions[key] = session
    return session


# ---------------------------------------------------------------------------
# Dataset resolution helpers
# ---------------------------------------------------------------------------
def _resolve_dataset_record(dataset_id: str):
    record = registry.get_dataset(dataset_id)
    if record is None:
        raise LookupError(dataset_id)
    return record


def _resolve_dataset_path(dataset_id: str) -> str:
    return _resolve_dataset_record(dataset_id).path


def _normalize_data_path(path: str) -> str:
    expanded = Path(path).expanduser()
    return str(expanded.resolve(strict=False))


def _read_dataset_file(path: str) -> pd.DataFrame:
    normalized_path = _normalize_data_path(path)
    lowered = normalized_path.lower()
    if lowered.endswith(".parquet") or lowered.endswith(".pq"):
        return pd.read_parquet(normalized_path)
    if lowered.endswith(".feather") or lowered.endswith(".arrow"):
        return pd.read_feather(normalized_path)
    return pd.read_csv(normalized_path)


def _write_csv(df: pd.DataFrame, target: str) -> None:
    df.to_csv(target, index=False)


def _write_parquet(df: pd.DataFrame, target: str) -> None:
    df.to_parquet(target, index=False)


def _write_feather(df: pd.DataFrame, target: str) -> None:
    df.reset_index(drop=True).to_feather(target)


def _bind_session_data(
    session: _PhilSession,
    df: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    data_path: str | None = None,
) -> None:
    session.data = df
    session.data_dataset_id = dataset_id
    session.data_path = _normalize_data_path(data_path) if data_path else None
    if dataset_id:
        session.dataset_id = dataset_id


def _dataset_id_for_path(dataset_id: str | None, data_path: str | None) -> str | None:
    if not dataset_id or not data_path:
        return None
    try:
        dataset_path = _normalize_data_path(_resolve_dataset_path(dataset_id))
    except LookupError:
        return None
    normalized_path = _normalize_data_path(data_path)
    return dataset_id if dataset_path == normalized_path else None


async def _load_session_dataframe(
    session: _PhilSession,
    *,
    dataset_id: str = "",
    data_path: str = "",
) -> tuple[pd.DataFrame, str]:
    if dataset_id:
        resolved_path = _resolve_dataset_path(dataset_id)
        normalized_path = _normalize_data_path(resolved_path)
        if (
            session.data is not None
            and session.data_dataset_id == dataset_id
            and session.data_path == normalized_path
        ):
            session.dataset_id = dataset_id
            return session.data, normalized_path

        df = await asyncio.to_thread(_read_dataset_file, normalized_path)
        _bind_session_data(
            session,
            df,
            dataset_id=dataset_id,
            data_path=normalized_path,
        )
        return df, normalized_path

    if not data_path:
        raise ToolError("Provide either data_path or dataset_id.")

    normalized_path = _normalize_data_path(data_path)
    if (
        session.data is not None
        and session.data_dataset_id is None
        and session.data_path == normalized_path
    ):
        return session.data, normalized_path

    df = await asyncio.to_thread(_read_dataset_file, normalized_path)
    _bind_session_data(session, df, data_path=normalized_path)
    return df, normalized_path


def _validate_config_path(path: str) -> None:
    if not Path(path).expanduser().exists():
        raise FileNotFoundError(path)


# ---------------------------------------------------------------------------
# Descriptor / sweep helpers
# ---------------------------------------------------------------------------
def _descriptor_stats(descriptors: list[np.ndarray]) -> dict[str, Any]:
    if not descriptors:
        return {
            "n_candidates": 0,
            "mean_pairwise_l2": 0.0,
            "selected_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
        }
    stacked = np.stack(descriptors)
    avg = stacked.mean(axis=0)
    diffs = (stacked - avg).reshape(len(descriptors), -1)
    norms = np.linalg.norm(diffs, axis=1)
    pairwise_means: list[float] = []
    n = len(descriptors)
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_means.append(
                float(np.linalg.norm((descriptors[i] - descriptors[j]).reshape(-1)))
            )
    return {
        "n_candidates": n,
        "mean_pairwise_l2": float(np.mean(pairwise_means)) if pairwise_means else 0.0,
        "selected_distance": float(np.min(norms)),
        "min_distance": float(np.min(norms)),
        "max_distance": float(np.max(norms)),
        "mean_distance_to_centroid": float(np.mean(norms)),
        "std_distance_to_centroid": float(np.std(norms)),
    }


def _method_counts(phil_obj: Phil) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not hasattr(phil_obj, "selected_imputers"):
        return counts
    for pipe in phil_obj.selected_imputers:
        try:
            est = pipe["imputer"].estimator
            label = type(est).__name__ if est is not None else "DefaultIterative"
        except (KeyError, AttributeError):
            label = "Unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts


def _format_grid_field(cfg: dict[str, Any]) -> str:
    imputation = cfg.get("imputation", {}) if isinstance(cfg, dict) else {}
    grid = imputation.get("grid", "default")
    return str(grid)


# ---------------------------------------------------------------------------
# Meta tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_workflow_guide(ctx: Context = None) -> str:
    """
    Return the recommended end-to-end Phil analysis workflow as markdown.

    Agents that want an opinionated phase-by-phase procedure (ingest →
    characterize → configure → run → diagnose → export) should call this
    once at the start of a session. Agents that drive Phil with their own
    workflow can ignore it.
    """
    return WORKFLOW_PROMPT


@mcp.tool()
async def get_runtime_context(ctx: Context = None) -> str:
    """
    Return the MCP server runtime context so agents can reason about path
    visibility and handle lifecycle before attempting file-based operations.
    """
    session = _get_session(ctx)
    payload = {
        "cwd": os.getcwd(),
        "cache_dir": str(registry.cache_dir),
        "temp_dir": os.getenv("TMPDIR", "/tmp"),
        "transport_assumption": "stdio-single-client",
        "session_id": _session_key(ctx),
        "dataset_handle_persistence": "on-disk registry under cache_dir",
        "run_handle_persistence": "on-disk registry under cache_dir/runs",
        "latest_dataset_id": session.dataset_id,
        "latest_run_id": session.latest_run_id,
        "session_memory_mb": session.calculate_memory_mb(),
        "path_guidance": [
            "Use ingest_dataset(path) for host-visible absolute paths.",
            "Polars users: write to Parquet (`df.write_parquet(...)`), then ingest the path.",
            "SANDBOX ISOLATION: If your file is in a sandbox (e.g. /home/claude), "
            "use the 'Cache-Bridge' pattern: Copy the file to the `cache_dir` shown "
            "above, then call `ingest_dataset(path)` on the destination.",
            "Chunked/Base64 uploads are a last-resort legacy fallback for remote-only "
            "servers. DO NOT use them for local files.",
        ],
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Ingest tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def ingest_dataset(path: str, ctx: Context = None) -> str:
    """
    Register a host-visible absolute dataset path and return a stable
    dataset_id handle. Supports CSV and Parquet. Use this only when the
    MCP server can read the path directly.
    """
    try:
        record = registry.register_dataset(path)
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except FileNotFoundError:
        return path_access_error(
            "ingest_dataset",
            path,
            missing_action=(
                "Ask the user for a host-visible absolute dataset path, then call "
                "ingest_dataset again."
            ),
            sandbox_action=(
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked "
                "uploads. Run a bash script to copy the file to the `cache_dir` "
                "(call `get_runtime_context` to find it), then retry "
                "`ingest_dataset(path)` with the new path."
            ),
        )
    except PermissionError:
        return mcp_error(
            "ingest_dataset",
            "Dataset path exists but is not readable by the MCP server.",
            error_code="FILE_PERMISSION_DENIED",
            agent_action="Provide a readable host-visible dataset path.",
            details={"path_context": {"attempted_path": path}},
        )
    except Exception as e:
        return mcp_error("ingest_dataset", str(e))


@mcp.tool()
async def begin_dataset_upload(
    filename: str,
    media_type: str = "text/csv",
    ctx: Context = None,
) -> str:
    """
    Begin a staged server-side upload for a dataset that is not reachable
    by path. Use this for larger sandboxed uploads, then append chunks
    and finalize to obtain a dataset_id.
    """
    try:
        record = registry.begin_upload(filename, media_type=media_type)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("begin_dataset_upload", str(e))


@mcp.tool()
async def append_dataset_chunk(
    upload_id: str,
    chunk: str,
    encoding: str = "base64",
    ctx: Context = None,
) -> str:
    """
    Append one chunk to a staged dataset upload. Use base64 encoding by
    default to avoid newline and control-character corruption.
    """
    try:
        if encoding == "base64":
            try:
                chunk_bytes = base64.b64decode(chunk, validate=True)
            except Exception:
                return mcp_error(
                    "append_dataset_chunk",
                    "Failed to decode base64 chunk payload.",
                    error_code="CHUNK_DECODE_FAILED",
                    agent_action=(
                        "Re-encode the chunk with standard base64 (no whitespace)."
                    ),
                )
        elif encoding in ("utf-8", "utf8", "text"):
            chunk_bytes = chunk.encode("utf-8")
        else:
            return mcp_error(
                "append_dataset_chunk",
                f"Unsupported chunk encoding '{encoding}'.",
                error_code="UNSUPPORTED_ENCODING",
                agent_action="Use encoding='base64' or 'utf-8'.",
            )
        record = registry.append_upload_chunk(upload_id, chunk_bytes)
        if record is None:
            return unknown_handle_error("append_dataset_chunk", "upload_id", upload_id)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("append_dataset_chunk", str(e))


@mcp.tool()
async def finalize_dataset_upload(upload_id: str, ctx: Context = None) -> str:
    """
    Finalize a staged upload and register it as a dataset_id for
    downstream tools.
    """
    try:
        record = registry.finalize_upload(upload_id)
        if record is None:
            return unknown_handle_error(
                "finalize_dataset_upload", "upload_id", upload_id
            )
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("finalize_dataset_upload", str(e))


# ---------------------------------------------------------------------------
# Characterization tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def characterize_dataset(
    dataset_id: str = "",
    data_path: str = "",
    ctx: Context = None,
) -> str:
    """
    Probe a dataset and return a sparse per-column schema: dtype,
    n_unique, missing percent. Use ``probe_columns`` for sample values
    and richer per-column inspection.
    """
    try:
        session = _get_session(ctx)
        df, normalized_path = await _load_session_dataframe(
            session, dataset_id=dataset_id, data_path=data_path
        )
        n_rows = int(len(df))
        n_cols = int(df.shape[1])
        column_profiles: list[dict[str, Any]] = []
        for col in df.columns:
            series = df[col]
            n_missing = int(series.isna().sum())
            column_profiles.append(
                {
                    "name": str(col),
                    "dtype": str(series.dtype),
                    "n_unique": int(series.nunique(dropna=True)),
                    "n_missing": n_missing,
                    "missing_pct": round(n_missing / n_rows * 100, 3)
                    if n_rows
                    else 0.0,
                }
            )
        total_missing = int(df.isna().sum().sum())
        payload = {
            "status": "ok",
            "dataset_id": dataset_id or None,
            "data_path": normalized_path,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "total_missing": total_missing,
            "overall_missing_pct": (
                round(total_missing / (n_rows * n_cols) * 100, 3)
                if n_rows and n_cols
                else 0.0
            ),
            "column_profiles": column_profiles,
        }
        return json.dumps(payload, indent=2)
    except LookupError:
        return unknown_handle_error("characterize_dataset", "dataset_id", dataset_id)
    except FileNotFoundError:
        return path_access_error("characterize_dataset", data_path or dataset_id)
    except ToolError as e:
        return mcp_error("characterize_dataset", str(e))
    except Exception as e:
        return mcp_error("characterize_dataset", str(e))


@mcp.tool()
async def probe_columns(
    dataset_id: str,
    columns: list[str],
    ctx: Context = None,
) -> str:
    """
    Generate rich, detailed profiles for specific columns (max 20):
    sample values, top frequencies, missing patterns.
    """
    try:
        if len(columns) > 20:
            return mcp_error(
                "probe_columns",
                "probe_columns supports at most 20 columns per call.",
                error_code="TOO_MANY_COLUMNS",
                agent_action="Split your request into multiple probe_columns calls.",
            )
        session = _get_session(ctx)
        df, normalized_path = await _load_session_dataframe(
            session, dataset_id=dataset_id
        )
        profiles: list[dict[str, Any]] = []
        for col in columns:
            if col not in df.columns:
                profiles.append({"name": col, "error": "COLUMN_NOT_FOUND"})
                continue
            series = df[col]
            non_null = series.dropna()
            sample_values = non_null.head(5).tolist()
            top_values = (
                non_null.value_counts().head(5).to_dict() if len(non_null) else {}
            )
            profile: dict[str, Any] = {
                "name": str(col),
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique(dropna=True)),
                "n_missing": int(series.isna().sum()),
                "sample_values": [_coerce_jsonable(v) for v in sample_values],
                "top_values": {str(k): int(v) for k, v in top_values.items()},
            }
            if pd.api.types.is_numeric_dtype(series):
                profile["min"] = _coerce_jsonable(
                    None if non_null.empty else float(non_null.min())
                )
                profile["max"] = _coerce_jsonable(
                    None if non_null.empty else float(non_null.max())
                )
                profile["mean"] = _coerce_jsonable(
                    None if non_null.empty else float(non_null.mean())
                )
                profile["std"] = _coerce_jsonable(
                    None if non_null.empty else float(non_null.std())
                )
            profiles.append(profile)
        return json.dumps(
            {
                "status": "ok",
                "dataset_id": dataset_id,
                "data_path": normalized_path,
                "column_profiles": profiles,
            },
            indent=2,
        )
    except LookupError:
        return unknown_handle_error("probe_columns", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error("probe_columns", str(e))


def _coerce_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


@dataclass
class SweepInputError(Exception):
    reason: str
    error_code: str
    agent_action: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.reason


def _prepare_dataframe_for_sweep(
    df: pd.DataFrame, *, kwargs: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    drop_cols = kwargs.get("drop_cols") or []
    missingness_thresh = kwargs.get("missingness_thresh")
    encode_categoricals = kwargs.get("encode_categoricals", True)

    invalid_drop_cols = sorted(set(drop_cols) - set(df.columns))
    if invalid_drop_cols:
        raise SweepInputError(
            reason=f"Invalid drop_cols entries: {invalid_drop_cols}",
            error_code="INVALID_DROP_COLS",
            agent_action=(
                "Provide only columns that exist in the dataset, or remove invalid "
                "names from imputation.drop_cols."
            ),
            details={"invalid_drop_cols": invalid_drop_cols},
        )

    threshold_drop_cols: list[str] = []
    if missingness_thresh is not None:
        missingness = df.isna().mean()
        threshold_drop_cols = sorted(
            [
                str(col)
                for col in missingness[missingness > float(missingness_thresh)].index
            ]
        )

    combined_drop_cols = sorted(set(drop_cols) | set(threshold_drop_cols))
    prepared = df.drop(columns=combined_drop_cols, errors="ignore")
    if prepared.shape[1] == 0:
        raise SweepInputError(
            reason="All columns were dropped before sweep execution.",
            error_code="ALL_COLUMNS_DROPPED",
            agent_action=(
                "Reduce missingness_thresh or drop fewer columns so at least one "
                "feature remains."
            ),
            details={
                "drop_cols": drop_cols,
                "threshold_drop_cols": threshold_drop_cols,
            },
        )

    categorical_columns = [
        str(col)
        for col in prepared.columns
        if pd.api.types.is_object_dtype(prepared[col])
        or pd.api.types.is_string_dtype(prepared[col])
        or pd.api.types.is_categorical_dtype(prepared[col])
    ]
    if categorical_columns and not encode_categoricals:
        raise SweepInputError(
            reason=f"Unsupported string columns detected: {categorical_columns}.",
            error_code="UNSUPPORTED_STRING_COLUMNS",
            agent_action=(
                "Enable imputation.encode_categoricals or add these columns to "
                "imputation.drop_cols."
            ),
            details={"string_columns": categorical_columns},
        )

    categorical_codebooks: dict[str, list[str]] = {}
    for col in categorical_columns:
        cat = pd.Categorical(prepared[col])
        categories = [str(v) for v in cat.categories.tolist()]
        categorical_codebooks[col] = categories
        codes = pd.Series(cat.codes, index=prepared.index).replace(-1, np.nan)
        prepared[col] = codes.astype(float)

    return prepared, combined_drop_cols, categorical_codebooks


def _decode_categorical_columns(
    imputed: pd.DataFrame, codebooks: dict[str, list[str]]
) -> pd.DataFrame:
    if not codebooks:
        return imputed

    decoded = imputed.copy()
    for col, categories in codebooks.items():
        if col not in decoded.columns:
            continue
        if not categories:
            decoded[col] = np.nan
            continue
        numeric = pd.to_numeric(decoded[col], errors="coerce")
        is_observed = numeric.notna()
        rounded = np.rint(numeric[is_observed]).astype(int)
        clipped = np.clip(rounded, 0, len(categories) - 1)
        restored = pd.Series(np.nan, index=decoded.index, dtype=object)
        restored.loc[is_observed] = [categories[idx] for idx in clipped]
        decoded[col] = restored
    return decoded


# ---------------------------------------------------------------------------
# Configure tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_grids(ctx: Context = None) -> str:
    """
    Return the built-in imputation grids registered with Phil's
    ``GridGallery``, including method lists and declarative metadata
    (intent, suitability, affinity, time complexity, scale limits).
    """
    try:
        payload = {
            "status": "ok",
            "grids": list_builtin_grids(),
            "default": "default",
            "custom_note": (
                "Set imputation.grid='custom' and provide imputation.custom = "
                "{methods, modules, grids} to use a bespoke grid."
            ),
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("list_grids", str(e))


@mcp.tool()
async def recommend_grid(
    dataset_id: str = "",
    data_path: str = "",
    ctx: Context = None,
) -> str:
    """
    Recommend a built-in imputation grid from dataset scale, categorical
    cardinality, and missingness heuristics. Prefer this after
    ``characterize_dataset`` and before ``create_config``.
    """
    try:
        session = _get_session(ctx)
        df, normalized_path = await _load_session_dataframe(
            session, dataset_id=dataset_id, data_path=data_path
        )
        payload = recommend_grid_for_dataframe(df)
        payload["dataset_id"] = dataset_id or None
        payload["data_path"] = normalized_path
        return json.dumps(payload, indent=2)
    except LookupError:
        return unknown_handle_error("recommend_grid", "dataset_id", dataset_id)
    except FileNotFoundError:
        return path_access_error("recommend_grid", data_path or dataset_id)
    except ToolError as e:
        return mcp_error("recommend_grid", str(e))
    except Exception as e:
        return mcp_error("recommend_grid", str(e))


@mcp.tool()
async def create_config(
    dataset_id: str,
    grid: str = "default",
    samples: int = 30,
    random_state: int = 42,
    max_iter: int = 5,
    run_name: str = "phil_sweep",
    intent: str = "",
    ctx: Context = None,
) -> str:
    """
    Generate a canonical Phil YAML config for an ingested ``dataset_id``.

    The generated config covers run metadata, imputation grid + sample
    count, and the ECT magic method defaults. Persist it in-session by
    passing the YAML to ``run_imputation_sweep`` or refining it via
    ``refine_active_config``.
    """
    try:
        record = _resolve_dataset_record(dataset_id)
        if grid not in GRID_INTENTS and grid != "custom":
            return mcp_error(
                "create_config",
                f"Unknown imputation grid '{grid}'.",
                error_code="UNKNOWN_GRID",
                agent_action="Call list_grids to see all available built-in grids.",
                details={"grid": grid, "available": sorted(GRID_INTENTS.keys())},
            )
        config = default_config_dict(run_name=run_name, data=record.path)
        config["imputation"]["grid"] = grid
        config["imputation"]["samples"] = int(samples)
        config["imputation"]["random_state"] = int(random_state)
        config["imputation"]["max_iter"] = int(max_iter)
        if grid == "custom":
            config["imputation"]["custom"] = {
                "methods": [],
                "modules": [],
                "grids": [],
            }
        config_yaml = yaml.safe_dump(config, sort_keys=False)
        session = _get_session(ctx)
        session.active_config_yaml = config_yaml
        session.active_config_dataset_id = dataset_id

        payload = {
            "status": "ok",
            "dataset_id": dataset_id,
            "data_path": record.path,
            "grid": grid,
            "grid_intent": GRID_INTENTS.get(grid, ""),
            "intent": intent,
            "config_yaml": config_yaml,
        }
        return json.dumps(payload, indent=2)
    except LookupError:
        return unknown_handle_error("create_config", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error("create_config", str(e))


@mcp.tool()
async def validate_config(
    config_yaml: str,
    dataset_id: str = "",
    ctx: Context = None,
) -> str:
    """
    Validate the shape of a Phil MCP config YAML and normalize it.
    Pass ``dataset_id`` to validate against an ingested dataset.
    """
    try:
        dataset_path = _resolve_dataset_path(dataset_id) if dataset_id else None
        report = validate_config_yaml(config_yaml, dataset_path=dataset_path)
        if dataset_id:
            _get_session(ctx).dataset_id = dataset_id
        return render_validation_report(report)
    except LookupError:
        return unknown_handle_error("validate_config", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error("validate_config", str(e))


@mcp.tool()
async def refine_config(
    config_yaml: str,
    overrides: dict[str, Any],
    ctx: Context = None,
) -> str:
    """
    Apply dotted-path overrides to a Phil config YAML and return the
    normalized result. Valid keys include e.g. ``imputation.samples``,
    ``imputation.grid``, and ``magic.num_thetas``.
    """
    try:
        result = apply_overrides(config_yaml, overrides)
        payload = {
            "status": "ok",
            "applied_overrides": result.applied_overrides,
            "diff": result.diff,
            "config_yaml": result.config_yaml,
        }
        return json.dumps(payload, indent=2)
    except ValueError as e:
        return mcp_error(
            "refine_config",
            str(e),
            error_code="UNKNOWN_OVERRIDE_KEY",
            agent_action=(
                "Use only valid override keys. See error message for the valid list."
            ),
        )
    except Exception as e:
        return mcp_error("refine_config", str(e))


@mcp.tool()
async def get_active_config(ctx: Context = None) -> str:
    """Return the active in-session config, if any."""
    session = _get_session(ctx)
    if not session.active_config_yaml:
        return mcp_error(
            "get_active_config",
            "No active config in session. Run create_config or refine_active_config first.",
            error_code="ACTIVE_CONFIG_MISSING",
            agent_action="Create or refine a config before requesting the active session config.",
        )
    return json.dumps(
        {
            "status": "ok",
            "config_yaml": session.active_config_yaml,
            "dataset_id": session.active_config_dataset_id,
        },
        indent=2,
    )


@mcp.tool()
async def refine_active_config(overrides: dict[str, Any], ctx: Context = None) -> str:
    """Apply dotted-path overrides to the session's active config."""
    session = _get_session(ctx)
    if not session.active_config_yaml:
        return mcp_error(
            "refine_active_config",
            "No active config in session. Run create_config first.",
            error_code="ACTIVE_CONFIG_MISSING",
            agent_action="Call create_config(dataset_id) before refining the session config.",
        )
    try:
        result = apply_overrides(session.active_config_yaml, overrides)
        session.active_config_yaml = result.config_yaml
        payload = {
            "status": "ok",
            "applied_overrides": result.applied_overrides,
            "diff": result.diff,
            "config_yaml": result.config_yaml,
            "dataset_id": session.active_config_dataset_id,
        }
        return json.dumps(payload, indent=2)
    except ValueError as e:
        return mcp_error(
            "refine_active_config",
            str(e),
            error_code="UNKNOWN_OVERRIDE_KEY",
            agent_action=(
                "Use only valid override keys. See error message for the valid list."
            ),
        )
    except Exception as e:
        return mcp_error("refine_active_config", str(e))


# ---------------------------------------------------------------------------
# Run tool
# ---------------------------------------------------------------------------
@mcp.tool()
async def run_imputation_sweep(
    config_yaml: str = "",
    config_path: str = "",
    dataset_id: str = "",
    save_config: bool = False,
    ctx: Context = None,
) -> str:
    """
    Run a Phil imputation sweep and select the most representative
    imputation.

    Args:
        config_yaml: Inline YAML config (preferred).
        config_path: Path to a YAML config on disk.
        dataset_id: Dataset handle; overrides ``run.data`` in the config.
        save_config: If True, persist the resolved config YAML under the
            registry cache_dir for reproducibility.

    Returns:
        Markdown diff against the previous run followed by the full
        execution summary as JSON.
    """
    try:
        session = _get_session(ctx)
        if config_yaml:
            current_yaml = config_yaml
        elif config_path:
            try:
                _validate_config_path(config_path)
            except FileNotFoundError:
                return path_access_error("run_imputation_sweep", config_path)
            with open(config_path) as f:
                current_yaml = f.read()
        elif session.active_config_yaml:
            current_yaml = session.active_config_yaml
        else:
            return mcp_error(
                "run_imputation_sweep",
                "Provide config_yaml, config_path, or establish an active config first.",
                error_code="CONFIG_MISSING",
                agent_action="Call create_config(dataset_id) to seed the session config.",
            )

        dataset_path = _resolve_dataset_path(dataset_id) if dataset_id else None
        report = validate_config_yaml(current_yaml, dataset_path=dataset_path)
        if not report.ok or report.normalized_yaml is None:
            return mcp_error(
                "run_imputation_sweep",
                "Config validation failed before execution.",
                error_code=report.error_code or "CONFIG_VALIDATION_FAILED",
                agent_action=report.agent_action,
                details={
                    "resolved_dataset_path": report.resolved_dataset_path,
                    "issues": [dataclasses.asdict(i) for i in report.issues],
                },
            )
        current_yaml = report.normalized_yaml
        session.active_config_yaml = current_yaml

        kwargs = config_to_phil_kwargs(current_yaml)
        effective_dataset_id = dataset_id
        effective_data_path = kwargs["data_path"] or dataset_path or ""
        if not effective_data_path and not effective_dataset_id:
            return mcp_error(
                "run_imputation_sweep",
                "No dataset specified. Pass dataset_id or set run.data in the config.",
                error_code="DATASET_MISSING",
                agent_action="Call ingest_dataset(path) or set run.data in the YAML.",
            )

        df, normalized_path = await _load_session_dataframe(
            session,
            dataset_id=effective_dataset_id,
            data_path=effective_data_path,
        )
        session.active_config_dataset_id = (
            effective_dataset_id
            or _dataset_id_for_path(session.active_config_dataset_id, normalized_path)
            or session.active_config_dataset_id
        )

        prepared_df, dropped_columns, categorical_codebooks = (
            _prepare_dataframe_for_sweep(df, kwargs=kwargs)
        )
        total_missing = int(prepared_df.isna().sum().sum())
        if total_missing == 0:
            return mcp_error(
                "run_imputation_sweep",
                (
                    "Dataset contains no missing values after applying drop controls; "
                    "Phil has nothing to impute."
                ),
                error_code="NO_MISSING_VALUES",
                agent_action="Pass a dataset containing missing values, or skip imputation.",
                details={
                    "n_rows": int(len(prepared_df)),
                    "n_cols": int(prepared_df.shape[1]),
                    "dropped_columns": dropped_columns,
                },
            )

        loop = asyncio.get_running_loop()

        def progress_callback(stage: str, fraction: float) -> None:
            if ctx is None:
                return
            try:
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=fraction, total=1.0, message=stage),
                    loop,
                )
            except RuntimeError:
                pass

        def _fit() -> tuple[Phil, pd.DataFrame, list[np.ndarray]]:
            progress_callback("imputing", 0.0)
            phil_obj = Phil(
                samples=kwargs["samples"],
                param_grid=kwargs["param_grid"],
                magic=kwargs["magic"],
                config=kwargs["config"],
                random_state=kwargs["random_state"],
            )
            progress_callback("imputing", 0.1)
            try:
                imputed = phil_obj.fit(prepared_df, max_iter=kwargs["max_iter"])
            except Exception as exc:  # pragma: no cover - defensive
                raise exc
            progress_callback("scoring", 0.85)
            descriptors = list(phil_obj.magic_descriptors)
            progress_callback("done", 1.0)
            return phil_obj, imputed, descriptors

        try:
            phil_obj, imputed, descriptors = await asyncio.to_thread(_fit)
        except ImportError as e:
            return mcp_error(
                "run_imputation_sweep",
                f"Required backend not available: {e}",
                error_code="MAGIC_BACKEND_MISSING",
                agent_action=(
                    "Install the ECT backend (`trailed`) before running sweeps."
                ),
            )
        except ValueError as e:
            return mcp_error(
                "run_imputation_sweep",
                f"{type(e).__name__}: {e}",
                error_code="SWEEP_VALIDATION_FAILED",
                agent_action="Inspect the dataset and config; adjust grid or samples.",
            )

        imputed = _decode_categorical_columns(imputed, categorical_codebooks)

        session.phil = phil_obj
        session.imputed = imputed
        session.descriptors = descriptors
        session.categorical_codebooks = categorical_codebooks
        session.dropped_columns = dropped_columns

        stats = _descriptor_stats(descriptors)
        method_counts = _method_counts(phil_obj)
        selected_index = int(getattr(phil_obj, "closest_index", -1))
        imputed_columns = [str(c) for c in imputed.columns]

        output_path: str | None = None
        out_section = yaml.safe_load(current_yaml).get("output", {}) or {}
        write_csv = out_section.get("write_csv") or ""
        write_parquet = out_section.get("write_parquet") or ""
        if write_csv:
            target = _normalize_data_path(write_csv)
            await asyncio.to_thread(_write_csv, imputed, target)
            output_path = target
        if write_parquet:
            target = _normalize_data_path(write_parquet)
            await asyncio.to_thread(_write_parquet, imputed, target)
            output_path = target

        saved_config_path: str | None = None
        if save_config:
            saved_dir = registry.cache_dir / "configs"
            saved_dir.mkdir(parents=True, exist_ok=True)
            target = saved_dir / f"{kwargs['run_name']}_{int(time.time())}.yaml"
            target.write_text(current_yaml)
            saved_config_path = str(target)

        record = registry.save_run(
            dataset_id=session.active_config_dataset_id,
            config_yaml=current_yaml,
            selected_index=selected_index,
            n_candidates=len(descriptors),
            descriptor_stats=stats,
            imputed_columns=imputed_columns,
            output_path=output_path,
            summary={
                "run_name": kwargs["run_name"],
                "grid": _format_grid_field(yaml.safe_load(current_yaml)),
                "samples": kwargs["samples"],
                "max_iter": kwargs["max_iter"],
                "random_state": kwargs["random_state"],
                "method_counts": method_counts,
                "n_rows": int(len(prepared_df)),
                "n_cols": int(prepared_df.shape[1]),
                "total_missing": total_missing,
                "dropped_columns": dropped_columns,
                "encoded_categorical_columns": sorted(categorical_codebooks.keys()),
                "saved_config_path": saved_config_path,
            },
        )
        session.latest_run_id = record.run_id
        sweep_record = SweepRecord(
            timestamp=time.time(),
            config_yaml=current_yaml,
            run_id=record.run_id,
            selected_index=selected_index,
            n_candidates=len(descriptors),
            descriptor_stats=stats,
        )
        previous = session.sweep_history[-1] if session.sweep_history else None
        session.sweep_history.append(sweep_record)

        diff_md = _render_sweep_diff(previous, sweep_record)
        payload = {
            "status": "ok",
            "run_id": record.run_id,
            "dataset_id": session.active_config_dataset_id,
            "selected_index": selected_index,
            "n_candidates": len(descriptors),
            "descriptor_stats": stats,
            "method_counts": method_counts,
            "imputed_columns": imputed_columns,
            "config_yaml": current_yaml,
            "output_path": output_path,
            "saved_config_path": saved_config_path,
            "dropped_columns": dropped_columns,
            "encoded_categorical_columns": sorted(categorical_codebooks.keys()),
            "diff_markdown": diff_md,
        }
        return json.dumps(payload, indent=2)
    except LookupError:
        return unknown_handle_error("run_imputation_sweep", "dataset_id", dataset_id)
    except FileNotFoundError as e:
        return path_access_error("run_imputation_sweep", str(e))
    except SweepInputError as e:
        return mcp_error(
            "run_imputation_sweep",
            e.reason,
            error_code=e.error_code,
            agent_action=e.agent_action,
            details=e.details,
        )
    except Exception as e:
        return mcp_error("run_imputation_sweep", str(e))


def _render_sweep_diff(previous: SweepRecord | None, current: SweepRecord) -> str:
    if previous is None:
        return (
            "| Field | Value |\n|---|---|\n"
            f"| run_id | {current.run_id} |\n"
            f"| selected_index | {current.selected_index} |\n"
            f"| n_candidates | {current.n_candidates} |\n"
            f"| mean_pairwise_l2 | {current.descriptor_stats.get('mean_pairwise_l2', 0):.4f} |\n"
        )
    rows = ["| Field | Previous | Current |", "|---|---|---|"]
    rows.append(f"| run_id | {previous.run_id} | {current.run_id} |")
    rows.append(
        f"| selected_index | {previous.selected_index} | {current.selected_index} |"
    )
    rows.append(f"| n_candidates | {previous.n_candidates} | {current.n_candidates} |")
    rows.append(
        "| mean_pairwise_l2 | "
        f"{previous.descriptor_stats.get('mean_pairwise_l2', 0):.4f} | "
        f"{current.descriptor_stats.get('mean_pairwise_l2', 0):.4f} |"
    )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Diagnose tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def diagnose_sweep(run_id: str = "", ctx: Context = None) -> str:
    """
    Summarize the descriptor spread and candidate composition of a
    completed sweep. Defaults to the latest run in the session.
    """
    try:
        session = _get_session(ctx)
        if not run_id:
            run_id = session.latest_run_id or ""
        if not run_id:
            return mcp_error(
                "diagnose_sweep",
                "No run_id provided and no sweep recorded in this session.",
                error_code="NO_SWEEP_AVAILABLE",
                agent_action="Run run_imputation_sweep first.",
            )
        record = registry.get_run(run_id)
        if record is None:
            return unknown_handle_error("diagnose_sweep", "run_id", run_id)
        payload = {
            "status": "ok",
            "run_id": run_id,
            "dataset_id": record.dataset_id,
            "selected_index": record.selected_index,
            "n_candidates": record.n_candidates,
            "descriptor_stats": record.descriptor_stats,
            "summary": record.summary,
            "config_yaml": record.config_yaml,
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("diagnose_sweep", str(e))


@mcp.tool()
async def get_candidate_descriptors(
    run_id: str = "",
    top_k: int = 5,
    ctx: Context = None,
) -> str:
    """
    Return the ``top_k`` candidate imputations from the latest sweep ranked
    by closeness to the mean descriptor. Includes the selected index for
    reference.
    """
    try:
        session = _get_session(ctx)
        if not session.descriptors:
            return mcp_error(
                "get_candidate_descriptors",
                "No descriptors cached in this session. Run a sweep first.",
                error_code="NO_DESCRIPTORS",
                agent_action="Call run_imputation_sweep first.",
            )
        if run_id and run_id != session.latest_run_id:
            return mcp_error(
                "get_candidate_descriptors",
                "Requested run_id is not the latest in-session run.",
                error_code="RUN_NOT_CACHED",
                agent_action="Re-run the sweep, or use diagnose_sweep for persisted runs.",
                details={"latest_run_id": session.latest_run_id},
            )
        stacked = np.stack(session.descriptors)
        avg = stacked.mean(axis=0)
        norms = np.linalg.norm(
            (stacked - avg).reshape(len(session.descriptors), -1), axis=1
        )
        order = np.argsort(norms)
        top_k = max(1, min(top_k, len(order)))
        ranking = [
            {
                "rank": rank + 1,
                "index": int(order[rank]),
                "distance_to_mean": float(norms[order[rank]]),
            }
            for rank in range(top_k)
        ]
        selected = int(getattr(session.phil, "closest_index", -1))
        return json.dumps(
            {
                "status": "ok",
                "run_id": session.latest_run_id,
                "selected_index": selected,
                "ranking": ranking,
            },
            indent=2,
        )
    except Exception as e:
        return mcp_error("get_candidate_descriptors", str(e))


@mcp.tool()
async def compare_sweeps(run_a: str, run_b: str, ctx: Context = None) -> str:
    """
    Compare two persisted sweep runs by config and descriptor statistics.
    """
    try:
        rec_a = registry.get_run(run_a)
        if rec_a is None:
            return unknown_handle_error("compare_sweeps", "run_id", run_a)
        rec_b = registry.get_run(run_b)
        if rec_b is None:
            return unknown_handle_error("compare_sweeps", "run_id", run_b)
        payload = {
            "status": "ok",
            "run_a": {
                "run_id": rec_a.run_id,
                "selected_index": rec_a.selected_index,
                "n_candidates": rec_a.n_candidates,
                "descriptor_stats": rec_a.descriptor_stats,
                "summary": rec_a.summary,
            },
            "run_b": {
                "run_id": rec_b.run_id,
                "selected_index": rec_b.selected_index,
                "n_candidates": rec_b.n_candidates,
                "descriptor_stats": rec_b.descriptor_stats,
                "summary": rec_b.summary,
            },
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("compare_sweeps", str(e))


@mcp.tool()
async def get_experiment_history(ctx: Context = None) -> str:
    """
    Return a markdown table of all imputation sweeps run in the current
    session. Use this to reason about trajectory across iterations.
    """
    session = _get_session(ctx)
    if not session.sweep_history:
        return (
            "No experiments run yet in this session.\n\n"
            "| Run | Grid | Samples | Selected | Candidates | Mean Pairwise L2 |\n"
            "|---|---|---|---|---|---|"
        )
    lines = [
        "| Run | Grid | Samples | Selected | Candidates | Mean Pairwise L2 |",
        "|---|---|---|---|---|---|",
    ]
    for i, record in enumerate(session.sweep_history, start=1):
        cfg = yaml.safe_load(record.config_yaml)
        grid = _format_grid_field(cfg)
        samples = cfg.get("imputation", {}).get("samples", "?")
        spread = record.descriptor_stats.get("mean_pairwise_l2", 0.0)
        lines.append(
            f"| {i} | {grid} | {samples} | {record.selected_index} | "
            f"{record.n_candidates} | {spread:.4f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_sweep_summary(run_id: str = "", ctx: Context = None) -> str:
    """
    Return the persisted ``RunRecord`` for ``run_id`` (or the latest run
    in the session). Includes the resolved config and descriptor stats.
    """
    try:
        session = _get_session(ctx)
        if not run_id:
            run_id = session.latest_run_id or ""
        if not run_id:
            return mcp_error(
                "get_sweep_summary",
                "No run_id provided and no sweep recorded in this session.",
                error_code="NO_SWEEP_AVAILABLE",
                agent_action="Run run_imputation_sweep first.",
            )
        record = registry.get_run(run_id)
        if record is None:
            return unknown_handle_error("get_sweep_summary", "run_id", run_id)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("get_sweep_summary", str(e))


@mcp.tool()
async def export_imputed_data(
    output_path: str,
    run_id: str = "",
    ctx: Context = None,
) -> str:
    """
    Write the selected imputed DataFrame from the most recent in-session
    sweep to ``output_path``. Format is inferred from the extension
    (``.csv``, ``.parquet``, ``.feather``).
    """
    try:
        session = _get_session(ctx)
        if session.imputed is None:
            return mcp_error(
                "export_imputed_data",
                "No imputed dataset cached in this session. Run a sweep first.",
                error_code="NO_IMPUTED_DATA",
                agent_action="Call run_imputation_sweep first.",
            )
        if run_id and run_id != session.latest_run_id:
            return mcp_error(
                "export_imputed_data",
                "Requested run_id is not the latest in-session run.",
                error_code="RUN_NOT_CACHED",
                agent_action="Re-run the sweep before exporting from a non-cached run.",
                details={"latest_run_id": session.latest_run_id},
            )
        target = _normalize_data_path(output_path)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        lowered = target.lower()
        if lowered.endswith(".parquet") or lowered.endswith(".pq"):
            await asyncio.to_thread(_write_parquet, session.imputed, target)
        elif lowered.endswith(".feather") or lowered.endswith(".arrow"):
            await asyncio.to_thread(_write_feather, session.imputed, target)
        else:
            await asyncio.to_thread(_write_csv, session.imputed, target)
        return json.dumps(
            {
                "status": "ok",
                "run_id": session.latest_run_id,
                "output_path": target,
                "n_rows": int(len(session.imputed)),
                "n_cols": int(session.imputed.shape[1]),
            },
            indent=2,
        )
    except Exception as e:
        return mcp_error("export_imputed_data", str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Console-script entry point: ``phil-mcp``."""
    mcp.run()


if __name__ == "__main__":
    main()
