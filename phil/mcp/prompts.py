"""
Opt-in workflow prompts for Phil MCP agents.

These are not injected into ``FastMCP(instructions=...)``. Agents request them
explicitly via the ``get_workflow_guide`` tool so clients that do not want
the opinionated workflow do not pay for it on every session.
"""

from __future__ import annotations


WORKFLOW_PROMPT = """\
# Phil Imputation Sweep Workflow

Choose the imputation that *represents* the data best, not the first one
that happens to converge.

## PHASE I: INGEST & CHARACTERIZE
1. Ingest: `ingest_dataset(path)` registers a CSV or Parquet file and
   returns a stable `dataset_id`. Prefer the handle everywhere downstream.
   Polars users: write to Parquet (`df.write_parquet(...)`) and ingest the
   path; the server reads it natively.
2. Characterize: `characterize_dataset(dataset_id)` returns a SPARSE
   schema — dtype, n_unique, missing percent — for every column. Use
   `probe_columns(dataset_id, ["col_a"])` for sample values and top
   frequencies. Max 20 columns per probe.
3. GATE: every column with ≥ 100% missingness must be dropped before the
   sweep — Phil cannot impute fully-empty features.

## PHASE II: CONFIGURE & VALIDATE
4. Discover grids: `list_grids()` enumerates the named imputation grids
   (`default`, `sampling`, `finance`, `healthcare`, `marketing`,
   `engineering`) with their method lists. Pick the one whose intent
   matches your dataset.
5. Create the config: `create_config(dataset_id, grid="...")` returns a
   canonical YAML scaffold. You can refine it with `refine_config` or
   keep it in-session via `refine_active_config`.
6. Validate: `validate_config(config_yaml, dataset_id)` confirms the
   shape, resolves grid names, and normalizes whitespace.

## PHASE III: RUN, DIAGNOSE, EXPORT
7. Run: `run_imputation_sweep` fits `samples` candidate imputations,
   scores each with the ECT magic method, and selects the candidate
   closest to the mean descriptor. Returns a markdown diff against the
   previous run.
8. Diagnose: `diagnose_sweep` reports descriptor spread, selected
   candidate index, and per-method candidate counts. If spread is near
   zero the grid is collapsing — broaden it. If spread is huge,
   consider raising `samples`.
9. Export: `export_imputed_data(run_id, output_path)` writes the chosen
   imputed dataframe to disk (CSV or Parquet by file extension).

## PHILOSOPHY
- Phil is a *representative selector*, not an optimizer. A wide grid
  yields more information; a narrow grid yields none.
- The ECT descriptor encodes shape, not error — small differences in
  descriptor space can still flag meaningful imputation shifts.
- Always pass `random_state` when you want a reproducible sweep.

## PATH VISIBILITY (Claude Desktop)
Claude Desktop sandboxes are isolated. DO NOT use chunked/base64 uploads
for local files. Use the 'Cache-Bridge' pattern:
1. Call `get_runtime_context` to find `cache_dir`.
2. Use a shell command to `cp` your file into that `cache_dir`.
3. Call `ingest_dataset(path)` on the new path.
This is 100x faster and avoids protocol overhead.
"""
