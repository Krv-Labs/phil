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

## PHASE I: INGEST, CHARACTERIZE & GENERATE RESEARCH TEMPLATE
1. Ingest: `ingest_dataset(path)` registers a CSV or Parquet file and
   returns a stable `dataset_id`. Prefer the handle everywhere downstream.
   Polars users: write to Parquet (`df.write_parquet(...)`) and ingest the
   path; the server reads it natively.
2. Characterize: `characterize_dataset(dataset_id)` returns a SPARSE
   schema — dtype, n_unique, missing percent — for every column. Use
   `probe_columns(dataset_id, ["col_a"])` for sample values and top
   frequencies. Max 20 columns per probe.
3. Profile & Research Template: Call `profile_predictability(dataset_id)`
   or `generate_research_template(dataset_id, domain_context="...")` to
   inspect empirical predictability (CV R2/accuracy), MAR driver scores,
   and receive a structured Research Worksheet & YAML Config Template.
   Phil stores zero domain knowledge itself — use the worksheet to search
   literature and populate `imputation.domain_knowledge` (`covariate_subsets`,
   `variable_bounds`, `covariance_matrix`, and required `citations`).
4. GATE: every column with ≥ 100% missingness must be dropped before the
   sweep — Phil cannot impute fully-empty features.

## PHASE II: CONFIGURE & VALIDATE
5. Recommend a grid: `recommend_grid(dataset_id)` applies scale,
   cardinality, and missingness heuristics and returns a concrete grid
   plus suggested `samples`. Optionally read the MCP resource
   `phil://docs/imputation-matrix` for the full comparison table.
6. Discover grids: `list_grids()` enumerates the named imputation grids
   (`default`, `sampling`, `finance`, `healthcare`, `marketing`,
   `engineering`) with declarative metadata (intent, suitability,
   affinity, time complexity, scale limits).
7. Create or populate the config: `create_config(dataset_id, grid="...")`
   or populate the template from `generate_research_template`. Refine via
   `refine_config` or `refine_active_config`.
8. Validate: `validate_config(config_yaml, dataset_id)` confirms the
   shape, resolves grid names, verifies required literature `citations`,
   and normalizes whitespace.

## PHASE III: RUN, NAVIGATE ECT DISTRIBUTION, COLLAPSE & EXPORT
9. Run: `run_imputation_sweep` fits `samples` candidate imputations,
   scores each with the ECT magic method, and collapses the ensemble into
   a single representation.
10. Navigate ECT Distribution: Call `inspect_ect_distribution()` and
    `list_collapse_strategies()` to inspect the geometry of the ECT
    signature distribution `{Psi_1, ..., Psi_M}` (outliers, multimodality,
    silhouette score, effective sample size).
11. Agentic Collapse: Call `collapse_ensemble(strategy=..., compare_all=True)`
    to navigate the distribution using `barycenter`, `trimmed_barycenter`,
    `frechet_medoid`, `euclidean_medoid`, `mode_medoid`,
    `cluster_barycenter`, or `auto`, and inspect `manifold_convexity_ratio`
    plus Rubin's per-column representational uncertainty.
12. Export: `export_imputed_data(output_path, run_id)` writes the collapsed
    imputed dataframe to disk (CSV, Parquet, or Feather).

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
