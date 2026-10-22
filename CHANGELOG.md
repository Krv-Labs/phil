# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-22

Versioning restarts at 0.1.0 to reflect Phil's experimental public API. APIs may
change between minor releases during the 0.x development series. Earlier 1.x
entries below are retained as historical records from before this reset.

### Added

- Empirical predictability profiling and research worksheets for cited domain priors, with MCP tools `profile_predictability` and `generate_research_template` ([#21](https://github.com/Krv-Labs/phil/pull/21)).
- Per-target predictor masks and estimator settings, bounded imputation, and covariance-guided candidates using full numeric covariance priors aligned by variable name and scale ([#21](https://github.com/Krv-Labs/phil/pull/21)).
- Six ECT ensemble collapse strategies plus automatic selection, distribution diagnostics, strategy comparisons, and per-column between-imputation variability; MCP tools `list_collapse_strategies`, `inspect_ect_distribution`, and `collapse_ensemble` ([#21](https://github.com/Krv-Labs/phil/pull/21)).
- Dataset-directory configuration through `PHIL_DATASETS_DIR` and upload confinement through `PHIL_DISABLE_UPLOAD=1`, including suppression of category counts below five in confined `probe_columns` responses ([#21](https://github.com/Krv-Labs/phil/pull/21)).
- Phil branding and expanded API, programmatic, and MCP documentation with executable examples ([#22](https://github.com/Krv-Labs/phil/pull/22)).

### Changed

- **Breaking:** `Phil.fit()` and `Phil.transform()` now return original column names and units, including categorical columns, rather than encoded and scaled feature columns. Consumers that require numeric model inputs must add their own encoding and scaling ([#21](https://github.com/Krv-Labs/phil/pull/21)).
- Pooling fidelity evaluates the actual returned data after categorical voting and bounds. Direct `collapse_ect_ensemble()` callers must supply the fitted `preprocessor` and `magic_estimator` to measure pooling fidelity; unavailable distances and ratios are `None` ([#21](https://github.com/Krv-Labs/phil/pull/21)).

### Fixed

- Preserve observed values outside imputation bounds and retain dropped identifier columns through MCP sweep, collapse, and export ([#21](https://github.com/Krv-Labs/phil/pull/21)).
- Apply full covariance priors to each target's actual predictors instead of silently discarding them, and report categorical ensemble disagreement for single-candidate selectors ([#21](https://github.com/Krv-Labs/phil/pull/21)).
- Validate collapse temperature and trimming parameters, and correct documented tool names, configuration paths, and result access ([#21](https://github.com/Krv-Labs/phil/pull/21), [#22](https://github.com/Krv-Labs/phil/pull/22)).

## [1.1.0] - 2026-07-30

### Added

- **FastMCP server** ([#15](https://github.com/Krv-Labs/phil/pull/15)) — `phil-mcp` / `philler[mcp]` exposes Phil's imputation sweep pipeline as MCP tools for agents (Claude Desktop, Cursor, Gemini CLI, etc.).
- **Medical demo and covariate sampling** ([#15](https://github.com/Krv-Labs/phil/pull/15)) — local `demos/medical` workflow, `CovariateDistributionImputer`, masked iterative imputer / domain-knowledge support, and MDS visualization of ECT descriptor space.
- **Declarative grid metadata and `recommend_grid`** ([#19](https://github.com/Krv-Labs/phil/pull/19)) — `GridMetadata` for built-in grids, `phil://docs/imputation-matrix` resource, and a rule-based recommender with literature-guided sample budgets.

### Changed

- Installation docs and examples use `pip install philler` ([#17](https://github.com/Krv-Labs/phil/pull/17)).
- Dev tooling upgraded to Ruff 0.16 with lint cleanup across `phil/` ([#19](https://github.com/Krv-Labs/phil/pull/19)).

## [1.0.1] - 2026-04-01

### Fixed

- Import `enable_iterative_imputer` at package level so multiprocessing workers (e.g. Windows) can construct `IterativeImputer` ([#14](https://github.com/Krv-Labs/phil/pull/14), fixes [#13](https://github.com/Krv-Labs/phil/issues/13)).
- Pin `trailed>=0.1.1` and drop unused uv source configuration.

## [1.0.0] - 2026-03-27

### Changed

- Rename **Benson → Phil**; PyPI package name is **`philler`** (import package remains `phil`) ([#11](https://github.com/Krv-Labs/phil/pull/11)).
- Switch ECT backend to the Rust `trailed` package; reorganize gallery, imputation, magic, and transformer modules.

> Pre-rename releases shipped as **`benson` 0.1.0–0.1.5** in this repository and are not `philler` releases.
