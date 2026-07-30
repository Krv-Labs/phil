# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
