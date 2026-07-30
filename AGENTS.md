# Phil — Agent Guide

PyPI package: `philler` (import: `phil`).

Phil is a representation-guided imputation library for missing tabular data. It runs a grid of imputation strategies, computes Euler Characteristic Transform (ECT) descriptors for each candidate via the `trailed` backend, and selects the most representative imputation.

## Package structure

```
phil/
  __init__.py          # Public API exports
  phil.py              # Phil class — orchestrates impute → describe → select
  transformers.py      # PhilTransformer (sklearn TransformerMixin)
  gallery.py           # GridGallery, ProcessingGallery, MagicGallery
  visualization.py     # plot_mds
  imputation/
    config.py          # ImputationConfig, PreprocessingConfig (pydantic)
    distribution.py      # DistributionImputer (empirical sampling)
    covariate_distribution.py
    masked_iterative_imputer.py
  magic/
    base.py            # Magic ABC
    config.py          # ECTConfig (pydantic)
    ect.py             # ECT — wraps rust_backend
    rust_backend.py    # Adapter for the trailed ECT backend
  mcp/
    server.py          # FastMCP server (phil-mcp entry point)
    config.py          # MCP sweep configuration
    recommend.py       # Grid recommendation helpers
    registry.py        # Grid metadata registry
tests/
  conftest.py
  imputation/
  magic/
  mcp/
  phil/
docs/source/           # Sphinx documentation
demos/medical/         # Local end-to-end examples
```

## Commands

```bash
uv sync --all-extras          # install all dependencies
uv run pytest -v              # run tests
uvx ruff format phil/ tests/  # format code
uvx ruff check phil/ tests/   # lint
uv run sphinx-build -M html docs/source docs/build
```

## Key design notes

- **ECT backend**: `rust_backend.py` loads `trailed` at import time and raises `ModuleNotFoundError` if it is absent. `trailed` is sourced from the KRV private PyPI index (`krv-research` in `pyproject.toml`).
- **ECT.configure()**: unpacks `ECTConfig` fields onto the `ECT` instance as flat attributes (e.g. `self.num_thetas`). `generate()` reads these attributes directly rather than going through `self.config`.
- **Phil.fit()**: mutates `self.representations`, `self.magic_descriptors`, `self.closest_index`, and `self.pipeline` — these are set during `fit` and required by `transform`.
- **Representative selection**: `_select_representative` stacks descriptors, computes the mean, and returns the index of the descriptor with minimum L2 distance to the mean.
- **Imputation pipeline**: each candidate is a sklearn `Pipeline([preprocessor, IterativeImputer(estimator)])`. The preprocessor is a `ColumnTransformer` built by `ProcessingGallery`.
- **MCP server**: `phil/mcp/server.py` exposes tools for the full sweep workflow (`ingest_dataset`, `characterize_dataset`, `recommend_grid`, `run_imputation_sweep`, etc.). See `docs/source/userGuides/mcp.rst`.

## Conventions

- Formatting and linting: Ruff (enforced in CI)
- Python: >=3.10, tested on 3.10–3.13
- Package manager: `uv`
- No type stubs; type hints in function signatures only
- Minimize scope — match existing patterns, avoid unrelated changes

## Releasing

1. Branch `release/vX.Y.Z` from `main`.
2. Bump `pyproject.toml`, `phil/__init__.py`, `docs/source/conf.py`; update `CHANGELOG.md`.
3. PR titled `release: vX.Y.Z`, then after merge: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. Tag push runs `.github/workflows/release.yml` (GitHub Release → PyPI).
