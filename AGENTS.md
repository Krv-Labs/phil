# Phil — Agent Guide

PyPI package: `philler` (import: `phil`).

Phil is a representation-guided imputation library for missing tabular data. It runs a grid of imputation strategies, computes Euler Characteristic Transform (ECT) descriptors for each candidate via the `trailed` backend, and selects the most representative imputation.

## Package structure

```
phil/
  __init__.py          # Public API exports
  phil.py              # Phil class — orchestrates impute → describe → select/collapse
  diagnostics.py       # Predictability profiling and literature research worksheet
  transformers.py      # PhilTransformer (sklearn TransformerMixin)
  gallery.py           # GridGallery, ProcessingGallery, MagicGallery
  visualization.py     # plot_mds
  imputation/
    config.py          # ImputationConfig, PreprocessingConfig, DomainKnowledge (pydantic)
    distribution.py      # DistributionImputer (empirical sampling)
    covariate_distribution.py
    masked_iterative_imputer.py
  magic/
    base.py            # Magic ABC
    collapse.py        # ECT signature distribution analysis & 6 collapse strategies
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
- **Representative selection & collapse**: `_select_representative` selects the candidate closest to the mean descriptor. `Phil.collapse()` supports 6 concrete ECT strategies plus `auto`, implemented in `phil/magic/collapse.py`. Pooling fidelity uses the returned DataFrame transformed through the fitted preprocessor; unavailable fidelity metrics are `None`.
- **Predictability & domain priors**: `phil/diagnostics.py` profiles empirical predictability (R²/balanced accuracy) per incomplete column (`profile_predictability`) and emits a YAML research worksheet (`build_research_worksheet`) with literature questions and prior slots. Users populate `DomainKnowledge`; the profiler does not automatically assign per-column imputers.
- **Imputation pipeline**: each candidate is a sklearn `Pipeline([preprocessor, imputer])`. `MaskedIterativeImputer` applies predictor masks and slices aligned covariance priors per target. Covariance priors use original numeric units and require numeric-only input. `fit()` and `transform()` return original columns and units; bounds affect only imputed cells.
- **MCP server**: `phil/mcp/server.py` exposes tools for the full sweep workflow (`ingest_dataset`, `characterize_dataset`, `profile_predictability`, `generate_research_template`, `recommend_grid`, `run_imputation_sweep`, `list_collapse_strategies`, `inspect_ect_distribution`, `collapse_ensemble`, etc.) and supports Phirewall confinement (`PHIL_DISABLE_UPLOAD=1`, `PHIL_CACHE_DIR`, `PHIL_DATASETS_DIR`). See `docs/source/userGuides/mcp.rst`.

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
