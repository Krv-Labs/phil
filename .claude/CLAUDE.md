# Phil — Claude Code Guide

## Project overview

`phil` is a Python library for representation-guided imputation of missing tabular data.
It lives in the `phil/` package (renamed from the old `benson/` package on the `rustify` branch).

## Package structure

```
phil/
  __init__.py          # Public API exports
  phil.py              # Phil class — orchestrates impute → describe → select
  transformers.py      # PhilTransformer (sklearn TransformerMixin)
  gallery.py           # GridGallery, ProcessingGallery, MagicGallery
  imputation/
    config.py          # ImputationConfig, PreprocessingConfig (pydantic)
    distribution.py    # DistributionImputer (empirical sampling)
  magic/
    base.py            # Magic ABC
    config.py          # ECTConfig (pydantic)
    ect.py             # ECT — wraps rust_backend
    rust_backend.py    # Adapter for the trailed ECT backend
tests/
  conftest.py
  imputation/          # DistributionImputer tests
  magic/               # ECT and Magic base tests
  phil/                # Phil end-to-end and unit tests
```

## Commands

```bash
uv sync --all-extras          # install all dependencies
uv run pytest -v              # run tests
uv run black phil/ tests/     # format code
uv run pdoc -d numpy phil phil.imputation phil.magic -o docs/api   # build docs
```

## Key design notes

- **ECT backend**: `rust_backend.py` loads `trailed` at import time and raises `ModuleNotFoundError` if it is absent. `trailed` is sourced from the KRV private PyPI index (`krv-research` in `pyproject.toml`).
- **ECT.configure()**: unpacks `ECTConfig` fields onto the `ECT` instance as flat attributes (e.g. `self.num_thetas`). `generate()` reads these attributes directly rather than going through `self.config`.
- **Phil.fit()**: mutates `self.representations`, `self.magic_descriptors`, `self.closest_index`, and `self.pipeline` — these are set during `fit` and required by `transform`.
- **Representative selection**: `_select_representative` stacks descriptors, computes the mean, and returns the index of the descriptor with minimum L2 distance to the mean.
- **Imputation pipeline**: each candidate is a sklearn `Pipeline([preprocessor, IterativeImputer(estimator)])`. The preprocessor is a `ColumnTransformer` built by `ProcessingGallery`.

## Conventions

- Formatting: `black` (enforced in CI)
- Python: >=3.10, tested on 3.10–3.13
- Package manager: `uv`
- No type stubs; type hints in function signatures only
