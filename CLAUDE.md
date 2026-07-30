# Phil

PyPI package: `philler` (import: `phil`). Format and lint with **Ruff** (`uvx ruff format phil/`, `uvx ruff check phil/`).

## Releasing

1. Branch `release/vX.Y.Z` from `main`.
2. Bump `pyproject.toml`, `phil/__init__.py`, `docs/source/conf.py`; update `CHANGELOG.md`.
3. PR titled `release: vX.Y.Z`, then after merge: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. Tag push runs `.github/workflows/release.yml` (GitHub Release → PyPI).
