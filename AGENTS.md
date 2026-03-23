# Agents

## Cursor Cloud specific instructions

**Benson** is a Python library for intelligent missing data imputation using topological data analysis. It is a pure Python library with no external services (no databases, no Docker, no API servers).

### Environment

- Python 3.13 (pinned in `.python-version`); `uv` is the package manager (lockfile: `uv.lock`).
- The `dect` dependency is fetched from GitHub (`https://github.com/aidos-lab/dect.git`) at install time — network access is required during `uv sync`.

### Key commands

All commands are run via `uv run` from the repository root. See `pyproject.toml` for the full dependency list and `.github/workflows/ci.yml` for CI parity.

| Task | Command |
|------|---------|
| Install deps | `uv sync --all-extras` |
| Run tests | `uv run pytest -v` |
| Lint (format check) | `uv run black --check --diff benson/ tests/` |
| Auto-format | `uv run black benson/ tests/` |

### Gotchas

- `torch` is a large dependency (~730 MB); initial `uv sync` can take 30+ seconds on a cold cache.
- The project has no `build` or `dev server` step — it is a library, not an application. Testing is purely via `pytest`.
