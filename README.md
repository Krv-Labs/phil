<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/phil-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/phil-logo.svg">
    <img src="docs/source/_static/phil-logo.svg" alt="Phil" width="400">
  </picture>
</p>

<h3 align="center">representation-guided imputation for missing tabular data</h3>

<p align="center">
  <a href="https://pypi.org/project/philler/"><img src="https://img.shields.io/pypi/v/philler?color=3776AB&logo=python&logoColor=ffd43b" alt="PyPI"></a>
  <a href="https://pypi.org/project/philler/"><img src="https://img.shields.io/pypi/pyversions/philler" alt="Python versions"></a>
  <a href="https://github.com/Krv-Labs/phil/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Krv-Labs/phil" alt="License"></a>
</p>

Phil fills missing values in tabular data by comparing several possible completions. It describes each candidate's structure with the Euler Characteristic Transform (ECT), then selects a candidate close to the ensemble's center. You can also inspect the candidates and combine them with an ensemble collapse strategy.

**Install `philler`, import `phil`.** Requires Python 3.10 or newer. Phil is experimental (`0.x`); public APIs may change between minor releases.

## Install

```bash
pip install philler
```

To use Phil from an AI assistant, install the optional MCP server:

```bash
pip install "philler[mcp]"
```

## Quick start

```python
import pandas as pd
from phil import Phil

df = pd.DataFrame({
    "x": [1.0, 2.0, None, 4.0, 5.0, 6.0],
    "y": [2.0, None, 6.0, 8.0, 10.0, 12.0],
})

imputer = Phil(samples=10, random_state=42)
completed = imputer.fit(df)
print(completed)
```

`samples` limits how many grid candidates are explored. The result keeps the original columns and units, and observed values stay unchanged. Reuse the selected fitted pipeline with `imputer.transform(other_df)`, supplying the same columns.

Phil selects for structural representativeness. Assess the completed data against the needs of your downstream analysis.

## Explore the ensemble

- **Choose a grid:** pass `param_grid="default"`, `"sampling"`, `"finance"`, `"healthcare"`, `"marketing"`, or `"engineering"` to `Phil`.
- **Understand missing columns:** use `profile_predictability(df)` to examine empirical predictors and missingness patterns.
- **Add domain knowledge:** `build_research_worksheet(df)` helps organize research for cited predictor sets, bounds, and covariance priors.
- **Compare collapse strategies:** inspect the ensemble and choose a single candidate or pool several candidates without fitting them again.

After the quick start above:

```python
comparison = imputer.collapse(strategy="auto", compare_all=True)
completed = comparison.collapsed_df
print(comparison.strategy)
print(comparison.column_variability)
```

The default `Phil` selector chooses the candidate closest to the mean ECT descriptor. `collapse(strategy="auto")` uses distribution diagnostics to choose among six selection and pooling strategies. ECT computation uses the [`trailed`](https://pypi.org/project/trailed/) backend.

See the [programmatic guide](https://docs.krv.ai/phil/userGuides/programmatic.html) for domain priors and the [advanced guide](https://docs.krv.ai/phil/userGuides/advanced.html) for collapse diagnostics.

## Use with an AI assistant

Phil's MCP server lets Claude, Cursor, Gemini, and other MCP clients run the workflow from a file path. A typical request is:

> Use Phil to fill missing values in `/data/input.csv`. Inspect the candidate ensemble and export the result to `/data/completed.csv`.

With `uv` installed, add this server to your MCP client's configuration:

```json
{
  "mcpServers": {
    "phil": {
      "command": "uv",
      "args": ["tool", "run", "--from", "philler[mcp]", "phil-mcp"]
    }
  }
}
```

The client must be able to find `uv`, and the server must be able to read the dataset path. You can also launch an installed server with `phil-mcp`.

The tools cover ingest, profiling, research templates, grid configuration, imputation sweeps, ensemble inspection, collapse, and export. For Polars data, write a Parquet file and ingest its path.

See the [MCP guide](https://docs.krv.ai/phil/userGuides/mcp.html) for tool details and the [medical demo](demos/medical/README.md) for a local walkthrough.

## Use in a scikit-learn pipeline

For numeric features, `PhilTransformer` can be the imputation step in a predictive pipeline:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from phil import PhilTransformer

pipeline = Pipeline([
    ("impute", PhilTransformer(samples=10, random_state=42)),
    ("model", RandomForestClassifier(random_state=42)),
])

pipeline.fit(df, [0, 0, 0, 1, 1, 1])
```

Phil returns categorical columns in their original form. Add an encoding step before models that require numeric inputs.

## Documentation and development

- [User guides](https://docs.krv.ai/phil/) — installation, configuration, and examples.
- [Changelog](CHANGELOG.md) — release notes and the reset to the experimental 0.x series.
- [Agent guide](AGENTS.md) — package layout and contributor conventions.

Install contributor dependencies and run the suite:

```bash
uv sync --group dev --group mcp --group docs
uv run pytest -v
uv run ruff check phil/
uv run ruff format --check phil/
```

Build the documentation:

```bash
uv run --group docs sphinx-build -M html docs/source docs/build
```

---

<p align="left">
  <a href="https://krv.ai">
    <img src="https://raw.githubusercontent.com/Krv-Labs/topos/main/docs/source/_static/made-by-krv.svg" alt="Made by Krv Labs" height="24">
  </a>
</p>
