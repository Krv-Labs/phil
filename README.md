# Phil

**Representation-guided imputation for missing tabular data** — PyPI package [`philler`](https://pypi.org/project/philler/) (import: `phil`).

Phil runs a grid of imputation strategies, scores each candidate with an Euler Characteristic Transform (ECT) descriptor via the [`trailed`](https://pypi.org/project/trailed/) backend, and selects the most representative result.

**Impute → Describe → Select → Transform**

## Installation

```bash
pip install philler          # core library
pip install "philler[mcp]"   # + FastMCP server for agents
```

## Quick start

```python
import pandas as pd
from phil import Phil

df = pd.read_csv("data_with_missing.csv")

phil = Phil(samples=30, random_state=42)
imputed_df = phil.fit(df)
new_df = phil.transform(new_data)  # reuse fitted pipeline
```

<details>
<summary><strong>MCP server</strong> — run sweeps from Claude, Cursor, Gemini CLI, etc.</summary>

Install the `mcp` extra and start the server:

```bash
pip install "philler[mcp]"
phil-mcp
# or ephemeral: uv tool run --from "philler[mcp]" phil-mcp
```

Example Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "phil": {
      "command": "uv tool run",
      "args": ["--from", "philler[mcp]", "phil-mcp"]
    }
  }
}
```

Key tools: `ingest_dataset`, `characterize_dataset`, `recommend_grid`, `list_grids`, `create_config`, `validate_config`, `run_imputation_sweep`, `diagnose_sweep`, `export_imputed_data`.

Agents can read `phil://docs/imputation-matrix` for grid comparison metadata. Polars users write to Parquet and ingest the file path.

See [`docs/source/userGuides/mcp.rst`](docs/source/userGuides/mcp.rst) for the full tool table and example dialog. Local end-to-end testing: [`demos/medical`](demos/medical/README.md).

</details>

<details>
<summary><strong>Configuration</strong> — grids and ECT settings</summary>

### Imputation grids

Named grids via `GridGallery`:

| Name          | Methods                                                     |
| ------------- | ----------------------------------------------------------- |
| `default`     | BayesianRidge, DecisionTree, RandomForest, GradientBoosting |
| `sampling`    | DistributionImputer (empirical sampling)                    |
| `finance`     | IterativeImputer, KNNImputer, SimpleImputer                 |
| `healthcare`  | KNNImputer, SimpleImputer, IterativeImputer                 |
| `marketing`   | SimpleImputer, KNNImputer, IterativeImputer                 |
| `engineering` | SimpleImputer, KNNImputer, IterativeImputer                 |

Custom grid:

```python
from phil import Phil, ImputationConfig
from sklearn.model_selection import ParameterGrid

config = ImputationConfig(
    methods=["KNNImputer"],
    modules=["sklearn.impute"],
    grids=[ParameterGrid({"n_neighbors": [3, 5, 7]})],
)
phil = Phil(param_grid=config)
```

### ECT descriptor

```python
from phil import Phil, ECTConfig

phil = Phil(config=ECTConfig(num_thetas=64, radius=1.0, resolution=100, scale=500, normalize=True, seed=42))
```

</details>

<details>
<summary><strong>scikit-learn pipelines</strong></summary>

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from phil import PhilTransformer

pipe = Pipeline([
    ("imputer", PhilTransformer(samples=20, random_state=0)),
    ("model", RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
```

</details>

<details>
<summary><strong>What's new in v1.1.0</strong></summary>

- **FastMCP server** — `phil-mcp` exposes the imputation sweep pipeline as MCP tools for agents.
- **Grid recommender** — `recommend_grid`, declarative `GridMetadata`, and the `phil://docs/imputation-matrix` resource.
- **Medical demo** — [`demos/medical`](demos/medical/README.md) with covariate sampling, masked iterative imputation, and MDS visualization of descriptor space.

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

</details>

<details>
<summary><strong>Development</strong></summary>

```bash
uv sync --all-extras
uv run pytest -v
uvx ruff format phil/ tests/
uvx ruff check phil/ tests/
```

Contributors: see [AGENTS.md](AGENTS.md) for package layout and design notes.

</details>

<details>
<summary><strong>Documentation</strong></summary>

Sphinx docs live under `docs/source`. Build locally:

```bash
uv run sphinx-build -M html docs/source docs/build
```

</details>
