# Phil

`Phil` is a representation-guided imputation library for missing tabular data.

It generates multiple imputations using a configurable strategy grid, computes
Euler Characteristic Transform (ECT) descriptors over each imputed dataset, and
selects the most representative imputation from the candidate set.

## Installation

```bash
pip install phil
```

`phil` requires the `trailed` backend for ECT computation. Install it from the
KRV research index or provide a compatible local build.

## What Phil Does

1. **Impute** — runs a grid of imputation strategies (sklearn estimators or custom) over the input dataframe, producing a set of candidate datasets
2. **Describe** — computes an ECT descriptor for each candidate via the `trailed` backend
3. **Select** — picks the candidate closest to the mean descriptor (most representative imputation)
4. **Transform** — exposes the fitted pipeline for inference on new data

## Quick Start

```python
import pandas as pd
from phil import Phil

df = pd.read_csv("data_with_missing.csv")

phil = Phil(samples=30, random_state=42)
imputed_df = phil.fit(df)

# Apply the same fitted pipeline to new data
new_df = phil.transform(new_data)
```

### scikit-learn Pipeline Integration

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

## Configuration

### Imputation grids

`Phil` ships with named grids accessible via `GridGallery`:

| Name | Methods |
|------|---------|
| `default` | BayesianRidge, DecisionTree, RandomForest, GradientBoosting |
| `sampling` | DistributionImputer (empirical sampling) |
| `finance` | IterativeImputer, KNNImputer, SimpleImputer |
| `healthcare` | KNNImputer, SimpleImputer, IterativeImputer |
| `marketing` | SimpleImputer, KNNImputer, IterativeImputer |
| `engineering` | SimpleImputer, KNNImputer, IterativeImputer |

Pass a grid name or an `ImputationConfig` directly:

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

ECT is configured via `ECTConfig`:

```python
from phil import Phil, ECTConfig

ect_config = ECTConfig(
    num_thetas=64,
    radius=1.0,
    resolution=100,
    scale=500,
    normalize=True,
    seed=42,
)
phil = Phil(config=ect_config)
```

## Development

```bash
uv sync --all-extras
uv run pytest -v
uv run black phil/ tests/
```
