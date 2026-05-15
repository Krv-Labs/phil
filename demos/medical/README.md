# Medical MCP Demo (Local)

Use this demo to test Phil's MCP server locally with medical datasets that have missing values.

## 1. Install dependencies

```bash
uv sync --group mcp
uv pip install ucimlrepo
```

## 2. Generate local CSVs

```bash
uv run python demos/medical/prepare_medical_datasets.py --output-dir demos/medical/data
```

This produces:

- `pima_complete.csv`
- `pima_mcar_15.csv`
- `heart_complete_numeric.csv`
- `heart_mcar_15.csv`
- `toy_medical_missing.csv` (small built-in fallback dataset for quick testing)

## 3. Start MCP server

```bash
uv run phil-mcp
```

## 4. Test in your MCP client

Use a prompt like:

```text
I have a dataset with missing values at /ABS/PATH/TO/demos/medical/data/pima_mcar_15.csv.
Use Phil to run an imputation sweep, diagnose spread, and export the selected imputed data to
/ABS/PATH/TO/demos/medical/data/pima_imputed.csv.
```

## 5. Optional high-missingness stress test

For tougher local recovery/error-handling checks, use:

- `toy_medical_missing.csv` (mixed numeric/string + sparse columns)

Then ask the agent to:

1. characterize missingness
2. set `imputation.drop_cols` and/or `imputation.missingness_thresh`
3. run and export the sweep
