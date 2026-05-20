from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def introduce_mcar(df: pd.DataFrame, rate: float, seed: int = 42) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    df_masked = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        observed = df[col].notna()
        observed_idx = np.where(observed)[0]
        if len(observed_idx) == 0:
            continue
        n_mask = max(1, int(len(observed_idx) * rate))
        chosen = rs.choice(observed_idx, size=n_mask, replace=False)
        df_masked.iloc[chosen, df.columns.get_loc(col)] = np.nan

    return df_masked


def write_toy_dataset(output_dir: Path) -> Path:
    toy = pd.DataFrame(
        {
            "age": [63, 54, np.nan, 48, 59, 67, 45, np.nan, 71, 52],
            "resting_bp": [145, np.nan, 130, 120, 138, np.nan, 128, 118, 150, 132],
            "cholesterol": [233, 250, np.nan, 204, 236, 280, np.nan, 190, 300, 221],
            "sex": ["M", "F", "M", "F", "M", "M", "F", "F", None, "M"],
            "pain_type": [
                "typical",
                "asymptomatic",
                None,
                "non-anginal",
                "typical",
                None,
                "atypical",
                "typical",
                "asymptomatic",
                "non-anginal",
            ],
            "risk_group": [
                "high",
                "high",
                "medium",
                "low",
                "high",
                "high",
                "low",
                "low",
                "high",
                None,
            ],
        }
    )
    target = output_dir / "toy_medical_missing.csv"
    toy.to_csv(target, index=False)
    return target


def prepare_pima(output_dir: Path, mask_rate: float, seed: int) -> list[Path]:
    from sklearn.datasets import fetch_openml

    pima = fetch_openml(
        name="diabetes", version=1, as_frame=True, parser="auto"
    ).data.copy()

    # In OpenML's 'diabetes' dataset, the columns are abbreviated
    zero_impossible_cols = ["plas", "pres", "skin", "insu", "mass"]
    existing = [c for c in zero_impossible_cols if c in pima.columns]

    # Replace 0 with NaN for these columns, handling string conversions if necessary
    pima[existing] = pima[existing].replace(0, np.nan)
    pima[existing] = pima[existing].replace("0", np.nan)

    # Ensure all columns are numeric
    pima = pima.astype(float)

    complete = pima.dropna().reset_index(drop=True)
    masked = introduce_mcar(complete, rate=mask_rate, seed=seed)

    out_complete = output_dir / "pima_complete.csv"
    out_masked = output_dir / "pima_mcar_15.csv"
    complete.to_csv(out_complete, index=False)
    masked.to_csv(out_masked, index=False)
    return [out_complete, out_masked]


def prepare_heart(output_dir: Path, mask_rate: float, seed: int) -> list[Path]:
    from ucimlrepo import fetch_ucirepo

    heart = fetch_ucirepo(id=45).data.features.copy()
    complete_numeric = (
        heart.dropna().select_dtypes(include="number").reset_index(drop=True)
    )
    masked = introduce_mcar(complete_numeric, rate=mask_rate, seed=seed)

    out_complete = output_dir / "heart_complete_numeric.csv"
    out_masked = output_dir / "heart_mcar_15.csv"
    complete_numeric.to_csv(out_complete, index=False)
    masked.to_csv(out_masked, index=False)
    return [out_complete, out_masked]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local medical CSVs for Phil MCP demos."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demos/medical/data"),
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--mask-rate",
        type=float,
        default=0.15,
        help="MCAR masking rate for generated missingness datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for masking.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = [write_toy_dataset(output_dir)]

    try:
        generated.extend(
            prepare_pima(output_dir, mask_rate=args.mask_rate, seed=args.seed)
        )
    except Exception as exc:
        print(f"[warn] Could not fetch/build PIMA dataset: {exc}")

    try:
        generated.extend(
            prepare_heart(output_dir, mask_rate=args.mask_rate, seed=args.seed)
        )
    except Exception as exc:
        print(f"[warn] Could not fetch/build Heart dataset: {exc}")

    print("Generated demo datasets:")
    for path in generated:
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()
