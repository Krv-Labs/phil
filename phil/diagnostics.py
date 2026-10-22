"""
Empirical predictability profiling and guided research worksheet generation for Phil.

Contains zero hardcoded domain facts. Instead, this module empirically profiles
missingness patterns, linear vs. nonlinear predictability, and observed feature
distributions, then emits a structured Research Worksheet & YAML Config Template
for frontier models to populate from scientific literature.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_TOKEN_RE = re.compile(r"^tok_[0-9a-fA-F]{8,}$")


def _detect_identifier_or_token_columns(df: pd.DataFrame) -> list[str]:
    """Identify high-cardinality string/token identifier columns (e.g. Phirewall tok_... vaults)."""
    id_cols = []
    n_rows = len(df)
    if n_rows == 0:
        return id_cols
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            continue
        obs = series.dropna()
        if len(obs) == 0:
            continue
        sample_strs = [str(v) for v in obs.head(20)]
        if any(_TOKEN_RE.match(s) for s in sample_strs):
            id_cols.append(str(col))
            continue
        # High-cardinality string column where every observed row is unique (for n >= 5)
        if len(obs) >= 5 and int(obs.nunique()) == len(obs):
            id_cols.append(str(col))
    return id_cols


def _encode_feature_matrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Convert a subset of DataFrame columns to a simple imputed numeric matrix for probing."""
    valid_cols = []
    arrays = []

    for col in feature_cols:
        series = df[col]
        if series.isna().all():
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            vals = series.to_numpy(dtype=float, copy=True)
            med = float(np.nanmedian(vals)) if not np.all(np.isnan(vals)) else 0.0
            vals[np.isnan(vals)] = med
        else:
            cat = pd.Categorical(series)
            codes = (
                pd.Series(cat.codes, index=df.index)
                .replace(-1, np.nan)
                .to_numpy(dtype=float, copy=True)
            )
            mode_val = (
                float(np.nanmedian(codes)) if not np.all(np.isnan(codes)) else 0.0
            )
            codes[np.isnan(codes)] = mode_val
            vals = codes
        if np.nanstd(vals) == 0.0:
            # Keep constant columns out of correlation / MI probes if others exist
            pass
        valid_cols.append(col)
        arrays.append(vals)

    if not arrays:
        return np.empty((len(df), 0), dtype=float), []
    return np.column_stack(arrays), valid_cols


def _summarize_observed_distribution(series: pd.Series) -> dict[str, Any]:
    """Summarize the empirical distribution of observed values in a column."""
    obs = series.dropna()
    n_obs = len(obs)
    if n_obs == 0:
        return {
            "n_observed": 0,
            "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "distribution_family_hint": "unknown",
        }

    is_num = bool(
        pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    )
    if not is_num:
        vc = obs.value_counts(normalize=True)
        return {
            "n_observed": n_obs,
            "is_numeric": False,
            "n_unique": int(obs.nunique()),
            "top_categories": {
                str(k): round(float(v), 4) for k, v in vc.head(5).items()
            },
            "distribution_family_hint": "categorical",
        }

    arr = obs.to_numpy(dtype=float)
    min_val = float(np.min(arr))
    p01 = float(np.percentile(arr, 1))
    p25 = float(np.percentile(arr, 25))
    median = float(np.median(arr))
    p75 = float(np.percentile(arr, 75))
    p99 = float(np.percentile(arr, 99))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skewness = float(stats.skew(arr)) if n_obs >= 3 and std_val > 0 else 0.0
        kurtosis = float(stats.kurtosis(arr)) if n_obs >= 4 and std_val > 0 else 0.0

    if not np.isfinite(skewness):
        skewness = 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 0.0

    # Empirical parametric family hint
    if min_val >= 0.0 and max_val <= 1.0 and n_obs >= 5:
        family_hint = "bounded"
    elif min_val > 0 and skewness > 1.0:
        family_hint = "lognormal"
    elif abs(skewness) > 1.0:
        family_hint = "skewed"
    else:
        family_hint = "normal"

    return {
        "n_observed": n_obs,
        "is_numeric": True,
        "min": round(min_val, 6),
        "p01": round(p01, 6),
        "p25": round(p25, 6),
        "median": round(median, 6),
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "p75": round(p75, 6),
        "p99": round(p99, 6),
        "max": round(max_val, 6),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "distribution_family_hint": family_hint,
    }


def _probe_missingness_mechanism(
    df: pd.DataFrame,
    target_col: str,
    other_cols: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    """Probe whether the missingness indicator M_j is predictable from observed columns (MAR vs MCAR)."""
    mask = df[target_col].isna().to_numpy(dtype=int)
    n_missing = int(mask.sum())
    n_total = len(df)
    if n_missing < 3 or (n_total - n_missing) < 3 or not other_cols:
        return {
            "mar_predictability_auc": 0.5,
            "mechanism_hint": "insufficient_missing_or_complete_rows",
            "missingness_drivers": [],
        }

    X_other, valid_cols = _encode_feature_matrix(df, other_cols)
    if X_other.shape[1] == 0:
        return {
            "mar_predictability_auc": 0.5,
            "mechanism_hint": "no_usable_predictors",
            "missingness_drivers": [],
        }

    n_splits = min(3, n_missing, n_total - n_missing)
    auc_score = 0.5
    importances = np.zeros(X_other.shape[1], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = ExtraTreesClassifier(
            n_estimators=25, max_depth=5, random_state=random_state
        )
        if n_splits >= 2:
            try:
                cv = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                )
                scores = cross_val_score(clf, X_other, mask, cv=cv, scoring="roc_auc")
                if len(scores) and np.all(np.isfinite(scores)):
                    auc_score = float(np.mean(scores))
            except (ValueError, RuntimeError):
                auc_score = 0.5
        try:
            clf.fit(X_other, mask)
            importances = np.asarray(clf.feature_importances_, dtype=float)
        except (ValueError, RuntimeError):
            importances = np.zeros(X_other.shape[1], dtype=float)

    driver_order = np.argsort(importances)[::-1]
    drivers = []
    for idx in driver_order[:5]:
        imp = float(importances[idx])
        if imp <= 0.0:
            continue
        drivers.append(
            {
                "column": valid_cols[int(idx)],
                "importance": round(imp, 4),
            }
        )

    if auc_score >= 0.65:
        mech_hint = "MAR_likely (missingness strongly predicted by observed variables)"
    elif auc_score >= 0.56:
        mech_hint = "MAR_weak (moderate missingness dependence on observed variables)"
    else:
        mech_hint = (
            "MCAR_or_unobserved_MNAR (missingness not explained by observed features)"
        )

    return {
        "mar_predictability_auc": round(auc_score, 4),
        "mechanism_hint": mech_hint,
        "missingness_drivers": drivers,
    }


def _rank_predictors(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    is_numeric: bool,
    random_state: int,
) -> list[dict[str, Any]]:
    """Rank predictors using mutual information and tree importance."""
    mutual_information = mutual_info_regression if is_numeric else mutual_info_classif
    model_type = ExtraTreesRegressor if is_numeric else ExtraTreesClassifier
    try:
        mi = mutual_information(
            X, y, n_neighbors=min(3, max(1, len(y) - 1)), random_state=random_state
        )
    except (ValueError, RuntimeError):
        mi = np.zeros(X.shape[1], dtype=float)
    try:
        model = model_type(n_estimators=30, max_depth=6, random_state=random_state)
        model.fit(X, y)
        tree_importance = np.asarray(model.feature_importances_, dtype=float)
    except (ValueError, RuntimeError):
        tree_importance = np.zeros(X.shape[1], dtype=float)

    mi_normalized = mi / (np.max(mi) if np.max(mi) > 0 else 1.0)
    combined = 0.5 * mi_normalized + 0.5 * tree_importance
    predictors = []
    for idx in np.argsort(combined)[::-1][:5]:
        predictor = {
            "column": columns[int(idx)],
            "combined_importance": round(float(combined[idx]), 4),
            "mutual_information": round(float(mi[idx]), 4),
            "tree_importance": round(float(tree_importance[idx]), 4),
        }
        if is_numeric:
            x = X[:, idx]
            pearson = spearman = 0.0
            if np.std(x) > 0 and np.std(y) > 0:
                pearson = float(np.corrcoef(x, y)[0, 1])
                spearman = float(stats.spearmanr(x, y).statistic)
            predictor["pearson_r"] = round(pearson if np.isfinite(pearson) else 0.0, 4)
            predictor["spearman_r"] = round(
                spearman if np.isfinite(spearman) else 0.0, 4
            )
        predictors.append(predictor)
    return predictors


def profile_predictability(
    df: pd.DataFrame,
    target_columns: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Empirically profile predictability, predictor rankings, MAR missingness
    drivers, and observed distributions for incomplete columns in ``df``.

    Contains zero hardcoded domain facts.
    """
    n_rows, n_cols = df.shape
    all_cols = [str(c) for c in df.columns]
    identifier_cols = _detect_identifier_or_token_columns(df)

    if target_columns is None:
        targets = [
            str(c)
            for c in df.columns
            if df[c].isna().any() and str(c) not in identifier_cols
        ]
    else:
        targets = [str(c) for c in target_columns if c in df.columns]

    feature_profiles = {}

    for col in targets:
        series = df[col]
        n_missing = int(series.isna().sum())
        missing_pct = round(n_missing / n_rows * 100.0, 3) if n_rows else 0.0
        other_cols = [c for c in all_cols if c != col and c not in identifier_cols]

        dist_summary = _summarize_observed_distribution(series)
        mar_diag = _probe_missingness_mechanism(
            df, col, other_cols, random_state=random_state
        )

        obs_mask = series.notna().to_numpy()
        n_obs = int(obs_mask.sum())
        is_numeric = dist_summary.get("is_numeric", False)

        X_other_full, valid_cols = _encode_feature_matrix(df, other_cols)
        X_obs = X_other_full[obs_mask] if len(valid_cols) else np.empty((n_obs, 0))

        linear_score = 0.0
        nonlinear_score = 0.0
        metric_name = "r2" if is_numeric else "balanced_accuracy"
        top_predictors = []

        if n_obs >= 6 and X_obs.shape[1] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if is_numeric:
                    y_obs = series.loc[obs_mask].to_numpy(dtype=float)
                    if np.nanstd(y_obs) > 0:
                        n_splits = min(3, max(2, n_obs // 3))
                        cv = KFold(
                            n_splits=n_splits, shuffle=True, random_state=random_state
                        )
                        lin_pipe = Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                                ("ridge", Ridge(alpha=1.0, random_state=random_state)),
                            ]
                        )
                        nonlin_model = ExtraTreesRegressor(
                            n_estimators=30,
                            max_depth=6,
                            random_state=random_state,
                        )
                        try:
                            l_scores = cross_val_score(
                                lin_pipe, X_obs, y_obs, cv=cv, scoring="r2"
                            )
                            linear_score = float(np.nanmean(l_scores))
                        except (ValueError, RuntimeError):
                            linear_score = 0.0
                        try:
                            nl_scores = cross_val_score(
                                nonlin_model, X_obs, y_obs, cv=cv, scoring="r2"
                            )
                            nonlinear_score = float(np.nanmean(nl_scores))
                        except (ValueError, RuntimeError):
                            nonlinear_score = 0.0

                        top_predictors = _rank_predictors(
                            X_obs, y_obs, valid_cols, True, random_state
                        )
                else:
                    cat = pd.Categorical(series.loc[obs_mask])
                    y_obs = np.asarray(cat.codes, dtype=int)
                    unique_classes, counts = np.unique(y_obs, return_counts=True)
                    if len(unique_classes) >= 2 and np.min(counts) >= 2:
                        n_splits = min(3, int(np.min(counts)))
                        cv = StratifiedKFold(
                            n_splits=n_splits, shuffle=True, random_state=random_state
                        )
                        lin_pipe = Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                                (
                                    "logreg",
                                    LogisticRegression(
                                        max_iter=200, random_state=random_state
                                    ),
                                ),
                            ]
                        )
                        nonlin_model = ExtraTreesClassifier(
                            n_estimators=30,
                            max_depth=6,
                            random_state=random_state,
                        )
                        try:
                            l_scores = cross_val_score(
                                lin_pipe,
                                X_obs,
                                y_obs,
                                cv=cv,
                                scoring="balanced_accuracy",
                            )
                            linear_score = float(np.nanmean(l_scores))
                        except (ValueError, RuntimeError):
                            linear_score = 0.0
                        try:
                            nl_scores = cross_val_score(
                                nonlin_model,
                                X_obs,
                                y_obs,
                                cv=cv,
                                scoring="balanced_accuracy",
                            )
                            nonlinear_score = float(np.nanmean(nl_scores))
                        except (ValueError, RuntimeError):
                            nonlinear_score = 0.0

                    if len(unique_classes) >= 2:
                        top_predictors = _rank_predictors(
                            X_obs, y_obs, valid_cols, False, random_state
                        )

        if not np.isfinite(linear_score):
            linear_score = 0.0
        if not np.isfinite(nonlinear_score):
            nonlinear_score = 0.0

        best_score = max(linear_score, nonlinear_score)
        nonlinear_gain = max(0.0, nonlinear_score - max(0.0, linear_score))

        if best_score >= 0.6:
            tier = "high"
        elif best_score >= 0.25:
            tier = "moderate"
        else:
            tier = "low"

        # Recommend candidate estimators empirically
        if tier == "low":
            recommended_estimators = ["BayesianRidge"]
            recommend_sample_posterior = True
            modeling_guidance = (
                "Low conditional predictability (< 0.25): deterministic regression "
                "imputers will collapse variance toward the mean. Enable "
                "sample_posterior=True or include CovariateDistributionImputer / "
                "DistributionImputer."
            )
        elif nonlinear_gain >= 0.10:
            recommended_estimators = [
                "ExtraTreesRegressor",
                "RandomForestRegressor",
                "HistGradientBoostingRegressor",
            ]
            recommend_sample_posterior = False
            modeling_guidance = (
                "Nonlinear/tree models outperform linear Ridge by >= 0.10: "
                "prefer tree ensembles (ExtraTrees, RandomForest, HistGradientBoosting)."
            )
        else:
            recommended_estimators = ["BayesianRidge", "ExtraTreesRegressor"]
            recommend_sample_posterior = tier != "high"
            modeling_guidance = (
                "Linear relationship dominates or matches tree models: "
                "BayesianRidge + ExtraTreesRegressor provide balanced coverage."
            )

        feature_profiles[col] = {
            "column": col,
            "dtype": str(series.dtype),
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "metric": metric_name,
            "linear_score": round(linear_score, 4),
            "linear_cv_score": round(linear_score, 4),
            "nonlinear_score": round(nonlinear_score, 4),
            "nonlinear_cv_score": round(nonlinear_score, 4),
            "best_predictability_score": round(best_score, 4),
            "nonlinear_gain": round(nonlinear_gain, 4),
            "predictability_tier": tier,
            "recommended_estimators": recommended_estimators,
            "recommend_sample_posterior": recommend_sample_posterior,
            "modeling_guidance": modeling_guidance,
            "top_empirical_predictors": top_predictors,
            "top_predictors": [p["column"] for p in top_predictors],
            "missingness_mechanism": mar_diag,
            "observed_distribution": dist_summary,
            "distribution_summary": dist_summary,
        }

    # Global empirical correlation/covariance summary on numeric columns
    numeric_cols = [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    X_num, valid_num_cols = _encode_feature_matrix(df, numeric_cols)
    if len(valid_num_cols) >= 2 and len(df) >= 3:
        Z = StandardScaler().fit_transform(X_num)
        emp_cov = np.cov(Z, rowvar=False)
        emp_cov = np.nan_to_num(emp_cov, nan=0.0)
        # Ensure positive-definite diagonal
        emp_cov = emp_cov + np.eye(len(valid_num_cols)) * 1e-3
        covariance_preview = {
            "variables": valid_num_cols,
            "standardized_covariance_matrix": [
                [round(float(val), 4) for val in row] for row in emp_cov
            ],
        }
    else:
        covariance_preview = {
            "variables": valid_num_cols,
            "standardized_covariance_matrix": [],
        }

    return {
        "status": "ok",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "identifier_or_token_columns": identifier_cols,
        "recommended_drop_cols": identifier_cols,
        "incomplete_columns": targets,
        "feature_profiles": feature_profiles,
        "target_profiles": feature_profiles,
        "empirical_standardized_covariance": covariance_preview,
    }


def build_research_worksheet(
    df_or_profile: pd.DataFrame | dict[str, Any],
    profile: dict[str, Any] | None = None,
    *,
    grid: str = "default",
    run_name: str = "phil_sweep",
    data_path: str = "",
    domain_context: str = "",
    study_context: str = "",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Build a structured Research Worksheet & YAML Config Template from empirical
    predictability profiles.

    Zero domain knowledge is hardcoded in Phil. Instead, this returns:
      1. ``feature_research_cards``: Per-column empirical findings paired with
         structured literature questions for the frontier agent to investigate.
      2. ``config_template_yaml``: A Phil YAML scaffold with pre-structured
         ``domain_knowledge`` slots (``covariate_subsets``, ``variable_bounds``,
         ``covariance_matrix``, and required ``citations`` placeholders) ready
         for the agent to populate after literature review.
    """
    effective_context = domain_context or study_context
    if isinstance(df_or_profile, pd.DataFrame):
        if profile is None:
            profile = profile_predictability(df_or_profile, random_state=random_state)
    elif isinstance(df_or_profile, dict):
        profile = df_or_profile
    else:
        raise TypeError("df_or_profile must be a pandas DataFrame or profile dict.")

    feature_profiles = profile.get("feature_profiles", {})
    research_cards = []
    covariate_subsets_template = {}
    variable_bounds_template = {}

    for col, fp in feature_profiles.items():
        top_preds = [p["column"] for p in fp.get("top_empirical_predictors", [])[:4]]
        mar_drivers = [
            d["column"]
            for d in fp.get("missingness_mechanism", {}).get("missingness_drivers", [])[
                :2
            ]
        ]
        merged_predictors = []
        for candidate in top_preds + mar_drivers:
            if candidate not in merged_predictors and candidate != col:
                merged_predictors.append(candidate)

        obs_dist = fp.get("observed_distribution", {})
        is_numeric = bool(obs_dist.get("is_numeric", False))

        questions = [
            (
                f"CAUSAL / MECHANISTIC PREDICTORS for '{col}': Empirical top correlates "
                f"are {top_preds} and MAR missingness drivers are {mar_drivers}. "
                f"According to domain literature, which of these are valid mechanistic "
                f"or clinical predictors of '{col}', and which should be excluded to "
                f"prevent collider bias or target leakage?"
            ),
            (
                f"FUNCTIONAL FORM & ESTIMATOR CHOICE for '{col}': Empirical linear "
                f"score={fp['linear_score']} vs nonlinear score={fp['nonlinear_score']} "
                f"(tier='{fp['predictability_tier']}'). Does literature support a linear "
                f"additive model or nonlinear/interaction effects for '{col}'?"
            ),
        ]
        if is_numeric:
            questions.append(
                f"PLAUSIBLE BOUNDS & DISTRIBUTION FAMILY for '{col}': Observed range is "
                f"[{obs_dist.get('min')}, {obs_dist.get('max')}] with skewness={obs_dist.get('skewness')} "
                f"(hint='{obs_dist.get('distribution_family_hint')}'). What are the "
                f"established physical/physiological/domain bounds [min_value, max_value] "
                f"and distribution family for '{col}' in the literature?"
            )

        research_cards.append(
            {
                "column": col,
                "missing_pct": fp["missing_pct"],
                "predictability_tier": fp["predictability_tier"],
                "linear_score": fp["linear_score"],
                "nonlinear_score": fp["nonlinear_score"],
                "empirical_top_predictors": top_preds,
                "mar_missingness_drivers": mar_drivers,
                "observed_distribution": obs_dist,
                "recommended_estimators": fp["recommended_estimators"],
                "recommend_sample_posterior": fp["recommend_sample_posterior"],
                "research_questions_to_answer": questions,
                "template_slot_paths": {
                    "covariate_subset": f"imputation.domain_knowledge.covariate_subsets.{col}",
                    "variable_bounds": (
                        f"imputation.domain_knowledge.variable_bounds.{col}"
                        if is_numeric
                        else None
                    ),
                },
            }
        )

        covariate_subsets_template[col] = {
            "predictors": merged_predictors,
            "estimators": fp["recommended_estimators"],
            "sample_posterior": fp["recommend_sample_posterior"],
            "citations": [
                (
                    "TODO: Replace with peer-reviewed literature citation(s) justifying predictors for "
                    f"'{col}' (e.g. 'Author et al. (Year), Journal / DOI')"
                )
            ],
        }

        if is_numeric:
            variable_bounds_template[col] = {
                "min_value": obs_dist.get("min"),
                "max_value": obs_dist.get("max"),
                "distribution_family": obs_dist.get(
                    "distribution_family_hint", "normal"
                ),
                "citations": [
                    (
                        "TODO: Replace with literature citation(s) justifying plausible range/bounds for "
                        f"'{col}'"
                    )
                ],
            }

    try:
        from phil.mcp.config import GRID_INTENTS, default_config_dict

        cfg = default_config_dict(run_name=run_name, data=data_path)
        valid_grids = set(GRID_INTENTS.keys())
    except ImportError:
        cfg = {
            "run": {"name": run_name, "data": data_path},
            "imputation": {
                "grid": "default",
                "samples": 30,
                "random_state": 42,
                "max_iter": 5,
                "drop_cols": [],
                "missingness_thresh": None,
                "encode_categoricals": True,
            },
            "magic": {
                "method": "ECT",
                "num_thetas": 64,
                "radius": 1.0,
                "resolution": 100,
                "scale": 500,
                "normalize": True,
                "seed": 42,
            },
            "output": {"write_csv": ""},
        }
        valid_grids = {
            "default",
            "sampling",
            "finance",
            "healthcare",
            "marketing",
            "engineering",
        }

    cfg["imputation"]["grid"] = (
        grid if grid in valid_grids or grid == "custom" else "default"
    )
    recommended_drops = list(profile.get("recommended_drop_cols", []) or [])
    if recommended_drops:
        cfg["imputation"]["drop_cols"] = recommended_drops
    cfg["imputation"]["domain_knowledge"] = {
        "covariate_subsets": covariate_subsets_template,
        "variable_bounds": variable_bounds_template,
    }
    cfg["magic"]["collapse_strategy"] = "auto"

    try:
        import yaml

        config_template_yaml = yaml.safe_dump(cfg, sort_keys=False)
    except ImportError:
        import json

        config_template_yaml = json.dumps(cfg, indent=2)

    return {
        "status": "ok",
        "domain_context": effective_context,
        "study_context": effective_context,
        "grid": cfg["imputation"]["grid"],
        "agent_instructions": [
            "1. Review `feature_research_cards` for each incomplete column.",
            "2. Search scientific/domain literature (or consult domain guidelines) to answer `research_questions_to_answer`.",
            "3. Refine `predictors`, `estimators`, `sample_posterior`, and `min_value`/`max_value` in `config_template_yaml`.",
            "4. Replace every 'TODO: ...' entry in `citations` with real literature citations.",
            "5. Optionally add `imputation.domain_knowledge.covariance_matrix` if literature or empirical prior covariance is desired for Mahalanobis `CovariateDistributionImputer`.",
            "6. Call `validate_config` and `run_imputation_sweep`, then use `inspect_ect_distribution` and `collapse_ensemble` to navigate the ECT signature distribution.",
        ],
        "feature_research_cards": research_cards,
        "research_questions": research_cards,
        "predictability_profile": profile,
        "config_template_yaml": config_template_yaml,
        "config_yaml_template": config_template_yaml,
    }
