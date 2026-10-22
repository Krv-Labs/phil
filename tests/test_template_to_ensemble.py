"""Tests for translating populated domain_knowledge templates into Phil ensembles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from phil import (
    CovarianceMatrix,
    CovariateDistributionImputer,
    CovariateSubset,
    DomainKnowledge,
    ImputationConfig,
    MaskedIterativeImputer,
    Phil,
    VariableBounds,
)
from phil.mcp.config import config_to_phil_kwargs, validate_config_yaml


def _make_domain_df(n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(25.0, 80.0, size=n)
    # Strictly positive lab value strongly determined by age
    egfr = np.clip(120.0 - 0.8 * age + rng.normal(0.0, 3.0, size=n), 15.0, 130.0)
    lactate = rng.uniform(0.5, 4.0, size=n)
    df = pd.DataFrame({"age": age, "egfr": egfr, "lactate": lactate})
    df.loc[df.index[:10], "egfr"] = np.nan
    df.loc[df.index[10:18], "lactate"] = np.nan
    return df


def test_config_to_phil_kwargs_materializes_domain_knowledge_for_builtin_grid() -> None:
    config_yaml = """run:
  name: dk_test
imputation:
  grid: healthcare
  samples: 4
  domain_knowledge:
    covariate_subsets:
      egfr:
        predictors: [age]
        estimators: [BayesianRidge, Ridge]
        sample_posterior: true
        citations: ["Levey et al. 2009 CKD-EPI"]
    variable_bounds:
      egfr:
        min_value: 15.0
        max_value: 130.0
        citations: ["KDIGO 2012"]
    covariance_matrix:
      matrix:
        - [1.0, -0.7, 0.1]
        - [-0.7, 1.0, -0.1]
        - [0.1, -0.1, 1.0]
      citations: ["Population prior covariance"]
magic:
  method: ECT
  collapse_strategy: barycenter
"""
    report = validate_config_yaml(config_yaml)
    assert report.ok, report.issues

    kwargs = config_to_phil_kwargs(config_yaml)
    pg = kwargs["param_grid"]
    assert isinstance(pg, ImputationConfig)
    assert pg.domain_knowledge is not None
    assert "egfr" in pg.domain_knowledge.covariate_subsets
    subset = pg.domain_knowledge.covariate_subsets["egfr"]
    assert subset.predictors == ["age"]
    assert subset.estimators == ["BayesianRidge", "Ridge"]
    assert subset.sample_posterior is True
    assert "egfr" in pg.domain_knowledge.variable_bounds
    assert pg.domain_knowledge.variable_bounds["egfr"].min_value == 15.0
    assert pg.domain_knowledge.covariance_matrix is not None


def test_phil_promotes_iterative_imputer_and_injects_covariance_imputer() -> None:
    df = _make_domain_df()
    dk = DomainKnowledge(
        covariate_subsets={
            "egfr": CovariateSubset(
                predictors=["age"],
                citations=["CKD-EPI 2009"],
                estimators=["BayesianRidge"],
                sample_posterior=True,
            )
        },
        variable_bounds={
            "egfr": VariableBounds(
                min_value=20.0,
                max_value=115.0,
                citations=["Clinical physiological bounds"],
            )
        },
        covariance_matrix=CovarianceMatrix(
            matrix=[
                [1.0, -0.6, 0.0],
                [-0.6, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            citations=["Prior covariance"],
        ),
    )
    base_cfg = ImputationConfig(
        methods=["IterativeImputer"],
        modules=["sklearn.impute"],
        grids=[ParameterGrid({"max_iter": [5]})],
        domain_knowledge=dk,
    )
    phil = Phil(samples=5, param_grid=base_cfg, random_state=42)
    imputed_df = phil.fit(df, max_iter=5)

    # Verify MaskedIterativeImputer and CovariateDistributionImputer were constructed
    imputer_types = {
        type(pipe.named_steps["imputer"]) for pipe in phil.selected_imputers
    }
    estimator_types = {
        type(getattr(pipe.named_steps["imputer"], "estimator", None))
        for pipe in phil.selected_imputers
    }
    assert MaskedIterativeImputer in imputer_types
    assert CovariateDistributionImputer in estimator_types

    # Verify inverse-transformation back to original clinical units and bounds enforcement
    assert list(imputed_df.columns) == ["age", "egfr", "lactate"]
    assert imputed_df.isna().sum().sum() == 0
    assert float(imputed_df["egfr"].min()) >= 20.0 - 1e-6
    assert float(imputed_df["egfr"].max()) <= 115.0 + 1e-6
    # Verify observed age values are preserved in original units (~25..80), not Z-scores
    np.testing.assert_allclose(
        imputed_df["age"].to_numpy(), df["age"].to_numpy(), rtol=1e-5
    )
