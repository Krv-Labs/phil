"""Regression coverage for the agentic ensemble PR review."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from phil import (
    CovarianceMatrix,
    CovariateSubset,
    DomainKnowledge,
    ImputationConfig,
    Phil,
    VariableBounds,
)
from phil.imputation.covariate_distribution import CovariateDistributionImputer
from phil.magic.collapse import COLLAPSE_STRATEGIES, collapse_ect_ensemble
from phil.mcp.config import validate_config_yaml


def test_fit_transform_and_collapse_preserve_observed_values_outside_bounds():
    original = pd.DataFrame(
        {"x": [10.0, 20.0, np.nan, 40.0, 50.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=[10, 20, 30, 40, 50],
    )
    config = ImputationConfig(
        methods=["BayesianRidge"],
        modules=["sklearn.linear_model"],
        grids=[ParameterGrid({})],
        domain_knowledge=DomainKnowledge(
            variable_bounds={"x": VariableBounds(max_value=30.0)}
        ),
    )
    model = Phil(param_grid=config, random_state=42)
    fitted = model.fit(original)
    for result in [
        fitted,
        model.transform(original),
        model.collapse("barycenter").collapsed_df,
    ]:
        observed = original["x"].notna()
        pd.testing.assert_frame_equal(result.loc[observed], original.loc[observed])
        assert result.loc[30, "x"] <= 30.0


@pytest.mark.parametrize("strategy", list(COLLAPSE_STRATEGIES))
def test_collapse_bounds_only_apply_to_missing_cells(strategy):
    original = pd.DataFrame({"x": [100.0, np.nan]}, index=[5, 8])
    candidates = [
        pd.DataFrame({"x": [100.0, fill]}, index=[5, 8]) for fill in [40.0, 50.0]
    ]
    result = collapse_ect_ensemble(
        candidates,
        [np.array([0.0]), np.array([1.0])],
        strategy=strategy,
        original_df=original,
        variable_bounds={"x": {"max_value": 30.0}},
    )
    assert result.collapsed_df.loc[5, "x"] == 100.0
    assert result.collapsed_df.loc[8, "x"] == 30.0


def test_mixed_fit_transform_and_collapse_share_original_schema():
    original = pd.DataFrame(
        {"x": [10.0, 20.0, np.nan, 40.0, 50.0], "group": ["A", "B", "A", "C", "A"]},
        index=[10, 20, 30, 40, 50],
    )
    model = Phil(samples=2, random_state=42)
    fitted = model.fit(original)
    pd.testing.assert_frame_equal(fitted, model.transform(original))
    pd.testing.assert_frame_equal(
        fitted, model.collapse("euclidean_medoid").collapsed_df
    )
    assert list(fitted.columns) == list(original.columns)
    pd.testing.assert_series_equal(fitted["group"], original["group"])


@pytest.mark.parametrize("explicit_cdi", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_full_named_covariance_is_used_in_each_target_regression(explicit_cdi, masked):
    rng = np.random.default_rng(11)
    original = pd.DataFrame(rng.normal(size=(30, 3)), columns=["x", "y", "z"])
    original["x"] *= 10
    original.loc[:4, "x"] = np.nan
    original.loc[5:9, "y"] = np.nan
    prior = np.array([[4.0, 0.2, 0.3], [0.2, 9.0, 0.4], [0.3, 0.4, 16.0]])
    domain = DomainKnowledge(
        covariance_matrix=CovarianceMatrix(
            matrix=prior.tolist(), variables=["z", "x", "y"], citations=["test prior"]
        ),
        covariate_subsets={
            "x": CovariateSubset(predictors=["z"], citations=["test mask"])
        }
        if masked
        else None,
    )
    config = ImputationConfig(
        methods=["CovariateDistributionImputer" if explicit_cdi else "BayesianRidge"],
        modules=["phil.imputation" if explicit_cdi else "sklearn.linear_model"],
        grids=[ParameterGrid({})],
        domain_knowledge=domain,
    )
    model = Phil(samples=10, param_grid=config, random_state=42)
    result = model.fit(original)
    assert not result.isna().any().any()
    covariance_imputers = [
        pipe.named_steps["imputer"]
        for pipe in model.selected_imputers
        if isinstance(
            pipe.named_steps["imputer"].estimator, CovariateDistributionImputer
        )
    ]
    assert covariance_imputers
    # Prior variables are z,x,y; actual data is x,y,z, with x on a larger scale.
    reordered = prior[np.ix_([1, 2, 0], [1, 2, 0])]
    scale = original.std(ddof=0).to_numpy()
    expected = reordered / np.outer(scale, scale)
    for imputer in covariance_imputers:
        assert imputer.imputation_sequence_
        for step in imputer.imputation_sequence_:
            neighbors = [2] if masked and step.feat_idx == 0 else step.neighbor_feat_idx
            np.testing.assert_allclose(
                step.estimator.VI_,
                np.linalg.inv(expected[np.ix_(neighbors, neighbors)]),
            )
            if masked and step.feat_idx == 0:
                assert step.estimator.X_obs_.shape[1] == 1


def test_selection_reports_categorical_ensemble_disagreement():
    result = collapse_ect_ensemble(
        [pd.DataFrame({"group": [value]}) for value in ["A", "B", "C"]],
        [np.array([value]) for value in [0.0, 1.0, 2.0]],
        strategy="euclidean_medoid",
        original_df=pd.DataFrame({"group": [None]}),
    )
    variability = result.column_variability["group"]
    assert variability["modal_agreement_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert variability["between_imputation_disagreement"] == pytest.approx(
        2 / 3, abs=1e-4
    )
    assert variability["uncertainty_tier"] == "high"


class RecordingECT:
    def generate(self, arrays):
        self.received = arrays[0]
        # This signature lies beyond the supplied candidate signatures.
        return [np.array([2.0])]


def test_fidelity_encodes_returned_category_vote_and_auto_falls_back():
    candidates = [pd.DataFrame({"group": [value]}) for value in ["A", "B"]]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(sparse_output=False), ["group"])]
    )
    preprocessor.fit(pd.concat(candidates))
    magic = RecordingECT()
    kwargs = {
        "candidates": candidates,
        "descriptors": [np.array([0.0]), np.array([1.0])],
        "standardized_candidates": [preprocessor.transform(c) for c in candidates],
        "magic_estimator": magic,
        "preprocessor": preprocessor,
    }
    pooled = collapse_ect_ensemble(strategy="barycenter", **kwargs)
    np.testing.assert_array_equal(
        magic.received, preprocessor.transform(pooled.collapsed_df)
    )
    np.testing.assert_array_equal(magic.received, [[1.0, 0.0]])
    assert pooled.manifold_convexity_ratio == 3.0
    assert pooled.non_convex_manifold_warning
    automatic = collapse_ect_ensemble(strategy="auto", **kwargs)
    assert automatic.strategy == "frechet_medoid"


def test_unavailable_fidelity_is_not_reported_as_success():
    result = collapse_ect_ensemble(
        [pd.DataFrame({"x": [value]}) for value in [0.0, 1.0]],
        [np.array([0.0]), np.array([1.0])],
        strategy="barycenter",
    )
    assert result.ect_distance_to_median is None
    assert result.manifold_convexity_ratio is None


def test_fidelity_evaluates_bounds_applied_to_selected_candidate():
    magic = RecordingECT()
    result = collapse_ect_ensemble(
        [pd.DataFrame({"x": [40.0]}), pd.DataFrame({"x": [50.0]})],
        [np.array([0.0]), np.array([1.0])],
        strategy="euclidean_medoid",
        variable_bounds={"x": {"max_value": 30.0}},
        preprocessor=FunctionTransformer(),
        magic_estimator=magic,
    )
    np.testing.assert_array_equal(magic.received, [[30.0]])
    assert result.ect_distance_to_median == 1.5


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", "bad"),
        ("temperature", "0"),
        ("temperature", ".nan"),
        ("trim_quantile", "true"),
        ("trim_quantile", "-.1"),
        ("trim_quantile", "1"),
    ],
)
def test_invalid_collapse_parameters_fail_config_validation(key, value):
    report = validate_config_yaml(f"magic:\n  {key}: {value}\n")
    assert not report.ok
    assert any(issue.path == f"$.magic.{key}" for issue in report.issues)
