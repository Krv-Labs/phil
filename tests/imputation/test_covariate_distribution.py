import numpy as np
import pytest

from phil.imputation.covariate_distribution import CovariateDistributionImputer


class TestCovariateDistributionImputer:
    def _make_rng(self):
        return np.random.RandomState(42)

    def test_numeric_imputation(self):
        imputer = CovariateDistributionImputer(n_neighbors=2, random_state=42)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([10.0, np.nan, 30.0, 40.0])
        imputer.fit(X, y)
        preds = imputer.predict(np.array([[2.5]]))
        assert preds.shape == (1,)
        assert not np.isnan(preds[0])
        assert preds[0] in [10.0, 30.0, 40.0]

    def test_categorical_imputation(self):
        imputer = CovariateDistributionImputer(n_neighbors=2, random_state=42)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array(["a", None, "b", "c"])
        imputer.fit(X, y)
        preds = imputer.predict(np.array([[2.5]]))
        assert preds.shape == (1,)
        assert preds[0] in ["a", "b", "c"]

    def test_neighbors_condition_on_covariates(self):
        """Closer neighbors should dominate sampling."""
        imputer = CovariateDistributionImputer(n_neighbors=1, random_state=0)
        X_obs = np.array([[0.0], [100.0]])
        y_obs = np.array([1.0, 999.0])
        imputer.fit(X_obs, y_obs)
        # Query near 0 → should pick neighbor with value 1.0
        for _ in range(20):
            assert imputer.predict(np.array([[0.1]]))[0] == 1.0
        # Query near 100 → should pick neighbor with value 999.0
        for _ in range(20):
            assert imputer.predict(np.array([[99.9]]))[0] == 999.0

    def test_threshold_behavior(self):
        imputer_strict = CovariateDistributionImputer(threshold=0.4, random_state=42)
        imputer_loose = CovariateDistributionImputer(threshold=0.8, random_state=42)
        X = np.ones((5, 1))
        y = np.array([1.0, np.nan, np.nan, np.nan, np.nan])  # 80% missing
        imputer_strict.fit(X, y)
        imputer_loose.fit(X, y)
        preds_strict = imputer_strict.predict(X)
        preds_loose = imputer_loose.predict(X)
        assert np.all(np.isnan(preds_strict))
        assert not np.all(np.isnan(preds_loose))

    def test_all_missing_returns_nan(self):
        imputer = CovariateDistributionImputer(random_state=42)
        X = np.ones((3, 1))
        y = np.array([np.nan, np.nan, np.nan])
        imputer.fit(X, y)
        preds = imputer.predict(X)
        assert np.all(np.isnan(preds))

    def test_random_state_reproducibility(self):
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        imp1 = CovariateDistributionImputer(random_state=7).fit(X, y)
        imp2 = CovariateDistributionImputer(random_state=7).fit(X, y)
        q = np.random.rand(5, 2)
        np.testing.assert_array_equal(imp1.predict(q), imp2.predict(q))

    def test_predict_without_fit_raises(self):
        imputer = CovariateDistributionImputer()
        with pytest.raises(RuntimeError, match="fit"):
            imputer.predict(np.array([[1.0]]))

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CovariateDistributionImputer(threshold=1.5)

    def test_invalid_n_neighbors_raises(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            CovariateDistributionImputer(n_neighbors=0)

    def test_k_larger_than_observed_falls_back(self):
        """When k > n_obs, should sample from all observed values."""
        imputer = CovariateDistributionImputer(n_neighbors=100, random_state=42)
        X = np.array([[1.0], [2.0]])
        y = np.array([5.0, 10.0])
        imputer.fit(X, y)
        preds = imputer.predict(np.array([[1.5]]))
        assert preds[0] in [5.0, 10.0]

    def test_invalid_y_dimensions_raises(self):
        imputer = CovariateDistributionImputer()
        X = np.ones((3, 2))
        y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError):
            imputer.fit(X, y)
