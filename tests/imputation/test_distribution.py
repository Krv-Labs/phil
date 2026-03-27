import numpy as np
import pandas as pd
import pytest

from phil.imputation import DistributionImputer


class TestDistributionImputer:
    """Tests for the DistributionImputer class."""

    def test_numeric_imputation(self):
        # Test with numeric data
        y = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        X = np.ones((len(y), 1))  # Dummy features

        imputer = DistributionImputer(random_state=42)
        imputer.fit(X, y)
        predictions = imputer.predict(X)

        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))
        assert all(val in [1.0, 3.0, 4.0] for val in predictions)

    def test_categorical_imputation(self):
        # Test with categorical data
        y = np.array(["a", None, "b", "c", None])
        X = np.ones((len(y), 1))  # Dummy features

        imputer = DistributionImputer()
        imputer.fit(X, y)
        predictions = imputer.predict(X)

        assert len(predictions) == len(y)
        assert all(val in ["a", "b", "c"] for val in predictions)

    def test_threshold_behavior(self):
        # Test threshold functionality
        y = np.array([1.0, np.nan, np.nan, 4.0, np.nan])  # 60% missing
        X = np.ones((len(y), 1))

        imputer_skip = DistributionImputer(threshold=0.4)
        imputer_skip.fit(X, y)
        predictions_skip = imputer_skip.predict(X)
        assert np.all(np.isnan(predictions_skip))

        imputer_noskip = DistributionImputer(threshold=0.6)
        imputer_noskip.fit(X, y)
        predictions_noskip = imputer_noskip.predict(X)
        assert not np.all(np.isnan(predictions_noskip))

    def test_empty_distribution(self):
        # Test behavior with all missing values
        y = np.array([np.nan, np.nan, np.nan])
        X = np.ones((len(y), 1))

        imputer = DistributionImputer()
        imputer.fit(X, y)
        predictions = imputer.predict(X)

        assert np.all(np.isnan(predictions))

    def test_custom_missing_value(self):
        # Test with custom missing value
        y = np.array([1.0, None, 3.0, 4.0, None])
        X = np.ones((len(y), 1))

        imputer = DistributionImputer(missing_values=None)
        imputer.fit(X, y)
        predictions = imputer.predict(X)

        assert len(predictions) == len(y)
        assert not np.any(pd.isnull(predictions))
        assert all(val in [1.0, 3.0, 4.0] for val in predictions)

    def test_invalid_input_dimensions(self):
        # Test handling of invalid input dimensions
        y = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D array
        X = np.ones((2, 1))

        imputer = DistributionImputer()
        with pytest.raises(ValueError):
            imputer.fit(X, y)

    def test_random_state_reproducibility(self):
        # Test that random state ensures reproducibility
        y = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        X = np.ones((len(y), 1))

        imputer1 = DistributionImputer(random_state=42)
        imputer2 = DistributionImputer(random_state=42)
        imputer1.fit(X, y)
        imputer2.fit(X, y)

        pred1 = imputer1.predict(X)
        pred2 = imputer2.predict(X)
        np.testing.assert_array_equal(pred1, pred2)

        # Different random state should give different results
        imputer3 = DistributionImputer(random_state=43)
        imputer3.fit(X, y)
        pred3 = imputer3.predict(X)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(pred1, pred3)

    def test_predict_without_fit(self):
        # Test error handling when predict is called before fit
        X = np.ones((3, 1))
        imputer = DistributionImputer()

        with pytest.raises(RuntimeError):
            imputer.predict(X)
