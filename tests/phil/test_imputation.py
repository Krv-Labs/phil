"""Tests for Phil's imputation functionality."""

import pytest
import pandas as pd
import numpy as np
from phil.phil import Phil
from phil import ImputationConfig
from sklearn.model_selection import ParameterGrid


class TestPhilImputationBehavior:
    """Tests for Phil's imputation behavior."""

    def test_impute_empty_dataframe(self, mocker):
        df = pd.DataFrame()
        phil = Phil()

        with pytest.raises(
            ValueError, match="No missing values found in the input DataFrame."
        ):
            phil.impute(df)

    def test_impute_no_missing_values(self):
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0],
                "num2": [5.0, 6.0, 7.0, 8.0],
                "cat1": ["a", "b", "c", "d"],
                "cat2": ["w", "x", "y", "z"],
            }
        )

        phil = Phil()

        with pytest.raises(
            ValueError, match="No missing values found in the input DataFrame."
        ):
            phil.impute(df, max_iter=10)

    def test_impute_with_missing_values(self, mocker):
        df = pd.DataFrame(
            {
                "num1": [1.0, np.nan, 3.0, 4.0],
                "num2": [np.nan, 6.0, 7.0, 8.0],
                "cat1": ["a", np.nan, "c", "d"],
                "cat2": ["x", "y", np.nan, "w"],
            }
        )

        phil = Phil()
        mock_identify = mocker.patch.object(
            phil,
            "_identify_column_types",
            return_value=(["cat1", "cat2"], ["num1", "num2"]),
        )
        mock_configure = mocker.patch.object(
            phil, "_configure_preprocessor", return_value=mocker.MagicMock()
        )
        mock_create = mocker.patch.object(
            phil, "_create_imputers", return_value=[mocker.MagicMock()]
        )
        mock_select = mocker.patch.object(
            phil, "_select_imputations", return_value=[mocker.MagicMock()]
        )
        mock_apply = mocker.patch.object(
            phil,
            "_apply_imputations",
            return_value=[np.array([[1, 2, 3, 4], [5, 6, 7, 8]])],
        )

        result = phil.impute(df, max_iter=10)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], np.ndarray)
        mock_identify.assert_called_once_with(df)
        mock_configure.assert_called_once_with(
            "default", ["cat1", "cat2"], ["num1", "num2"]
        )
        mock_create.assert_called_once()
        mock_select.assert_called_once()
        mock_apply.assert_called_once()

    def test_impute_samples_larger_than_imputers(self, mocker):
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, np.nan, 4.0],
                "num2": [5.0, np.nan, 7.0, 8.0],
                "cat1": ["a", "b", np.nan, "d"],
                "cat2": [np.nan, "y", "z", "w"],
            }
        )

        phil = Phil()
        phil.samples = 10  # Set samples larger than the number of imputers

        mock_identify = mocker.patch.object(
            phil,
            "_identify_column_types",
            return_value=(["cat1", "cat2"], ["num1", "num2"]),
        )
        mock_configure = mocker.patch.object(
            phil, "_configure_preprocessor", return_value=mocker.MagicMock()
        )
        mock_create = mocker.patch.object(
            phil,
            "_create_imputers",
            return_value=[mocker.MagicMock() for _ in range(3)],
        )
        mock_select = mocker.patch.object(
            phil, "_select_imputations", return_value=[mocker.MagicMock()]
        )
        mock_apply = mocker.patch.object(
            phil,
            "_apply_imputations",
            return_value=[np.array([[1, 2, 3, 4], [5, 6, 7, 8]])],
        )

        result = phil.impute(df, max_iter=15)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], np.ndarray)
        mock_identify.assert_called_once_with(df)
        mock_configure.assert_called_once_with(
            "default", ["cat1", "cat2"], ["num1", "num2"]
        )
        mock_create.assert_called_once()
        mock_select.assert_called_once()
        mock_apply.assert_called_once()

    def test_custom_impute_with_distribution_preserver(self, mocker):
        """Test that Phil works correctly with DistributionPreserver"""
        df = pd.DataFrame(
            {
                "num_col": [1.0, 2.0, np.nan, 4.0],
                "cat_col": ["a", "b", np.nan, "d"],
            }
        )

        # Configure Phil to use DistributionPreserver
        distribution_params = ImputationConfig(
            methods=["DistributionImputer"],
            modules=["phil.imputation"],
            grids=[ParameterGrid({"random_state": [42]})],
        )

        # Initialize Phil with custom parameter grid
        phil = Phil(param_grid=distribution_params)

        # Mock the internal methods to verify the flow
        mock_identify = mocker.patch.object(
            phil,
            "_identify_column_types",
            return_value=(["cat_col"], ["num_col"]),
        )
        mock_configure = mocker.patch.object(
            phil, "_configure_preprocessor", return_value=mocker.MagicMock()
        )

        # Create a mock imputer with DistributionPreserver structure
        mock_imputer = mocker.MagicMock()
        mock_imputer.named_steps = {
            "imputer": mocker.MagicMock(spec=["DistributionPreserver"])
        }
        mock_imputers = [mock_imputer]

        mock_create = mocker.patch.object(
            phil, "_create_imputers", return_value=mock_imputers
        )
        mock_select = mocker.patch.object(
            phil, "_select_imputations", return_value=[mocker.MagicMock()]
        )
        mock_apply = mocker.patch.object(
            phil,
            "_apply_imputations",
            return_value=[np.array([[1, 2, 3, 4], ["a", "b", "c", "d"]]).T],
        )

        result = phil.impute(df, max_iter=10)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], np.ndarray)
        mock_identify.assert_called_once_with(df)
        mock_configure.assert_called_once_with("default", ["cat_col"], ["num_col"])
        mock_create.assert_called_once()
        mock_select.assert_called_once()
        mock_apply.assert_called_once()
