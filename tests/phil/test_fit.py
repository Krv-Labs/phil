"""Tests for Phil's fit behavior."""

import pytest
import pandas as pd
import numpy as np
from phil.phil import Phil


class TestPhilFitBehavior:
    """Tests for Phil's transform behavior."""

    def test_fit_returns_imputed_dataframe_numeric(self, mocker):
        # Create a test dataframe with missing values
        df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8]})

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        representations = [imputed_array]

        # Setup mocks
        mocker.patch.object(phil, "impute", return_value=representations)
        mocker.patch.object(
            phil, "generate_descriptors", return_value=[np.array([0.1, 0.2])]
        )
        mocker.patch.object(phil, "_select_representative", return_value=0)

        # Create a mock for selected_imputers
        mock_pipeline = mocker.MagicMock()
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, "_get_imputed_columns", return_value=["A", "B"])
        mock_pipeline.named_steps = {"imputer": mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        result = phil.fit(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert list(result.columns) == ["A", "B"]
        phil.impute.assert_called_once_with(df, 5)
        assert phil.closest_index == 0

    def test_fit_imputes_missing_values_mixed_types(self, mocker):
        df = pd.DataFrame(
            {
                "num_col": [1.0, 2.0, np.nan, 4.0],
                "cat_col": ["a", "b", np.nan, "d"],
            }
        )

        # Mock the necessary methods
        phil = Phil()

        # Create mock return values
        imputed_df = pd.DataFrame(
            {"num_col": [1.0, 2.0, 3.0, 4.0], "cat_col": ["a", "b", "c", "d"]}
        ).values

        mock_representations = [imputed_df]
        mock_descriptors = [np.array([0.1, 0.2, 0.3])]

        # Setup mocks
        mocker.patch.object(phil, "impute", return_value=mock_representations)
        mocker.patch.object(phil, "generate_descriptors", return_value=mock_descriptors)
        mocker.patch.object(phil, "_select_representative", return_value=0)

        # Create a mock imputer pipeline
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(
            phil, "_get_imputed_columns", return_value=["num_col", "cat_col"]
        )

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.named_steps = {"imputer": mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        result = phil.fit(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert list(result.columns) == ["num_col", "cat_col"]
        phil.impute.assert_called_once_with(df, 5)
        assert phil.closest_index == 0

    def test_fit_raises_error_with_no_missing_values(self, mocker):
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})

        phil = Phil()
        mocker.patch.object(
            phil,
            "impute",
            side_effect=ValueError("No missing values found in the input DataFrame."),
        )

        with pytest.raises(
            ValueError, match="No missing values found in the input DataFrame."
        ):
            phil.fit(df)

        phil.impute.assert_called_once_with(df, 5)

    def test_fit_with_various_max_iter(self, mocker):
        df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8]})

        phil = Phil()

        # Create mock return values
        imputed_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        representations = [imputed_array]

        # Setup mocks
        mocker.patch.object(phil, "impute", return_value=representations)
        mocker.patch.object(
            phil, "generate_descriptors", return_value=[np.array([0.1, 0.2])]
        )
        mocker.patch.object(phil, "_select_representative", return_value=0)

        # Create a mock for selected_imputers
        mock_pipeline = mocker.MagicMock()
        mock_imputer = mocker.MagicMock()
        mocker.patch.object(phil, "_get_imputed_columns", return_value=["A", "B"])
        mock_pipeline.named_steps = {"imputer": mock_imputer}
        phil.selected_imputers = [mock_pipeline]

        # Test with different max_iter values
        result_5 = phil.fit(df, max_iter=5)
        result_10 = phil.fit(df, max_iter=10)

        assert isinstance(result_5, pd.DataFrame)
        assert result_5.shape == (4, 2)
        assert list(result_5.columns) == ["A", "B"]

        assert isinstance(result_10, pd.DataFrame)
        assert result_10.shape == (4, 2)
        assert list(result_10.columns) == ["A", "B"]

        assert phil.impute.call_count == 2
        phil.impute.assert_any_call(df, 5)
        phil.impute.assert_any_call(df, 10)
