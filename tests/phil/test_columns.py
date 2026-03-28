"""Tests for Phil's column identification and configuration functionality."""

import pytest
import pandas as pd
from phil.phil import Phil


class TestPhilColumnBehavior:
    """Tests for Phil's column identification and type handling behavior."""

    def test_identifies_mixed_data_types_correctly_with_boolean(self):
        data = {
            "string_col": ["apple", "banana", "cherry"],
            "category_col": pd.Series(["dog", "cat", "mouse"]).astype("category"),
            "integer_col": [10, 20, 30],
            "float_col": [0.1, 0.2, 0.3],
            "boolean_col": [True, False, True],
        }
        df = pd.DataFrame(data)

        categorical_cols, numerical_cols = Phil._identify_column_types(df)

        assert set(categorical_cols) == {"string_col", "category_col"}
        assert set(numerical_cols) == {
            "integer_col",
            "float_col",
            "boolean_col",
        }

    def test_identifies_column_types_correctly(self):
        data = {
            "string_col": ["apple", "banana", "cherry"],
            "category_col": pd.Series(["dog", "cat", "mouse"]).astype("category"),
            "integer_col": [10, 20, 30],
            "float_col": [0.1, 0.2, 0.3],
        }
        df = pd.DataFrame(data)

        categorical_cols, numerical_cols = Phil._identify_column_types(df)

        assert set(categorical_cols) == {"string_col", "category_col"}
        assert set(numerical_cols) == {"integer_col", "float_col"}

    def test_preserves_column_order_within_each_type_category(self):
        data = {
            "int_col": [1, 2, 3],
            "object_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3],
            "category_col": pd.Series(["x", "y", "z"]).astype("category"),
        }
        df = pd.DataFrame(data)

        categorical_cols, numerical_cols = Phil._identify_column_types(df)

        assert categorical_cols == ["object_col", "category_col"]
        assert numerical_cols == ["int_col", "float_col"]

    def test_get_imputed_columns_returns_feature_names(self, mocker):
        from sklearn.compose import ColumnTransformer

        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        mock_transformer.get_feature_names_out.return_value = [
            "imputed_col1",
            "imputed_col2",
        ]

        result = Phil._get_imputed_columns(mock_transformer)

        assert result == ["imputed_col1", "imputed_col2"]
        mock_transformer.get_feature_names_out.assert_called_once()

    def test_raises_error_for_unfitted_transformer(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        transformer = ColumnTransformer(
            transformers=[("imputer", SimpleImputer(), ["col1", "col2"])]
        )

        with pytest.raises(ValueError):
            Phil._get_imputed_columns(transformer)

    def test_handles_missing_get_feature_names_out_method(self, mocker):
        from sklearn.compose import ColumnTransformer

        mock_transformer = mocker.Mock(spec=ColumnTransformer)
        del mock_transformer.get_feature_names_out

        with pytest.raises(AttributeError):
            Phil._get_imputed_columns(mock_transformer)
