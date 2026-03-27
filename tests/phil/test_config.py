"""Tests for Phil's configuration functionality."""

import pytest
from phil.phil import Phil
from phil import ImputationConfig
from phil.gallery import GridGallery
from pydantic import BaseModel
from sklearn.model_selection import ParameterGrid


class TestPhilConfigBehavior:
    """Tests for Phil's parameter grid configuration behavior."""

    def test_configure_param_grid_with_valid_string(self, mocker):
        """Handles valid string inputs like 'default', 'finance', 'healthcare'."""
        mock_grid = ImputationConfig(
            methods=["TestMethod"], modules=["test.module"], grids=[]
        )
        mocker.patch.object(GridGallery, "get", return_value=mock_grid)

        result = Phil._configure_param_grid("finance")

        GridGallery.get.assert_called_once_with("finance")
        assert result == mock_grid

    def test_configure_param_grid_with_invalid_inputs(self):
        """Ensures incorrect inputs raise ValueError."""

        class MockInvalidBaseModel(BaseModel):
            field: str = "invalid"

        with pytest.raises(ValueError, match="Invalid parameter grid configuration."):
            Phil._configure_param_grid(MockInvalidBaseModel())

        with pytest.raises(ValueError, match="Invalid parameter grid configuration."):
            Phil._configure_param_grid({"invalid": "data"})

        with pytest.raises(ValueError, match="Invalid parameter grid type."):
            Phil._configure_param_grid(123)

    def test_configure_param_grid_with_valid_base_model(self):
        """Ensures valid BaseModel instances are converted to ImputationConfig."""

        class MockValidBaseModel(BaseModel):
            methods: list = ["value"]
            modules: list = ["value"]
            grids: list = [ParameterGrid({})]

        mock_param_grid = MockValidBaseModel()

        result = Phil._configure_param_grid(mock_param_grid)

        assert isinstance(result, ImputationConfig)
        assert result.methods == ["value"]
        assert result.modules == ["value"]
