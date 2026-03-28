"""Tests for Phil's initialization functionality."""

import pytest
from phil.phil import Phil
from phil.magic import Magic, ECT, ECTConfig


class TestPhilInitializationBehavior:
    """Tests for Phil's initialization behavior."""

    def test_init_with_default_parameters(self, mocker):
        from pydantic import BaseModel

        mock_config = mocker.Mock(spec=BaseModel)
        mock_magic = mocker.Mock(spec=Magic)
        mock_param_grid = mocker.Mock()

        mocker.patch.object(
            Phil,
            "_configure_magic_method",
            return_value=(mock_config, mock_magic),
        )
        mocker.patch.object(Phil, "_configure_param_grid", return_value=mock_param_grid)

        phil = Phil()

        Phil._configure_magic_method.assert_called_once_with(magic="ECT", config=None)
        Phil._configure_param_grid.assert_called_once_with("default")

        assert phil.samples == 30
        assert phil.param_grid == mock_param_grid
        assert phil.random_state is None
        assert phil.config == mock_config
        assert phil.magic == mock_magic
        assert phil.representations == []
        assert phil.magic_descriptors == []

    def test_init_with_invalid_magic_method(self, mocker):
        mocker.patch.object(
            Phil,
            "_configure_magic_method",
            side_effect=ValueError("Magic method 'INVALID_MAGIC' not found."),
        )

        with pytest.raises(ValueError) as excinfo:
            Phil(magic="INVALID_MAGIC")

        assert "Magic method 'INVALID_MAGIC' not found." in str(excinfo.value)
        Phil._configure_magic_method.assert_called_once_with(
            magic="INVALID_MAGIC", config=None
        )

    def test_init_with_custom_magic_method(self, mocker):
        # Create mock magic object with a custom configuration
        mock_config = ECTConfig(
            num_thetas=64,
            radius=1.0,
            resolution=64,
            scale=500,
            seed=42,
        )
        mock_magic = ECT(config=mock_config)

        mocker.patch.object(
            Phil,
            "_configure_magic_method",
            return_value=(mock_config, mock_magic),
        )
        mocker.patch.object(
            Phil, "_configure_param_grid", return_value={"some": "params"}
        )

        phil = Phil(magic="CustomMagic")

        Phil._configure_magic_method.assert_called_once_with(
            magic="CustomMagic", config=None
        )
        Phil._configure_param_grid.assert_called_once_with("default")

        assert phil.samples == 30
        assert phil.param_grid == {"some": "params"}
        assert phil.random_state is None
        assert phil.representations == []
        assert phil.magic_descriptors == []
        assert phil.config == mock_config
        assert phil.magic == mock_magic
