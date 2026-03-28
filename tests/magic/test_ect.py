import pytest
import numpy as np
from phil.magic.ect import ECT
from phil.magic.config import ECTConfig


class TestECT:
    def test_init_with_valid_config(self, mocker):
        # Arrange
        mock_configure = mocker.patch("phil.magic.ect.ECT.configure")
        config = ECTConfig(
            num_thetas=10,
            radius=1.0,
            resolution=100,
            scale=1,
            seed=42,
        )

        # Act
        ect = ECT(config)

        # Assert
        assert ect.config == config
        mock_configure.assert_called_once_with(**config.model_dump())

    def test_valid_configuration_keys_are_updated(self, mocker):
        # Arrange
        mock_config = mocker.MagicMock()
        mock_config.num_thetas = 100
        mock_config.seed = 42
        mock_config.model_dump.return_value = {
            "num_thetas": 100,
            "seed": 42,
        }

        # Mock hasattr to return True for valid keys on the config object
        original_hasattr = hasattr

        def mock_hasattr(obj, attr):
            if obj is mock_config and attr in ["num_thetas", "seed"]:
                return True
            return original_hasattr(obj, attr)

        mocker.patch("builtins.hasattr", side_effect=mock_hasattr)

        # Create instance with mocked config
        ect_instance = ECT(config=mock_config)

        # Act
        ect_instance.configure(num_thetas=200, seed=123)

        # Assert
        assert ect_instance.num_thetas == 200
        assert ect_instance.seed == 123

    def test_generate_ect_2d(self):
        B = 5
        N = 8
        R = 10
        D = 2
        # Arrange
        config = ECTConfig(
            num_thetas=N,
            radius=1.0,
            resolution=R,
            scale=1,
            seed=42,
        )
        ect = ECT(config)
        X = [np.random.rand(100, D) for _ in range(B)]  # 5 batches of 100 2D points

        # Act
        result = ect.generate(X)

        # Assert
        assert isinstance(result, list)
        assert len(result) == B  # Check number of batches

        for batch in result:
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] == config.num_thetas
            assert batch.shape[1] == config.resolution

    def test_generate_ect_3d(self):
        # Arrange
        B = 5
        N = 8
        R = 10
        D = 3
        config = ECTConfig(
            num_thetas=N,
            radius=1.0,
            resolution=R,
            scale=1,
            seed=42,
        )
        ect = ECT(config)
        X = [np.random.rand(100, D) for _ in range(5)]  # 5 batches of 100 3D points

        # Act
        result = ect.generate(X)

        # Assert
        assert isinstance(result, list)
        assert len(result) == B  # Check number of batches
        for batch in result:
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] == config.num_thetas
            assert batch.shape[1] == config.resolution

    def test_generate_with_empty_input(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            seed=42,
        )
        ect = ECT(config)
        X = [np.array([]).reshape(0, 2) for _ in range(2)]  # Empty point cloud

        # Act
        with pytest.raises(ValueError):
            ect.generate(X)

    def test_generate_with_single_point(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            seed=42,
        )
        ect = ECT(config)
        X = [np.random.rand(1, 2)]  # List containing a single point

        # Act
        result = ect.generate(X)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        for batch in result:
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] == config.num_thetas
            assert batch.shape[1] == config.resolution

    def test_generate_with_single_array_raises_error(self):
        # Arrange
        config = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            seed=42,
        )
        ect = ECT(config)
        X = np.random.rand(10, 2)  # Single numpy array, not in a list

        # Act & Assert
        with pytest.raises(ValueError, match="Input must be a list of numpy arrays"):
            ect.generate(X)

    def test_normalization(self):
        """Test that normalization parameter affects the output."""
        # Arrange
        config_normalized = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            normalize=True,
            seed=42,
        )
        config_unnormalized = ECTConfig(
            num_thetas=8,
            radius=1.0,
            resolution=10,
            scale=1,
            normalize=False,
            seed=42,
        )

        ect_normalized = ECT(config_normalized)
        ect_unnormalized = ECT(config_unnormalized)

        # Create some random point clouds
        X = [np.random.rand(100, 2) for _ in range(3)]

        # Act
        result_normalized = ect_normalized.generate(X)
        result_unnormalized = ect_unnormalized.generate(X)

        # Assert
        # Check that results are different when normalization is applied
        for norm, unnorm in zip(result_normalized, result_unnormalized):
            assert not np.allclose(norm, unnorm), (
                "Normalized and unnormalized outputs should be different"
            )

            # Normalized values should be between 0 and 1
            assert np.all(norm >= 0) and np.all(norm <= 1), (
                "Normalized values should be between 0 and 1"
            )

            # Check shapes are the same
            assert norm.shape == unnorm.shape, (
                "Output shapes should be the same regardless of normalization"
            )
