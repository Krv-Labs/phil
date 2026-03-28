"""Tests for Phil's descriptor generation functionality."""

import numpy as np
from phil.phil import Phil


class TestPhilDescriptorBehavior:
    """Tests for Phil's descriptor generation behavior."""

    def test_generates_descriptors_for_imputed_datasets_with_mocked_configure_magic_method(
        self, mocker
    ):
        # Create mock magic object with a transformation that preserves array structure
        mock_magic = mocker.Mock()
        mock_magic.generate = mocker.Mock(
            side_effect=lambda arrays: [array + 1 for array in arrays]
        )

        # Mock the _configure_magic_method to return a tuple of (mock_config, mock_magic)
        mock_config = mocker.Mock()
        mocker.patch.object(
            Phil,
            "_configure_magic_method",
            return_value=(mock_config, mock_magic),
        )

        # Create test instance with mocked _configure_magic_method
        phil = Phil(magic="test_magic")

        # Set up test data with explicit arrays
        test_representations = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        phil.representations = test_representations

        # Act
        result = phil.generate_descriptors()

        # Assert
        assert len(result) == len(test_representations)
        assert isinstance(result, list)
        assert all(isinstance(desc, np.ndarray) for desc in result)
        np.testing.assert_array_equal(result[0], test_representations[0] + 1)
        np.testing.assert_array_equal(result[1], test_representations[1] + 1)

        # Verify the generate method was called once with all arrays
        mock_magic.generate.assert_called_once_with(test_representations)

    def test_generate_descriptors_with_batch_processing(self, mocker):
        """Test that generate_descriptors correctly handles batch processing of arrays."""
        # Create test instance with mocked magic method
        mock_magic = mocker.Mock()
        # Mock the generate method to verify batch processing
        mock_magic.generate = mocker.Mock(
            side_effect=lambda x: [np.array([i + 1]) for i in range(len(x))]
        )

        # Mock the _configure_magic_method
        mock_config = mocker.Mock()
        mocker.patch.object(
            Phil,
            "_configure_magic_method",
            return_value=(mock_config, mock_magic),
        )

        # Create test instance
        phil = Phil(magic="test_magic")

        # Test with different batch sizes
        test_cases = [
            # Single array
            [np.array([[1, 2], [3, 4]])],
            # Small batch of arrays
            [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
            # Larger batch of arrays
            [
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6], [7, 8]]),
                np.array([[9, 10], [11, 12]]),
            ],
        ]

        for batch in test_cases:
            phil.representations = batch
            result = phil.generate_descriptors()

            # Verify results
            assert len(result) == len(batch)
            assert isinstance(result, list)
            assert all(isinstance(desc, np.ndarray) for desc in result)

            # Verify the magic.generate method was called with the entire batch
            mock_magic.generate.assert_called_with(batch)

            # Verify each result matches expected output
            for i, desc in enumerate(result):
                assert desc == np.array([i + 1])  # Based on our mock's side_effect

    def test_generate_descriptors_handles_empty_representations(self, mocker):
        """Test that generate_descriptors handles empty representations appropriately."""
        # Create mock magic object
        mock_magic = mocker.Mock()
        mock_magic.generate.return_value = []

        # Mock the _configure_magic_method
        mock_config = mocker.Mock()
        mocker.patch.object(
            Phil,
            "_configure_magic_method",
            return_value=(mock_config, mock_magic),
        )

        # Create test instance
        phil = Phil(magic="test_magic")
        phil.representations = []

        # Execute and verify
        result = phil.generate_descriptors()
        assert isinstance(result, list)
        assert len(result) == 0
        mock_magic.generate.assert_called_once_with([])

    def test_generate_descriptors_maintains_array_types(self, mocker):
        """Test that generate_descriptors maintains the correct array types."""
        # Create mock magic object
        mock_magic = mocker.Mock()
        mock_magic.generate = mocker.Mock(
            side_effect=lambda x: [arr.astype(arr.dtype) for arr in x]
        )

        # Mock the _configure_magic_method
        mock_config = mocker.Mock()
        mocker.patch.object(
            Phil,
            "_configure_magic_method",
            return_value=(mock_config, mock_magic),
        )

        # Create test instance
        phil = Phil(magic="test_magic")

        # Test with arrays of different dtypes
        test_arrays = [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64),
            np.array([[True, False], [False, True]], dtype=bool),
        ]
        phil.representations = test_arrays

        # Execute
        result = phil.generate_descriptors()

        # Verify results
        assert len(result) == len(test_arrays)
        for original, processed in zip(test_arrays, result):
            assert processed.dtype == original.dtype
            assert processed.shape == original.shape

        # Verify the generate method was called with all arrays
        mock_magic.generate.assert_called_once_with(test_arrays)
