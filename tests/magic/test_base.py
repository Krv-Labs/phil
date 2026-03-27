import pytest
from phil.magic.base import Magic


class TestMagic:

    # Subclass implements configure method correctly
    def test_subclass_implements_configure_correctly(self):
        import numpy as np
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            param1: int = 10
            param2: str = "test"

        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs

            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                return X

        # Create an instance of the concrete subclass with config
        config = TestConfig()
        magic = ConcreteMagic(config=config)

        # Configure with some test parameters
        test_params = {"param1": 10, "param2": "test"}
        magic.configure(**test_params)

        # Verify the configuration was applied correctly
        assert hasattr(magic, "config"), "Configure method should store parameters"
        assert (
            magic.config == test_params
        ), "Configure method should store the provided parameters"

    # Instantiating the abstract base class directly raises TypeError
    def test_instantiating_abstract_base_class_raises_error(self):
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            param: int = 1

        # Attempting to instantiate the abstract base class should raise TypeError
        with pytest.raises(TypeError) as excinfo:
            magic = Magic(config=TestConfig())

        # Verify the error message indicates it's due to abstract methods
        assert (
            "abstract" in str(excinfo.value).lower()
        ), "Error should mention abstract methods"

    # Subclass implements generate method correctly
    def test_subclass_implements_generate_correctly(self):
        import numpy as np
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            param1: int = 10
            param2: str = "test"

        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs

            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                return X * 2  # Example transformation

        # Create an instance of the concrete subclass with config
        config = TestConfig()
        magic = ConcreteMagic(config=config)

        # Test data
        X = np.array([[1, 2], [3, 4]])
        result = magic.generate(X)

        # Verify the transformation was applied
        assert np.array_equal(
            result, X * 2
        ), "Generate method should multiply input by 2"

    def test_generate_returns_correct_shape(self):
        import numpy as np
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            param1: int = 10

        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs

            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                # For testing, simply return the input array
                return X

        # Create an instance of the concrete subclass with config
        config = TestConfig()
        magic = ConcreteMagic(config=config)

        # Test with different array shapes
        shapes = [(3, 4), (5, 2), (10, 1)]
        for shape in shapes:
            X = np.random.rand(*shape)
            result = magic.generate(X)
            assert (
                result.shape == X.shape
            ), f"Output shape {result.shape} doesn't match input shape {X.shape}"

    def test_generate_with_empty_array(self):
        import numpy as np
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            param1: int = 10

        class ConcreteMagic(Magic):
            def configure(self, **kwargs) -> None:
                self.config = kwargs

            def generate(self, X: np.ndarray, **kwargs) -> np.ndarray:
                return X

        # Create an instance of the concrete subclass with config
        config = TestConfig()
        magic = ConcreteMagic(config=config)

        # Test with empty array
        X = np.array([])
        result = magic.generate(X)
        assert isinstance(
            result, np.ndarray
        ), "Should return numpy array even for empty input"
        assert len(result) == 0, "Should preserve empty array length"
