import pytest
import pandas as pd
from phil.magic import ECTConfig


@pytest.fixture
def test_dataFrame():
    return pd.DataFrame(
        {
            "A": [1, 2, None, 4, 5],
            "B": ["a", "b", None, "d", "d"],
            "C": ["x", "y", "z", None, "w"],
            "D": [None, 10, 20, 30, 40],
            "E": ["p", "q", "r", "r", None],
            "F": ["u", "v", None, "x", "y"],
        }
    )


@pytest.fixture
def mock_ECTConfig():
    return ECTConfig(
        num_thetas=100,
        radius=1,
        resolution=100,
        scale=1,
        seed=0,
    )
