"""Pipeline construction tests for built-in gallery grids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import IterativeImputer

from phil.gallery import GRID_METADATA, GridGallery, grid_candidate_count
from phil.phil import Phil


def _tiny_missing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0, 5.0],
            "y": [2.0, 3.0, np.nan, 5.0, 6.0],
        }
    )


@pytest.mark.parametrize("grid_name", sorted(GRID_METADATA))
def test_builtin_grid_constructs_and_imputes(grid_name: str) -> None:
    phil = Phil(param_grid=grid_name, samples=2, random_state=0)
    result = phil.impute(_tiny_missing_frame(), max_iter=2)
    assert len(result) == 2
    assert all(isinstance(arr, np.ndarray) for arr in result)


def test_caller_max_iter_applied_to_standalone_iterative() -> None:
    phil = Phil(param_grid="marketing", samples=3, random_state=0)
    # Build pipelines without sampling so we can inspect max_iter.
    categorical_columns, numerical_columns = phil._identify_column_types(
        _tiny_missing_frame()
    )
    preprocessor = phil._configure_preprocessor(
        "default", categorical_columns, numerical_columns
    )
    preprocessor.fit(_tiny_missing_frame())
    phil.feature_names_out_ = preprocessor.get_feature_names_out().tolist()
    pipelines = phil._create_imputers(preprocessor, max_iter=17)

    iterative = [
        pipe.named_steps["imputer"]
        for pipe in pipelines
        if isinstance(pipe.named_steps["imputer"], IterativeImputer)
    ]
    assert iterative
    assert all(imp.get_params()["max_iter"] == 17 for imp in iterative)


def test_grid_metadata_matches_gallery_keys() -> None:
    assert set(GRID_METADATA).issubset(set(GridGallery._grids))


def test_marketing_candidate_count_pinned() -> None:
    # SimpleImputer×3 + KNN×2 + Iterative×3 = 8 after constant/"unknown" removal.
    assert grid_candidate_count("marketing") == 8
