"""
Phil package.
"""

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from phil.gallery import GridGallery
from phil.imputation import (
    CovariateDistributionImputer,
    DistributionImputer,
    ImputationConfig,
    PreprocessingConfig,
)
from phil.magic import ECT, ECTConfig
from phil.phil import Phil
from phil.transformers import PhilTransformer
from phil.visualization import plot_mds

__version__ = "1.1.0"
__all__ = [
    "ECT",
    "CovariateDistributionImputer",
    "DistributionImputer",
    "ECTConfig",
    "GridGallery",
    "ImputationConfig",
    "Phil",
    "PhilTransformer",
    "PreprocessingConfig",
    "plot_mds",
]

