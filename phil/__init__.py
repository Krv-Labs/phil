"""
Phil package.
"""

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from phil.gallery import GridGallery
from phil.imputation import DistributionImputer, ImputationConfig, PreprocessingConfig
from phil.magic import ECT, ECTConfig
from phil.phil import Phil
from phil.transformers import PhilTransformer

__version__ = "0.1.0"
__all__ = [
    "Phil",
    "PhilTransformer",
    "GridGallery",
    "ECT",
    "ECTConfig",
    "ImputationConfig",
    "PreprocessingConfig",
    "DistributionImputer",
]
