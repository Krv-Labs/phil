"""
Phil package.
"""

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from phil.diagnostics import build_research_worksheet, profile_predictability
from phil.gallery import GridGallery
from phil.imputation import (
    CovarianceMatrix,
    CovariateDistributionImputer,
    CovariateSubset,
    DistributionImputer,
    DomainKnowledge,
    ImputationConfig,
    MaskedIterativeImputer,
    PreprocessingConfig,
    VariableBounds,
)
from phil.magic import (
    COLLAPSE_STRATEGIES,
    ECT,
    ECTCollapseResult,
    ECTConfig,
    analyze_ect_distribution,
    collapse_ect_ensemble,
    compute_weiszfeld_median,
)
from phil.phil import Phil
from phil.transformers import PhilTransformer
from phil.visualization import plot_mds

__version__ = "0.1.1"
__all__ = [
    "COLLAPSE_STRATEGIES",
    "ECT",
    "CovarianceMatrix",
    "CovariateDistributionImputer",
    "CovariateSubset",
    "DistributionImputer",
    "DomainKnowledge",
    "ECTCollapseResult",
    "ECTConfig",
    "GridGallery",
    "ImputationConfig",
    "MaskedIterativeImputer",
    "Phil",
    "PhilTransformer",
    "PreprocessingConfig",
    "VariableBounds",
    "analyze_ect_distribution",
    "build_research_worksheet",
    "collapse_ect_ensemble",
    "compute_weiszfeld_median",
    "plot_mds",
    "profile_predictability",
]
