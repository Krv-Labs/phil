"""
Phil magic module.
"""

from phil.magic.base import Magic
from phil.magic.collapse import (
    COLLAPSE_STRATEGIES,
    ECTCollapseResult,
    analyze_ect_distribution,
    collapse_ect_ensemble,
    compute_weiszfeld_median,
    list_collapse_strategies_metadata,
)
from phil.magic.config import ECTConfig
from phil.magic.ect import ECT

__all__ = [
    "COLLAPSE_STRATEGIES",
    "ECT",
    "ECTCollapseResult",
    "ECTConfig",
    "Magic",
    "analyze_ect_distribution",
    "collapse_ect_ensemble",
    "compute_weiszfeld_median",
    "list_collapse_strategies_metadata",
]
