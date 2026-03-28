"""
Configuration models for Phil's imputation strategies.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from sklearn.model_selection import ParameterGrid


class ImputationConfig(BaseModel):
    """Configuration for imputation methods and parameter grids."""

    model_config = {"arbitrary_types_allowed": True}

    methods: List[str] = Field(..., description="Names of imputation methods")
    modules: List[str] = Field(..., description="Python modules containing methods")
    grids: List[ParameterGrid] = Field(..., description="Parameter grids for methods")


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing steps."""

    method: str
    module: str = "sklearn.preprocessing"
    params: Dict[str, Any] = Field(default_factory=dict)
