"""
Configuration models for Phil's imputation strategies.
"""

from typing import Any

from pydantic import BaseModel, Field
from sklearn.model_selection import ParameterGrid


class CovariateSubset(BaseModel):
    predictors: list[str]
    citations: list[str]


class CovarianceMatrix(BaseModel):
    matrix: list[list[float]]
    variables: list[str]
    citations: list[str]


class DomainKnowledge(BaseModel):
    covariate_subsets: dict[str, CovariateSubset] | None = None
    covariance_matrix: CovarianceMatrix | None = None


class ImputationConfig(BaseModel):
    """Configuration for imputation methods and parameter grids."""

    model_config = {"arbitrary_types_allowed": True}

    methods: list[str] = Field(..., description="Names of imputation methods")
    modules: list[str] = Field(..., description="Python modules containing methods")
    grids: list[ParameterGrid] = Field(..., description="Parameter grids for methods")
    domain_knowledge: DomainKnowledge | None = Field(
        None, description="Domain knowledge for covariate imputation"
    )


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing steps."""

    method: str
    module: str = "sklearn.preprocessing"
    params: dict[str, Any] = Field(default_factory=dict)
