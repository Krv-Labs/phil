"""
Euler Characteristic Transform (ECT) configuration.
"""

from pydantic import BaseModel, Field


class ECTConfig(BaseModel):
    num_thetas: int = Field(..., description="Number of angles to sample")
    radius: float = Field(..., description="Maximum radius for filtration")
    resolution: int = Field(..., description="Number of points per direction")
    scale: int = Field(..., description="Scaling factor for point cloud")
    normalize: bool = Field(True, description="Whether to normalize the ECT output")
    seed: int = Field(0, description="Random seed for reproducibility")
