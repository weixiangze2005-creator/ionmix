from __future__ import annotations

from pydantic import BaseModel, Field


class WeightInput(BaseModel):
    solubility: float = Field(0.35, ge=0, le=1)
    conductivity: float = Field(0.30, ge=0, le=1)
    safety: float = Field(0.12, ge=0, le=1)
    stability: float = Field(0.15, ge=0, le=1)
    low_temperature: float = Field(0.08, ge=0, le=1)


class RecommendationRequest(BaseModel):
    salt: str = "LiNO3"
    temperature_c: float = Field(25.0, ge=-60, le=150)
    concentration: float = Field(1.0, gt=0, le=10)
    concentration_unit: str = "mol/kg"
    min_flash_point_c: float = Field(-20.0, ge=-100, le=300)
    max_mixture_viscosity: float = Field(6.0, gt=0, le=100)
    exclude_high_hazard: bool = True
    application: str = Field("lithium_metal", pattern="^(lithium_metal|high_voltage|balanced)$")
    top_k: int = Field(10, ge=1, le=30)
    allow_relaxed_fallback: bool = True
    weights: WeightInput = WeightInput()
