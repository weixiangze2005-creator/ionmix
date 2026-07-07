from __future__ import annotations

from typing import Any

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
    score_threshold: float = Field(0.0, ge=0, le=100)
    max_results: int = Field(80, ge=1, le=200)
    max_components: int = Field(2, ge=2, le=3)
    return_all_above_threshold: bool = False
    allow_relaxed_fallback: bool = True
    weights: WeightInput = WeightInput()


class AuthRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=8, max_length=200)
    display_name: str | None = Field(None, max_length=40)


class SavedFormulaRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    recommendation: dict[str, Any]
    request_context: dict[str, Any] | None = None
