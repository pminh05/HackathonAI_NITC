"""Structured need-profile models for dishwasher advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProductType = Literal["freestanding", "built_in", "semi_integrated", "mini"]
CapacitySegment = Literal["compact", "standard", "large", "open"]
UsagePreference = Literal[
    "quick_cycle",
    "pots_and_pans",
    "glass_care",
    "hygiene_care",
    "quiet_night",
    "water_saving",
    "drying_performance",
    "smart_control",
    "flexible_racks",
]


class DishwasherHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``mayruachen`` metadata."""

    brands: list[str] = Field(default_factory=list)
    product_types: list[ProductType] = Field(default_factory=list)
    min_place_settings: int | None = Field(default=None, ge=1, le=100)
    min_vietnamese_meals: int | None = Field(default=None, ge=1, le=100)
    max_width_cm: float | None = Field(default=None, gt=0, le=500)
    max_height_cm: float | None = Field(default=None, gt=0, le=500)
    max_depth_cm: float | None = Field(default=None, gt=0, le=500)
    max_noise_db: float | None = Field(default=None, gt=0, le=200)
    max_water_l_per_cycle: float | None = Field(default=None, gt=0, le=100)


class DishwasherNeedProfile(BaseModel):
    """Customer needs consumed by the dishwasher advisory pipeline."""

    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    capacity_segment: CapacitySegment | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: DishwasherHardConstraints = Field(
        default_factory=DishwasherHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class DishwasherCustomAnswer(BaseModel):
    """Validated interpretation of a dishwasher free-form form answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    capacity_segment: CapacitySegment | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: DishwasherHardConstraints = Field(
        default_factory=DishwasherHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
