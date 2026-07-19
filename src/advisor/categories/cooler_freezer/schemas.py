"""Structured need-profile models for cooler/freezer advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ProductFamily = Literal["cooler", "freezer", "open"]
SizeVariant = Literal["mini", "standard"]
FeatureTag = Literal[
    "glass_door",
    "convertible_mode",
    "fast_freeze",
    "lock",
    "wheels",
    "external_temperature_control",
    "led_light",
    "drain",
]
UsagePreference = Literal[
    "display_drinks",
    "fresh_food_cooling",
    "bulk_frozen_storage",
    "commercial_storage",
    "convertible_use",
    "energy_saving",
]


class CoolerFreezerHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``tumattudong`` metadata."""

    brands: list[str] = Field(default_factory=list)
    size_variants: list[SizeVariant] = Field(default_factory=list)
    min_capacity_lit: int | None = Field(default=None, gt=0, le=10_000)
    max_capacity_lit: int | None = Field(default=None, gt=0, le=10_000)
    required_temperature_c: float | None = Field(default=None, ge=-60, le=20)
    max_width_cm: float | None = Field(default=None, gt=0, le=1_000)
    max_height_cm: float | None = Field(default=None, gt=0, le=1_000)
    max_depth_cm: float | None = Field(default=None, gt=0, le=1_000)
    inverter: bool | None = None
    gas_types: list[Literal["R134a", "R290", "R600a"]] = Field(
        default_factory=list
    )
    required_features: list[FeatureTag] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_capacity_range(self) -> "CoolerFreezerHardConstraints":
        if (
            self.min_capacity_lit is not None
            and self.max_capacity_lit is not None
            and self.min_capacity_lit > self.max_capacity_lit
        ):
            raise ValueError("min_capacity_lit cannot exceed max_capacity_lit")
        return self


class CoolerFreezerNeedProfile(BaseModel):
    """Customer needs consumed by the cooler/freezer pipeline."""

    product_family: ProductFamily | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: CoolerFreezerHardConstraints = Field(
        default_factory=CoolerFreezerHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class CoolerFreezerCustomAnswer(BaseModel):
    """Validated interpretation of a cooler/freezer free-form answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    product_family: ProductFamily | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: CoolerFreezerHardConstraints = Field(
        default_factory=CoolerFreezerHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
