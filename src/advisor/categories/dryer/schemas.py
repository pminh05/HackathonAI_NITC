"""Structured need-profile models for clothes-dryer advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


DryerType = Literal["heat_pump", "condenser", "vented"]
UsagePreference = Literal[
    "rainy_season",
    "frequent_drying",
    "bulky_items",
    "delicate_care",
    "energy_saving",
    "quick_dry",
]


class DryerHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``maysayquanao`` metadata."""

    brands: list[str] = Field(default_factory=list)
    dryer_types: list[DryerType] = Field(default_factory=list)
    min_dry_capacity_kg: float | None = Field(default=None, gt=0, le=50)
    max_dry_capacity_kg: float | None = Field(default=None, gt=0, le=50)
    max_width_cm: float | None = Field(default=None, gt=0, le=500)
    max_height_cm: float | None = Field(default=None, gt=0, le=500)
    max_depth_cm: float | None = Field(default=None, gt=0, le=500)
    max_power_w: int | None = Field(default=None, gt=0, le=20_000)
    inverter: bool | None = None
    sensor: bool | None = None

    @model_validator(mode="after")
    def validate_capacity_range(self) -> "DryerHardConstraints":
        if (
            self.min_dry_capacity_kg is not None
            and self.max_dry_capacity_kg is not None
            and self.min_dry_capacity_kg > self.max_dry_capacity_kg
        ):
            raise ValueError("min_dry_capacity_kg cannot exceed max_dry_capacity_kg")
        return self


class DryerNeedProfile(BaseModel):
    """Customer needs consumed by the clothes-dryer pipeline."""

    household_size: int | None = Field(default=None, ge=1, le=30)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: DryerHardConstraints = Field(
        default_factory=DryerHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class DryerCustomAnswer(BaseModel):
    """Validated interpretation of a clothes-dryer free-form answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    household_size: int | None = Field(default=None, ge=1, le=30)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: DryerHardConstraints = Field(
        default_factory=DryerHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
