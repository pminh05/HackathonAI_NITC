"""Structured need-profile models for washing-machine advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProductType = Literal[
    "top_load", "front_load", "washer_dryer", "mini", "wash_tower"
]
DrumType = Literal["vertical", "horizontal"]
UsagePreference = Literal[
    "daily_laundry",
    "bulky_items",
    "hygiene_care",
    "energy_saving",
    "quick_wash",
    "wash_and_dry",
]


class WashingMachineHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``maygiat`` metadata."""

    brands: list[str] = Field(default_factory=list)
    product_types: list[ProductType] = Field(default_factory=list)
    drum_types: list[DrumType] = Field(default_factory=list)
    min_wash_capacity_kg: float | None = Field(default=None, gt=0, le=100)
    max_width_cm: float | None = Field(default=None, gt=0, le=500)
    max_height_cm: float | None = Field(default=None, gt=0, le=500)
    max_depth_cm: float | None = Field(default=None, gt=0, le=500)
    inverter: bool | None = None
    dryer: bool | None = None


class WashingMachineNeedProfile(BaseModel):
    """Customer needs consumed by the washing-machine pipeline."""

    household_size: int | None = Field(default=None, ge=1, le=30)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: WashingMachineHardConstraints = Field(
        default_factory=WashingMachineHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class WashingMachineCustomAnswer(BaseModel):
    """Validated interpretation of a washing-machine free-form answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    household_size: int | None = Field(default=None, ge=1, le=30)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: WashingMachineHardConstraints = Field(
        default_factory=WashingMachineHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
