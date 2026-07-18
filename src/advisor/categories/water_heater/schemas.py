"""Structured need-profile models for water-heater advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProductType = Literal["direct", "indirect", "solar", "direct_multipoint"]
WaterSupply = Literal["stable", "low_pressure", "multi_outlet", "open"]
UsagePreference = Literal[
    "instant_heating",
    "stored_hot_water",
    "multi_outlet",
    "low_pressure",
    "energy_saving",
    "compact_installation",
    "fast_heating",
    "stable_temperature",
]
RequiredFeature = Literal["booster_pump", "included_shower"]
SafetyFeature = Literal[
    "elcb",
    "rcd",
    "overheat_cutoff",
    "flow_sensor",
    "pressure_relief_valve",
    "waterproof",
    "anti_scald",
    "thermal_stabilizer",
]


class WaterHeaterHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``maynuocnong`` metadata."""

    brands: list[str] = Field(default_factory=list)
    product_types: list[ProductType] = Field(default_factory=list)
    min_capacity_lit: int | None = Field(default=None, ge=0, le=1000)
    max_capacity_lit: int | None = Field(default=None, ge=0, le=1000)
    max_power_w: int | None = Field(default=None, gt=0, le=100_000)
    max_heating_time_minutes: float | None = Field(default=None, ge=0, le=1440)
    max_width_cm: float | None = Field(default=None, gt=0, le=1000)
    max_height_cm: float | None = Field(default=None, gt=0, le=1000)
    max_depth_cm: float | None = Field(default=None, gt=0, le=1000)
    required_features: list[RequiredFeature] = Field(default_factory=list)
    required_safety_features: list[SafetyFeature] = Field(default_factory=list)
    ip_ratings: list[str] = Field(default_factory=list)


class WaterHeaterNeedProfile(BaseModel):
    """Customer needs consumed by the water-heater advisory pipeline."""

    household_size: int | None = Field(default=None, ge=1, le=50)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    water_supply: WaterSupply | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: WaterHeaterHardConstraints = Field(
        default_factory=WaterHeaterHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class WaterHeaterCustomAnswer(BaseModel):
    """Validated interpretation of a water-heater free-form form answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    household_size: int | None = Field(default=None, ge=1, le=50)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    water_supply: WaterSupply | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: WaterHeaterHardConstraints = Field(
        default_factory=WaterHeaterHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
