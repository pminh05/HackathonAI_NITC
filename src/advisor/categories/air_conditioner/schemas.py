"""Structured need-profile models for air-conditioner advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MachineType = Literal["one_way", "two_way", "multi_indoor", "multi_outdoor"]
GasType = Literal["r32", "r410a", "r22"]
UsagePreference = Literal[
    "quiet_sleep",
    "energy_saving",
    "fast_cooling",
    "air_quality",
    "smart_control",
    "heating",
]


class AirConditionerHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``maylanh`` metadata."""

    brands: list[str] = Field(default_factory=list)
    machine_types: list[MachineType] = Field(default_factory=list)
    inverter: bool | None = None
    gas_types: list[GasType] = Field(default_factory=list)


class AirConditionerNeedProfile(BaseModel):
    """Customer needs used by the air-conditioner category pipeline."""

    room_area_m2: float | None = Field(default=None, gt=0, le=1000)
    room_volume_m3: float | None = Field(default=None, gt=0, le=5000)
    room_type: Literal["bedroom", "living_room", "office", "shop", "other"] | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: AirConditionerHardConstraints = Field(
        default_factory=AirConditionerHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class AirConditionerCustomAnswer(BaseModel):
    """Validated interpretation of an air-conditioner free-form answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    room_area_m2: float | None = Field(default=None, gt=0, le=1000)
    room_volume_m3: float | None = Field(default=None, gt=0, le=5000)
    room_type: Literal["bedroom", "living_room", "office", "shop", "other"] | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: AirConditionerHardConstraints = Field(
        default_factory=AirConditionerHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)

