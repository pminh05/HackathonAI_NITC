"""Structured need-profile models for smartwatch advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SmartwatchUse = Literal[
    "health_monitoring",
    "fitness_sports",
    "outdoor_navigation",
    "calls_notifications",
    "children_safety",
    "everyday_style",
]
PhonePlatform = Literal["ios", "android", "flexible"]
CallRequirement = Literal["on_wrist", "standalone"]
DisplayFamily = Literal[
    "amoled_oled", "mip", "tft_lcd", "ips_lcd", "lcd", "other"
]
StrapMaterialFamily = Literal[
    "silicone",
    "rubber_tpu",
    "leather",
    "fabric_nylon",
    "metal",
    "titanium",
    "composite",
    "other",
]
HealthFeature = Literal[
    "heart_rate",
    "spo2",
    "sleep",
    "stress",
    "ecg",
    "blood_pressure",
    "step_count",
    "menstrual_cycle",
    "vo2_max",
    "body_composition",
]


class SmartwatchHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed smartwatch metadata."""

    brands: list[str] = Field(default_factory=list)
    display_families: list[DisplayFamily] = Field(default_factory=list)
    strap_material_families: list[StrapMaterialFamily] = Field(default_factory=list)
    min_screen_size_inch: float | None = Field(default=None, ge=0.3, le=5)
    max_screen_size_inch: float | None = Field(default=None, ge=0.3, le=5)
    max_case_width_mm: float | None = Field(default=None, ge=10, le=100)
    max_weight_g: float | None = Field(default=None, gt=0, le=1000)
    wrist_circumference_cm: float | None = Field(default=None, ge=5, le=50)
    min_typical_battery_hours: float | None = Field(default=None, gt=0, le=24000)
    min_water_resistance_atm: float | None = Field(default=None, gt=0, le=100)
    call_requirement: CallRequirement | None = None
    requires_cellular: bool | None = None
    requires_gps: bool | None = None
    requires_notifications: bool | None = None
    requires_swimming: bool | None = None
    required_health_features: list[HealthFeature] = Field(default_factory=list)


class SmartwatchNeedProfile(BaseModel):
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    phone_platform: PhonePlatform | None = None
    usage_preferences: list[SmartwatchUse] = Field(default_factory=list)
    hard_constraints: SmartwatchHardConstraints = Field(
        default_factory=SmartwatchHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class SmartwatchCustomAnswer(BaseModel):
    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    phone_platform: PhonePlatform | None = None
    usage_preferences: list[SmartwatchUse] = Field(default_factory=list)
    hard_constraints: SmartwatchHardConstraints = Field(
        default_factory=SmartwatchHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
