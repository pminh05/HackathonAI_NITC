"""Structured need-profile models for computer-monitor advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PanelFamily = Literal["ips", "va", "tn", "oled"]
ScreenShape = Literal["flat", "curved"]
ScreenSizePreference = Literal["compact", "standard", "large", "ultrawide", "flexible"]
UsagePreference = Literal[
    "office_study",
    "programming_multitasking",
    "gaming",
    "creative_color",
    "entertainment",
    "general",
]
ConnectionTag = Literal[
    "hdmi",
    "displayport",
    "usb_c",
    "thunderbolt",
    "vga",
    "dvi",
    "usb_a",
    "ethernet",
    "audio_out",
]
FeatureTag = Literal[
    "freesync",
    "gsync",
    "adaptive_sync",
    "flicker_free",
    "low_blue_light",
    "anti_glare",
    "hdr",
    "height_adjust",
    "pivot",
    "swivel",
    "webcam",
    "smart_monitor",
]
ResponseMetric = Literal["gtg", "mprt", "prt"]


class MonitorHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``manhinhmaytinh`` metadata."""

    brands: list[str] = Field(default_factory=list)
    panel_families: list[PanelFamily] = Field(default_factory=list)
    screen_shapes: list[ScreenShape] = Field(default_factory=list)
    resolution_keys: list[str] = Field(default_factory=list)
    required_connections: list[ConnectionTag] = Field(default_factory=list)
    required_features: list[FeatureTag] = Field(default_factory=list)
    response_time_metrics: list[ResponseMetric] = Field(default_factory=list)
    min_screen_size_inch: float | None = Field(default=None, gt=0, le=100)
    max_screen_size_inch: float | None = Field(default=None, gt=0, le=100)
    min_resolution_width_px: int | None = Field(default=None, gt=0, le=20000)
    min_resolution_height_px: int | None = Field(default=None, gt=0, le=10000)
    max_response_time_ms: float | None = Field(default=None, ge=0, le=1000)
    min_brightness_nits: float | None = Field(default=None, gt=0, le=10000)
    min_srgb_coverage_pct: float | None = Field(default=None, ge=0, le=500)
    min_dci_p3_coverage_pct: float | None = Field(default=None, ge=0, le=500)
    requires_speakers: bool | None = None
    requires_vesa: bool | None = None
    requires_touch: bool | None = None
    max_width_mm: float | None = Field(default=None, gt=0, le=5000)


class MonitorNeedProfile(BaseModel):
    """Customer needs consumed by the monitor pipeline."""

    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    screen_size_preference: ScreenSizePreference | None = None
    preferred_screen_size_inch: float | None = Field(default=None, gt=0, le=100)
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: MonitorHardConstraints = Field(
        default_factory=MonitorHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class MonitorCustomAnswer(BaseModel):
    """Validated interpretation of a monitor free-form answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    screen_size_preference: ScreenSizePreference | None = None
    preferred_screen_size_inch: float | None = Field(default=None, gt=0, le=100)
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: MonitorHardConstraints = Field(
        default_factory=MonitorHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
