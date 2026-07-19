"""Structured need-profile models for tablet advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TabletUse = Literal[
    "study_work", "entertainment", "gaming", "drawing_notes", "children", "general"
]
ConnectivitySegment = Literal["wifi_only", "cellular_4g", "cellular_5g", "flexible"]
OSFamily = Literal["android", "ipados", "harmonyos"]
DisplayFamily = Literal["ips_lcd", "oled_amoled", "mini_led", "tft_lcd", "lcd", "other"]


class TabletHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``maytinhbang`` metadata."""

    brands: list[str] = Field(default_factory=list)
    os_families: list[OSFamily] = Field(default_factory=list)
    display_families: list[DisplayFamily] = Field(default_factory=list)
    min_ram_gb: int | None = Field(default=None, ge=1, le=128)
    min_storage_gb: int | None = Field(default=None, ge=1, le=8192)
    min_screen_size_inch: float | None = Field(default=None, gt=0, le=30)
    max_screen_size_inch: float | None = Field(default=None, gt=0, le=30)
    max_weight_g: int | None = Field(default=None, gt=0, le=10000)
    requires_calls: bool | None = None
    requires_memory_card: bool | None = None


class TabletNeedProfile(BaseModel):
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    connectivity_segment: ConnectivitySegment | None = None
    usage_preferences: list[TabletUse] = Field(default_factory=list)
    hard_constraints: TabletHardConstraints = Field(default_factory=TabletHardConstraints)
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class TabletCustomAnswer(BaseModel):
    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    connectivity_segment: ConnectivitySegment | None = None
    usage_preferences: list[TabletUse] = Field(default_factory=list)
    hard_constraints: TabletHardConstraints = Field(default_factory=TabletHardConstraints)
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
