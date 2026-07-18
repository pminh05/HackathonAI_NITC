"""Structured need-profile models for printer advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PrintPurpose = Literal[
    "mono_documents", "color_documents", "photo", "receipt_label", "general"
]
MonthlyVolumeSegment = Literal["light", "regular", "office", "high", "open"]
PrintTechnology = Literal["laser", "inkjet", "thermal"]
ColorMode = Literal["color", "monochrome"]
ConnectionTag = Literal["wifi", "wifi_direct", "lan", "usb", "bluetooth", "mobile_print"]
PaperSizeTag = Literal["a3", "a4", "a5", "a6", "b5", "f4", "letter", "legal"]


class PrinterHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``mayin`` metadata."""

    brands: list[str] = Field(default_factory=list)
    technologies: list[PrintTechnology] = Field(default_factory=list)
    color_modes: list[ColorMode] = Field(default_factory=list)
    min_print_speed_ppm: float | None = Field(default=None, gt=0, le=1000)
    required_connections: list[ConnectionTag] = Field(default_factory=list)
    required_paper_sizes: list[PaperSizeTag] = Field(default_factory=list)
    requires_duplex: bool | None = None
    max_width_mm: int | None = Field(default=None, gt=0, le=10000)
    max_height_mm: int | None = Field(default=None, gt=0, le=10000)
    max_depth_mm: int | None = Field(default=None, gt=0, le=10000)


class PrinterNeedProfile(BaseModel):
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    monthly_volume_segment: MonthlyVolumeSegment | None = None
    monthly_pages_estimate: int | None = Field(default=None, ge=1, le=10_000_000)
    usage_preferences: list[PrintPurpose] = Field(default_factory=list)
    hard_constraints: PrinterHardConstraints = Field(default_factory=PrinterHardConstraints)
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class PrinterCustomAnswer(BaseModel):
    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    monthly_volume_segment: MonthlyVolumeSegment | None = None
    monthly_pages_estimate: int | None = Field(default=None, ge=1, le=10_000_000)
    usage_preferences: list[PrintPurpose] = Field(default_factory=list)
    hard_constraints: PrinterHardConstraints = Field(default_factory=PrinterHardConstraints)
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
