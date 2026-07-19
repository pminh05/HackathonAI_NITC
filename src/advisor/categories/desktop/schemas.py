"""Structured need-profile models for desktop-computer advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DesktopUse = Literal[
    "office_study",
    "programming_multitasking",
    "gaming",
    "creative_content",
    "engineering_workstation",
    "general",
]
DesktopForm = Literal["all_in_one", "separate_unit", "flexible"]
CPUVendor = Literal["intel", "amd", "apple"]
OSFamily = Literal["windows", "macos", "linux", "freedos", "no_os"]
StorageType = Literal["nvme", "ssd", "hdd"]
GPUType = Literal["integrated", "discrete"]


class DesktopHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``maytinhdeban`` metadata."""

    brands: list[str] = Field(default_factory=list)
    cpu_vendors: list[CPUVendor] = Field(default_factory=list)
    os_families: list[OSFamily] = Field(default_factory=list)
    storage_types: list[StorageType] = Field(default_factory=list)
    gpu_types: list[GPUType] = Field(default_factory=list)
    min_ram_gb: int | None = Field(default=None, ge=1, le=1024)
    min_supported_ram_gb: int | None = Field(default=None, ge=1, le=2048)
    min_storage_gb: int | None = Field(default=None, ge=1, le=131072)
    min_screen_size_inch: float | None = Field(default=None, gt=0, le=100)
    max_screen_size_inch: float | None = Field(default=None, gt=0, le=100)
    requires_wifi: bool | None = None


class DesktopNeedProfile(BaseModel):
    """Customer needs consumed by the desktop advisory pipeline."""

    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    form_preference: DesktopForm | None = None
    usage_preferences: list[DesktopUse] = Field(default_factory=list)
    hard_constraints: DesktopHardConstraints = Field(
        default_factory=DesktopHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class DesktopCustomAnswer(BaseModel):
    """Validated interpretation of a desktop free-form clarification answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    form_preference: DesktopForm | None = None
    usage_preferences: list[DesktopUse] = Field(default_factory=list)
    hard_constraints: DesktopHardConstraints = Field(
        default_factory=DesktopHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
