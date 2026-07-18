"""Structured need-profile models for micro-karaoke advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


UsageContext = Literal["home_family", "karaoke_room", "stage_event", "portable"]
ConnectionPreference = Literal["wired", "wireless", "open"]
MicrophoneType = Literal["wired", "wireless"]
WirelessBand = Literal["uhf", "2_4_ghz"]


class KaraokeMicrophoneHardConstraints(BaseModel):
    """Constraints backed by normalized, indexed ``microkaraoke`` metadata."""

    brands: list[str] = Field(default_factory=list)
    microphone_types: list[MicrophoneType] = Field(default_factory=list)
    wireless_bands: list[WirelessBand] = Field(default_factory=list)


class KaraokeMicrophoneNeedProfile(BaseModel):
    connection_preference: ConnectionPreference | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsageContext] = Field(default_factory=list)
    hard_constraints: KaraokeMicrophoneHardConstraints = Field(
        default_factory=KaraokeMicrophoneHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class KaraokeMicrophoneCustomAnswer(BaseModel):
    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    connection_preference: ConnectionPreference | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsageContext] = Field(default_factory=list)
    hard_constraints: KaraokeMicrophoneHardConstraints = Field(
        default_factory=KaraokeMicrophoneHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
