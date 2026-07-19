"""Structured need-profile models for recording-microphone advice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RecordingSetup = Literal[
    "iphone_lightning",
    "iphone_usb_c",
    "android_usb_c",
    "camera_3_5mm",
    "computer_usb",
    "open",
]
ProductType = Literal["wireless_recording", "podcast_livestream"]
CompatibilityTag = Literal[
    "ios", "ipados", "android", "camera", "macos", "windows", "playstation"
]
ConnectorType = Literal[
    "lightning", "usb_c", "3_5mm", "xlr", "micro_usb", "aux_in"
]
PickupPattern = Literal["omnidirectional", "supercardioid"]
WirelessBand = Literal["2_4_ghz"]
FeatureTag = Literal[
    "noise_reduction",
    "auto_connect",
    "mute",
    "voice_change",
    "magnetic_mount",
    "safety_track",
    "touch_control",
    "audio_settings",
    "shock_mount",
    "status_light",
    "digital_gain_limiter",
]
UsagePreference = Literal[
    "solo_content", "two_person_interview", "outdoor_mobile", "podcast_livestream"
]


class PhoneRecordingMicrophoneHardConstraints(BaseModel):
    """Only constraints backed by normalized and indexed microphone metadata."""

    brands: list[str] = Field(default_factory=list)
    product_types: list[ProductType] = Field(default_factory=list)
    required_compatibility_tags: list[CompatibilityTag] = Field(default_factory=list)
    connector_types: list[ConnectorType] = Field(default_factory=list)
    min_transmitter_count: int | None = Field(default=None, ge=1, le=16)
    min_runtime_hours: float | None = Field(default=None, gt=0, le=200)
    min_transmission_range_m: float | None = Field(default=None, gt=0, le=5000)
    pickup_patterns: list[PickupPattern] = Field(default_factory=list)
    wireless_bands: list[WirelessBand] = Field(default_factory=list)
    required_features: list[FeatureTag] = Field(default_factory=list)


class PhoneRecordingMicrophoneNeedProfile(BaseModel):
    recording_setup: RecordingSetup | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: PhoneRecordingMicrophoneHardConstraints = Field(
        default_factory=PhoneRecordingMicrophoneHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class PhoneRecordingMicrophoneCustomAnswer(BaseModel):
    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recording_setup: RecordingSetup | None = None
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[UsagePreference] = Field(default_factory=list)
    hard_constraints: PhoneRecordingMicrophoneHardConstraints = Field(
        default_factory=PhoneRecordingMicrophoneHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
