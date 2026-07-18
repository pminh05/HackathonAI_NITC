"""Application settings and structured contracts for the advisor graph."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """Environment-backed runtime settings.

    The repository is commonly run either from the workspace root or from the
    Python project directory, hence the two conventional dotenv locations.
    Real environment variables still take precedence over dotenv values.
    """

    app_env: str = "development"
    log_level: str = "INFO"
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash"
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_seconds: int = 60
    checkpoint_backend: Literal["sqlite", "postgres"] = "sqlite"
    checkpoint_db_path: Path = Path(".data/checkpoints.sqlite")
    supabase_database_url: SecretStr | None = None
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_cors_origins: str = "http://localhost:3000,http://localhost:5173"
    sse_heartbeat_seconds: float = 15.0

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.api_cors_origins.split(",") if item.strip()]


class IntentLabel(StrEnum):
    REFRIGERATOR = "Tủ Lạnh"
    AIR_CONDITIONER = "Máy lạnh"
    WASHING_MACHINE = "Máy giặt"
    DRYER = "Máy sấy quần áo"
    DISHWASHER = "Máy rửa chén"
    COOLER_FREEZER = "Tủ mát, tủ đông"
    WATER_HEATER = "Máy nước nóng"
    KARAOKE_MICROPHONE = "Micro karaoke"
    PHONE_RECORDING_MICROPHONE = "Micro thu âm điện thoại"
    SMARTWATCH = "Đồng hồ thông minh"
    DESKTOP = "Máy tính để bàn"
    MONITOR = "Màn hình máy tính"
    PRINTER = "Máy in"
    TABLET = "Máy tính bảng"
    OTHER = "Khác"


class IntentResult(BaseModel):
    label: IntentLabel


class TurnAction(StrEnum):
    DISCOVER = "discover"
    REFINE_NEEDS = "refine_needs"
    PRODUCT_DETAIL = "product_detail"
    COMPARE = "compare"
    EXPLAIN = "explain"
    MORE_OPTIONS = "more_options"
    SWITCH_CATEGORY = "switch_category"
    RESTART_CATEGORY = "restart_category"
    CONVERSATION = "conversation"


class CategoryTransition(StrEnum):
    INHERIT = "inherit"
    NEW = "new"
    SWITCH = "switch"


class ExecutionMode(StrEnum):
    REUSE = "reuse"
    RERANK = "rerank"
    RETRIEVE = "retrieve"


class TurnAnalysisResult(BaseModel):
    """One-call interpretation of a natural-language conversation turn."""

    category: IntentLabel
    category_transition: CategoryTransition = CategoryTransition.INHERIT
    switch_evidence: str | None = None
    action: TurnAction = TurnAction.DISCOVER
    scope: Literal["current_recommendations", "category", "unspecified"] = (
        "unspecified"
    )
    referenced_product_ids: list[str] = Field(default_factory=list, max_length=12)
    has_profile_update: bool = False
    direct_reply: str | None = None


class ProfilePatch(BaseModel):
    """Category-owned profile mutations extracted from one user turn.

    Paths are validated by the category patch applier. Keeping the operation
    buckets generic lets future category handlers expose their own paths without
    changing the graph contract.
    """

    set: dict[str, Any] = Field(default_factory=dict)
    replace: dict[str, Any] = Field(default_factory=dict)
    add: dict[str, list[Any]] = Field(default_factory=dict)
    remove: dict[str, list[Any]] = Field(default_factory=dict)
    clear: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class RefrigeratorHardConstraints(BaseModel):
    """Only constraints that are safe to translate to Qdrant filters."""

    brands: list[str] = Field(default_factory=list)
    styles: list[str] = Field(default_factory=list)
    min_capacity_lit: int | None = Field(default=None, ge=0)
    max_capacity_lit: int | None = Field(default=None, ge=0)
    max_width_cm: float | None = Field(default=None, gt=0)
    max_height_cm: float | None = Field(default=None, gt=0)
    max_depth_cm: float | None = Field(default=None, gt=0)
    required_features: list[Literal["inverter", "external_water", "automatic_mode"]] = (
        Field(default_factory=list)
    )


class RefrigeratorNeedExtraction(BaseModel):
    """Structured output used to update a refrigerator need profile."""

    household_size: int | None = Field(default=None, ge=1, le=30)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[
        Literal["daily_shopping", "weekly_storage", "frozen_storage", "energy_saving"]
    ] = Field(default_factory=list)
    hard_constraints: RefrigeratorHardConstraints = Field(
        default_factory=RefrigeratorHardConstraints
    )
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class ClarificationDecision(BaseModel):
    """LLM decision; canonical question contents come from category config."""

    sufficient: bool
    question_ids: list[str] = Field(default_factory=list, max_length=3)


class ClarificationOption(BaseModel):
    option_id: str
    label: str


class ClarificationQuestion(BaseModel):
    question_id: str
    question_type: Literal["explicit", "implicit"]
    question: str
    options: list[ClarificationOption]


class ClarificationAnswer(BaseModel):
    question_id: str
    option_id: str
    custom_answer: str | None = None

    @model_validator(mode="after")
    def require_text_for_other(self) -> ClarificationAnswer:
        if self.option_id == "other" and not (self.custom_answer or "").strip():
            raise ValueError("custom_answer is required when option_id is 'other'")
        return self


class ClarificationSubmission(BaseModel):
    answers: list[ClarificationAnswer] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def reject_duplicate_questions(self) -> ClarificationSubmission:
        ids = [answer.question_id for answer in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("Each clarification question may be answered only once")
        return self


class CustomAnswerInterpretation(BaseModel):
    """Safe, category-owned interpretation of a free-form option answer."""

    interpretation_status: Literal[
        "mapped", "custom_value", "partially_understood", "unresolved"
    ]
    raw_answer: str
    household_size: int | None = Field(default=None, ge=1, le=30)
    budget_max_vnd: int | None = Field(default=None, ge=0)
    budget_segment: Literal["premium", "open"] | None = None
    usage_preferences: list[
        Literal["daily_shopping", "weekly_storage", "frozen_storage", "energy_saving"]
    ] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    implicit_needs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SelectedProduct(BaseModel):
    product_id: str
    reason: str
    trade_off: str


class RankingResult(BaseModel):
    """Structured selection performed before the streamable response call."""

    selected_products: list[SelectedProduct] = Field(min_length=1, max_length=3)


class ProductCandidate(BaseModel):
    """Compact, JSON-safe product data supplied to the final LLM call."""

    model_config = ConfigDict(extra="allow")

    product_id: str
    name: str
    qdrant_score: float
    brand: str | None = None
    style: str | None = None
    effective_price_vnd: int | None = None
    original_price_vnd: int | None = None
    promotional_price_vnd: int | None = None
    capacity_lit: int | None = None
    suitable_for: str | None = None
    description: str | None = None


class StructuredPayload(BaseModel):
    """Backward-compatible generic container retained for extension points."""

    data: dict[str, Any] = Field(default_factory=dict)
