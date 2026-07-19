"""Unit tests for the Gemini -> FPT chat-model fallback."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from advisor.nodes import (
    FallbackChatModel,
    ProviderFallbackError,
    _add_fpt_structured_output_contract,
    _deterministic_turn_analysis,
    _message_text,
    create_advisor_chat_model,
)
from advisor.schemas import (
    ApplicationSettings,
    CategoryTransition,
    IntentLabel,
    ProfilePatch,
    RankingResult,
    TurnAction,
    TurnAnalysisResult,
)


class FakeRunnable:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.fallbacks: list[FakeRunnable] = []
        self.inputs: list[Any] = []

    def with_fallbacks(self, fallbacks: list[FakeRunnable]) -> FakeRunnable:
        self.fallbacks = fallbacks
        return self

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.inputs.append(input)
        if self.error is None:
            return self.result
        for fallback in self.fallbacks:
            try:
                return fallback.invoke(input, config=config, **kwargs)
            except Exception:
                continue
        raise self.error

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.invoke(input, config=config, **kwargs)

    def __or__(self, other: Any) -> FakeRunnablePipeline:
        return FakeRunnablePipeline(self, other)


class FakeRunnablePipeline(FakeRunnable):
    def __init__(self, first: FakeRunnable, second: Any) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        value = self.first.invoke(input, config=config, **kwargs)
        return self.second.invoke(value, config=config)


class FakeChatModel(FakeRunnable):
    def __init__(
        self,
        result: Any = None,
        error: Exception | None = None,
        structured: FakeRunnable | None = None,
    ) -> None:
        super().__init__(result=result, error=error)
        self.structured = structured
        self.structured_calls: list[tuple[type[Any], dict[str, Any]]] = []

    def with_structured_output(self, schema: type[Any], **kwargs: Any) -> FakeRunnable:
        self.structured_calls.append((schema, kwargs))
        assert self.structured is not None
        return self.structured


def test_plain_invoke_falls_back_when_primary_fails() -> None:
    model = FallbackChatModel(
        primary=FakeChatModel(error=RuntimeError("Gemini unavailable")),
        fallback=FakeChatModel(result="FPT response"),
    )

    assert model.invoke("hello") == "FPT response"


def test_invalid_primary_key_opens_circuit_and_uses_fallback_directly() -> None:
    class InvalidKeyModel(FakeChatModel):
        calls = 0

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            self.calls += 1
            raise RuntimeError("API_KEY_INVALID: API key not valid")

    primary = InvalidKeyModel()
    model = FallbackChatModel(
        primary=primary,
        fallback=FakeChatModel(result="FPT response"),
    )

    assert model.invoke("first") == "FPT response"
    assert model.invoke("second") == "FPT response"
    assert primary.calls == 1
    assert model._primary_is_disabled()


def test_factory_wraps_fpt_as_openai_compatible_fallback() -> None:
    settings = ApplicationSettings(
        _env_file=None,
        google_api_key="fake-google-key",
        fpt_api_key="fake-fpt-key",
    )

    model = create_advisor_chat_model(settings)

    assert isinstance(model, FallbackChatModel)
    assert type(model.primary).__name__ == "ChatGoogleGenerativeAI"
    assert type(model.fallback).__name__ == "ChatOpenAI"
    assert str(model.fallback.openai_api_base) == "https://mkp-api.fptcloud.com"
    assert model.fallback.model_name == "Qwen3.6-27B"
    assert model.fallback.tags == ["provider:fpt-ai-factory", "role:fallback"]
    assert model.fallback.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert model.fallback.metadata["thinking_enabled"] is False
    assert model.fallback.request_timeout == 30.0
    assert model.fallback.max_retries == 0
    assert model.fallback.disable_streaming is True


def test_factory_supports_fpt_without_a_google_key() -> None:
    settings = ApplicationSettings(
        _env_file=None,
        google_api_key=None,
        fpt_api_key="fake-fpt-key",
    )

    model = create_advisor_chat_model(settings)

    assert isinstance(model, FallbackChatModel)
    assert model.primary is None
    assert type(model.fallback).__name__ == "ChatOpenAI"
    assert model.fallback.model_name == "Qwen3.6-27B"


def test_structured_output_configures_both_providers_before_fallback() -> None:
    class ResultSchema(BaseModel):
        ok: bool

    primary = FakeChatModel(
        structured=FakeRunnable(error=ValueError("invalid Gemini JSON"))
    )
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content='{"ok": true}'),
                "parsed": {"ok": True},
                "parsing_error": None,
            }
        )
    )
    model = FallbackChatModel(primary=primary, fallback=fallback)

    runnable = model.with_structured_output(ResultSchema, method="json_schema")

    assert runnable.invoke("return JSON") == ResultSchema(ok=True)
    assert primary.structured_calls == [(ResultSchema, {"method": "json_schema"})]
    fallback_schema, fallback_kwargs = fallback.structured_calls[0]
    assert isinstance(fallback_schema, dict)
    assert fallback_schema["title"] == "ResultSchema"
    assert fallback_kwargs == {"method": "json_schema", "include_raw": True}
    fallback_input = fallback.structured.inputs[-1]
    assert isinstance(fallback_input[0], SystemMessage)
    assert "Chỉ trả đúng một JSON object" in fallback_input[0].content
    assert '"ok"' in fallback_input[0].content


def test_profile_patch_uses_fence_tolerant_json_schema_fallback() -> None:
    primary = FakeChatModel(
        structured=FakeRunnable(error=ValueError("invalid Gemini JSON"))
    )
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content="{}"),
                "parsed": {},
                "parsing_error": None,
            }
        )
    )
    model = FallbackChatModel(primary=primary, fallback=fallback)

    result = model.with_structured_output(ProfilePatch, method="json_schema").invoke(
        "extract patch"
    )

    assert result == ProfilePatch()
    fallback_schema, fallback_kwargs = fallback.structured_calls[0]
    assert isinstance(fallback_schema, dict)
    assert fallback_schema["title"] == "ProfilePatch"
    assert fallback_kwargs == {"method": "json_schema", "include_raw": True}


def test_turn_analysis_normalizes_qwen_aliases_without_function_calling() -> None:
    primary = FakeChatModel(
        structured=FakeRunnable(error=ValueError("Gemini unavailable"))
    )
    markdown_result = """- **category**: washing_machine
- **category_transition**: init
- **action**: more_options
- **scope**: recommendation
- **referenced_product_ids**: []
- **has_profile_update**: false"""
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content=markdown_result),
                "parsed": None,
                "parsing_error": OutputParserException("not JSON"),
            }
        )
    )
    model = FallbackChatModel(primary=primary, fallback=fallback)

    result = model.with_structured_output(
        TurnAnalysisResult, method="json_schema"
    ).invoke("analyze")

    assert result.category is IntentLabel.WASHING_MACHINE
    assert result.category_transition is CategoryTransition.NEW
    assert result.action is TurnAction.MORE_OPTIONS
    assert result.scope == "current_recommendations"
    fallback_schema, fallback_kwargs = fallback.structured_calls[0]
    assert isinstance(fallback_schema, dict)
    assert fallback_schema["title"] == "TurnAnalysisResult"
    assert fallback_kwargs == {"method": "json_schema", "include_raw": True}


def test_fpt_only_structured_output_uses_the_same_markdown_tolerant_adapter() -> None:
    markdown_result = """- **category**: dryer
- **category_transition**: inherit
- **action**: refine_needs
- **scope**: refine_needs
- **referenced_product_ids**: []
- **has_profile_update**: true"""
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content=markdown_result),
                "parsed": None,
                "parsing_error": OutputParserException("not JSON"),
            }
        )
    )
    model = FallbackChatModel(primary=None, fallback=fallback)

    result = model.with_structured_output(
        TurnAnalysisResult, method="json_schema"
    ).invoke("analyze")

    assert result.category is IntentLabel.DRYER
    assert result.category_transition is CategoryTransition.INHERIT
    assert result.action is TurnAction.REFINE_NEEDS
    assert result.scope == "unspecified"
    assert result.has_profile_update is True


def test_nested_markdown_ranking_is_parsed_and_validated() -> None:
    markdown_result = """- **selected_products**:
  - **product_id**: sku-1
    **reason**: Phù hợp nhu cầu
    **trade_off**: Giá cao hơn
  - **product_id**: sku-2
    **reason**: Tiết kiệm điện
    **trade_off**: Ít tính năng"""
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content=markdown_result),
                "parsed": None,
                "parsing_error": OutputParserException("not JSON"),
            }
        )
    )
    model = FallbackChatModel(primary=None, fallback=fallback)

    result = model.with_structured_output(
        RankingResult, method="json_schema"
    ).invoke("rank")

    assert [item.product_id for item in result.selected_products] == ["sku-1", "sku-2"]


def test_raw_tool_call_arguments_are_recovered_before_content_parsing() -> None:
    class ResultSchema(BaseModel):
        ok: bool

    raw = AIMessage(
        content="",
        tool_calls=[
            {"name": "ResultSchema", "args": {"ok": True}, "id": "call-1"}
        ],
    )
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": raw,
                "parsed": None,
                "parsing_error": OutputParserException("parser mismatch"),
            }
        )
    )

    result = FallbackChatModel(primary=None, fallback=fallback).with_structured_output(
        ResultSchema, method="json_schema"
    ).invoke("return tool args")

    assert result == ResultSchema(ok=True)


def test_custom_answer_aliases_and_empty_lists_are_normalized() -> None:
    class CustomResult(BaseModel):
        interpretation_status: Literal[
            "mapped", "custom_value", "partially_understood", "unresolved"
        ]
        raw_answer: str
        confidence: float = Field(ge=0, le=1)
        soft_preferences: list[str] = Field(default_factory=list)

    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content=""),
                "parsed": {
                    "status": "resolved",
                    "raw_answer": "Nhà tôi có 4 người",
                    "confidence": "75%",
                    "soft_preferences": {},
                },
                "parsing_error": None,
            }
        )
    )

    result = FallbackChatModel(primary=None, fallback=fallback).with_structured_output(
        CustomResult, method="json_schema"
    ).invoke("interpret")

    assert result.interpretation_status == "mapped"
    assert result.confidence == 0.75
    assert result.soft_preferences == []


def test_direct_profile_shape_is_recovered_as_profile_patch() -> None:
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content=""),
                "parsed": {
                    "product_family": "freezer",
                    "usage_preferences": ["bulk_frozen_storage"],
                    "hard_constraints": {"inverter": True},
                    "evidence": {"product_family": "tủ đông"},
                },
                "parsing_error": None,
            }
        )
    )

    result = FallbackChatModel(primary=None, fallback=fallback).with_structured_output(
        ProfilePatch, method="json_schema"
    ).invoke("extract")

    assert result.set == {
        "product_family": "freezer",
        "hard_constraints.inverter": True,
    }
    assert result.replace == {"usage_preferences": ["bulk_frozen_storage"]}
    assert result.evidence == {"product_family": "tủ đông"}


def test_numbered_markdown_ranking_is_recovered() -> None:
    prose = """**Lựa chọn:**
1. **sku-1** (5.000.000 VNĐ)
   - *Reason:* Phù hợp nhu cầu cơ bản.
   - *Trade-off:* Thiếu dữ liệu về công nghệ.

2. **sku-2** (10.000.000 VNĐ)
   - *Reason:* Cân bằng giá và tính năng.
   - *Trade-off:* Giá cao hơn sku-1."""
    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content=prose),
                "parsed": None,
                "parsing_error": OutputParserException("not JSON"),
            }
        )
    )

    result = FallbackChatModel(primary=None, fallback=fallback).with_structured_output(
        RankingResult, method="json_schema"
    ).invoke("rank")

    assert [item.product_id for item in result.selected_products] == ["sku-1", "sku-2"]
    assert result.selected_products[0].trade_off == "Thiếu dữ liệu về công nghệ."


def test_fenced_json_in_content_blocks_is_recovered() -> None:
    class ResultSchema(BaseModel):
        ok: bool

    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(
                    content=[{"type": "text", "text": "```json\n{\"ok\": true}\n```"}]
                ),
                "parsed": None,
                "parsing_error": OutputParserException("fenced JSON"),
            }
        )
    )

    result = FallbackChatModel(primary=None, fallback=fallback).with_structured_output(
        ResultSchema, method="json_schema"
    ).invoke("return JSON")

    assert result.ok is True


def test_invalid_fallback_output_retains_the_parser_error() -> None:
    class ResultSchema(BaseModel):
        ok: bool

    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content="This is not structured output"),
                "parsed": None,
                "parsing_error": OutputParserException("not JSON"),
            }
        )
    )

    runnable = FallbackChatModel(
        primary=None, fallback=fallback
    ).with_structured_output(ResultSchema, method="json_schema")

    try:
        runnable.invoke("return JSON")
    except ProviderFallbackError as exc:
        assert isinstance(exc.fallback, ValueError)
    else:
        raise AssertionError("Invalid fallback output must not bypass validation")


def test_async_structured_fallback_uses_the_same_adapter() -> None:
    class ResultSchema(BaseModel):
        ok: bool

    fallback = FakeChatModel(
        structured=FakeRunnable(
            result={
                "raw": AIMessage(content="```json\n{\"ok\": true}\n```"),
                "parsed": None,
                "parsing_error": OutputParserException("fenced JSON"),
            }
        )
    )
    runnable = FallbackChatModel(
        primary=None, fallback=fallback
    ).with_structured_output(ResultSchema, method="json_schema")

    result = asyncio.run(runnable.ainvoke("return JSON"))

    assert result == ResultSchema(ok=True)


def test_output_contract_keeps_the_guardrail_system_message_first() -> None:
    class ResultSchema(BaseModel):
        ok: bool

    trusted = SystemMessage(content="trusted guardrail")
    messages = _add_fpt_structured_output_contract(
        [trusted, HumanMessage(content="task data")], ResultSchema
    )

    assert messages[0] is trusted
    assert isinstance(messages[1], SystemMessage)
    assert "JSON Schema" in messages[1].content
    assert isinstance(messages[2], HumanMessage)


def test_deterministic_turn_fallback_handles_unaccented_category_and_inheritance() -> None:
    initial = _deterministic_turn_analysis("Tu van may giat", None)
    followup = _deterministic_turn_analysis("Nhà tôi có bốn người", "washing_machine")

    assert initial.category is IntentLabel.WASHING_MACHINE
    assert initial.category_transition is CategoryTransition.NEW
    assert followup.category is IntentLabel.WASHING_MACHINE
    assert followup.category_transition is CategoryTransition.INHERIT
    assert followup.action is TurnAction.REFINE_NEEDS


def test_plain_chat_text_parser_handles_openai_content_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "Phần một. "},
            {"type": "text", "text": "Phần hai."},
        ]
    )

    assert _message_text(message) == "Phần một. Phần hai."
