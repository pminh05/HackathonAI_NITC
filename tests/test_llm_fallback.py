"""Unit tests for the Gemini -> FPT chat-model fallback."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from advisor.nodes import FallbackChatModel, create_advisor_chat_model
from advisor.schemas import ApplicationSettings, ProfilePatch


class FakeRunnable:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.fallbacks: list[FakeRunnable] = []

    def with_fallbacks(self, fallbacks: list[FakeRunnable]) -> FakeRunnable:
        self.fallbacks = fallbacks
        return self

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
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

    def with_structured_output(
        self, schema: type[Any], **kwargs: Any
    ) -> FakeRunnable:
        self.structured_calls.append((schema, kwargs))
        assert self.structured is not None
        return self.structured


def test_plain_invoke_falls_back_when_primary_fails() -> None:
    model = FallbackChatModel(
        primary=FakeChatModel(error=RuntimeError("Gemini unavailable")),
        fallback=FakeChatModel(result="FPT response"),
    )

    assert model.invoke("hello") == "FPT response"


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


def test_structured_output_configures_both_providers_before_fallback() -> None:
    class ResultSchema(BaseModel):
        ok: bool

    primary = FakeChatModel(
        structured=FakeRunnable(error=ValueError("invalid Gemini JSON"))
    )
    fallback = FakeChatModel(structured=FakeRunnable(result=ResultSchema(ok=True)))
    model = FallbackChatModel(primary=primary, fallback=fallback)

    runnable = model.with_structured_output(ResultSchema, method="json_schema")

    assert runnable.invoke("return JSON") == ResultSchema(ok=True)
    assert primary.structured_calls == [
        (ResultSchema, {"method": "json_schema"})
    ]
    fallback_schema, fallback_kwargs = fallback.structured_calls[0]
    assert fallback_schema is ResultSchema
    assert fallback_kwargs == {"method": "function_calling"}


def test_profile_patch_uses_fence_tolerant_json_schema_fallback() -> None:
    primary = FakeChatModel(
        structured=FakeRunnable(error=ValueError("invalid Gemini JSON"))
    )
    fallback = FakeChatModel(structured=FakeRunnable(result={}))
    model = FallbackChatModel(primary=primary, fallback=fallback)

    result = model.with_structured_output(
        ProfilePatch, method="json_schema"
    ).invoke("extract patch")

    assert result == ProfilePatch()
    fallback_schema, fallback_kwargs = fallback.structured_calls[0]
    assert isinstance(fallback_schema, dict)
    assert fallback_schema["title"] == "ProfilePatch"
    assert fallback_kwargs == {"method": "json_schema"}
