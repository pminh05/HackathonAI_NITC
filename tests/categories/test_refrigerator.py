"""Behavior tests for the refrigerator MVP."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from qdrant_client import models

from advisor.categories.refrigerator import load_config
from advisor.categories.refrigerator.filter_builder import build_filter
from advisor.categories.refrigerator.setup_indexes import ensure_payload_indexes
from advisor.graph import build_graph
from advisor.schemas import (
    ClarificationDecision,
    CustomAnswerInterpretation,
    IntentLabel,
    IntentResult,
    RankingResult,
    RefrigeratorNeedExtraction,
)


class FakeStructuredModel:
    def __init__(self, owner: FakeLLM, schema: type[Any]) -> None:
        self.owner = owner
        self.schema = schema

    def invoke(self, prompt: str) -> Any:
        self.owner.calls.append((self.schema.__name__, prompt))
        return self.owner.response_for(self.schema)


class FakeLLM:
    def __init__(
        self,
        *,
        intent: IntentLabel = IntentLabel.REFRIGERATOR,
        extraction: RefrigeratorNeedExtraction | None = None,
        custom: CustomAnswerInterpretation | None = None,
    ) -> None:
        self.intent = intent
        self.extraction = extraction or RefrigeratorNeedExtraction()
        self.custom = custom
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        return AIMessage(
            content="Mình đề xuất mẫu Tủ lạnh thử nghiệm vì phù hợp nhu cầu hiện tại."
        )

    def response_for(self, schema: type[Any]) -> Any:
        if schema is IntentResult:
            return IntentResult(label=self.intent)
        if schema is RefrigeratorNeedExtraction:
            return self.extraction
        if schema is ClarificationDecision:
            return ClarificationDecision(
                sufficient=False,
                question_ids=["household_size", "budget", "usage_preferences"],
            )
        if schema is CustomAnswerInterpretation and self.custom:
            return self.custom
        if schema is RankingResult:
            return RankingResult(
                selected_products=[
                    {
                        "product_id": "sku-1",
                        "reason": "Phù hợp dung tích và ngân sách.",
                        "trade_off": "Ít tiện ích cao cấp hơn.",
                    }
                ],
            )
        raise AssertionError(f"No fake response for {schema}")


class FakeQdrant:
    def __init__(self) -> None:
        config = load_config()
        self.payload_schema = {
            field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
            for field, schema in config["payload_indexes"].items()
        }
        self.query_kwargs: dict[str, Any] | None = None
        self.created: list[tuple[str, str]] = []

    def get_collection(self, _: str) -> Any:
        return SimpleNamespace(payload_schema=self.payload_schema)

    def query_points(self, **kwargs: Any) -> Any:
        self.query_kwargs = kwargs
        point = SimpleNamespace(
            id="point-1",
            score=0.91,
            payload={
                "product_id": "sku-1",
                "name": "Tủ lạnh thử nghiệm",
                "text": "Dung tích 350 lít, có Inverter.",
                "metadata": {
                    "brand": "Test",
                    "kieu_dang_chuan": "Ngăn đá trên",
                    "gia_goc_vnd": 18_000_000,
                    "gia_khuyen_mai_vnd": 15_000_000,
                    "dung_tich_su_dung_lit": 350,
                    "Số người sử dụng": "3 - 4 người",
                    "co_inverter": True,
                },
            },
        )
        return SimpleNamespace(points=[point])

    def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema: Any, wait: bool
    ) -> None:
        del collection_name, wait
        schema = field_schema.value
        self.created.append((field_name, schema))
        self.payload_schema[field_name] = SimpleNamespace(
            data_type=SimpleNamespace(value=schema)
        )


def test_refrigerator_scaffold_files_exist() -> None:
    category_dir = Path(__file__).parents[2] / "src/advisor/categories/refrigerator"
    assert (category_dir / "config.yaml").is_file()
    assert (category_dir / "prompts.py").is_file()
    assert (category_dir / "filter_builder.py").is_file()
    assert (category_dir / "setup_indexes.py").is_file()


def test_filter_contains_budget_people_and_explicit_constraints() -> None:
    query_filter = build_filter(
        {
            "household_size": 4,
            "budget_max_vnd": 20_000_000,
            "hard_constraints": {
                "brands": ["Panasonic"],
                "styles": ["Ngăn đá dưới"],
                "max_width_cm": 70,
                "required_features": ["inverter"],
            },
        }
    )
    assert query_filter is not None
    dumped = query_filter.model_dump(mode="json", exclude_none=True)
    text = str(dumped)
    assert "metadata.gia_khuyen_mai_vnd" in text
    assert "metadata.so_nguoi_min" in text
    assert "metadata.brand" in text
    assert "metadata.ngang_cm" in text
    assert "metadata.co_inverter" in text


def test_generic_query_interrupts_once_and_resumes_to_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "generic-flow"}}

    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi một chiếc tủ lạnh")]},
        config,
    )
    assert "__interrupt__" in interrupted
    payload = interrupted["__interrupt__"][0].value
    assert payload["type"] == "clarification_required"
    assert [item["question_id"] for item in payload["questions"]] == [
        "household_size",
        "budget",
        "usage_preferences",
    ]
    assert all(
        item["options"][-1]["option_id"] == "other"
        for item in payload["questions"]
    )

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "household_size", "option_id": "three_four"},
                    {"question_id": "budget", "option_id": "10m_20m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "energy_saving",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["control"]["stage"] == "completed"
    assert completed["need_profile"]["household_size"] == 4
    assert completed["need_profile"]["budget_max_vnd"] == 20_000_000
    assert completed["ranking"]["selected_products"][0]["product_id"] == "sku-1"
    assert completed["response"]["answer"].startswith("Mình đề xuất")
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "tulanh"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_complete_profile_skips_interrupt() -> None:
    llm = FakeLLM(
        extraction=RefrigeratorNeedExtraction(
            household_size=4,
            budget_max_vnd=20_000_000,
            usage_preferences=["weekly_storage"],
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    completed = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Tủ lạnh cho 4 người, dưới 20 triệu, mua đồ cả tuần"
                )
            ]
        },
        {"configurable": {"thread_id": "complete-flow"}},
    )
    assert "__interrupt__" not in completed
    assert completed["control"]["stage"] == "completed"
    assert not any(name == "ClarificationDecision" for name, _ in llm.calls)


def test_other_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=CustomAnswerInterpretation(
            interpretation_status="custom_value",
            raw_answer="Nhà có 7 người",
            household_size=7,
            confidence=0.99,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "custom-flow"}}
    graph.invoke(
        {"messages": [HumanMessage(content="Tôi cần mua tủ lạnh")]}, config
    )
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "household_size",
                        "option_id": "other",
                        "custom_answer": "Nhà có 7 người",
                    },
                    {"question_id": "budget", "option_id": "20m_30m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "weekly_storage",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["household_size"] == 7
    assert completed["need_profile"]["custom_answers"]["household_size"][
        "raw_answer"
    ] == "Nhà có 7 người"


def test_non_refrigerator_intent_returns_placeholder_without_qdrant() -> None:
    llm = FakeLLM(intent=IntentLabel.WASHING_MACHINE)
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy giặt")]},
        {"configurable": {"thread_id": "placeholder"}},
    )
    assert "mới hỗ trợ tư vấn tủ lạnh" in result["response"]["answer"]
    assert qdrant.query_kwargs is None


def test_index_setup_is_idempotent() -> None:
    qdrant = FakeQdrant()
    config = load_config()
    removed_field = "metadata.gia_goc_vnd"
    del qdrant.payload_schema[removed_field]

    missing = ensure_payload_indexes(
        qdrant, "tulanh", config["payload_indexes"], apply=False
    )
    assert missing == {removed_field: "integer"}
    assert qdrant.created == []

    remaining = ensure_payload_indexes(
        qdrant, "tulanh", config["payload_indexes"], apply=True
    )
    assert remaining == {}
    assert qdrant.created == [(removed_field, "integer")]
    assert (
        ensure_payload_indexes(
            qdrant, "tulanh", config["payload_indexes"], apply=True
        )
        == {}
    )
    assert qdrant.created == [(removed_field, "integer")]


def test_other_requires_custom_text() -> None:
    graph = build_graph(llm=FakeLLM(), qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "invalid-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua tủ lạnh")]}, config)
    with pytest.raises(ValueError, match="custom_answer"):
        graph.invoke(
            Command(
                resume={
                    "answers": [
                        {"question_id": "household_size", "option_id": "other"},
                        {"question_id": "budget", "option_id": "10m_20m"},
                        {
                            "question_id": "usage_preferences",
                            "option_id": "weekly_storage",
                        },
                    ]
                }
            ),
            config,
        )
