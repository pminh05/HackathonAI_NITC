"""Behavior tests for the refrigerator MVP."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from qdrant_client import QdrantClient, models

from advisor.categories.refrigerator import load_config
from advisor.categories.refrigerator.filter_builder import (
    STANDARDIZED_METADATA_KEYS,
    build_filter,
)
from advisor.categories.refrigerator.setup_indexes import ensure_payload_indexes
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.retrieval.qdrant import normalize_candidate
from advisor.schemas import (
    ClarificationDecision,
    CustomAnswerInterpretation,
    IntentLabel,
    IntentResult,
    ProfilePatch,
    RankingResult,
    RefrigeratorNeedExtraction,
    TurnAnalysisResult,
)


CHANGE_NAME_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/tu_lanh/changeName.py"
)
KEY_MAPPING = CHANGE_NAME_MODULE["KEY_MAPPING"]
normalize_metadata = CHANGE_NAME_MODULE["normalize_metadata"]


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
        analyses: list[TurnAnalysisResult] | None = None,
        patches: list[ProfilePatch] | None = None,
        rankings: list[RankingResult] | None = None,
    ) -> None:
        self.intent = intent
        self.extraction = extraction or RefrigeratorNeedExtraction()
        self.custom = custom
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.rankings = list(rankings or [])
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        return AIMessage(
            content="Mình đề xuất mẫu Tủ lạnh thử nghiệm vì phù hợp nhu cầu hiện tại."
        )

    def response_for(self, schema: type[Any]) -> Any:
        if schema is TurnAnalysisResult:
            if self.analyses:
                return self.analyses.pop(0)
            return TurnAnalysisResult(
                category=self.intent,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if schema is IntentResult:
            return IntentResult(label=self.intent)
        if schema is ProfilePatch:
            if self.patches:
                return self.patches.pop(0)
            values = self.extraction.model_dump(exclude_none=True)
            patch = ProfilePatch()
            for key in ("household_size", "budget_max_vnd", "budget_segment"):
                if values.get(key) is not None:
                    patch.set[key] = values[key]
            for key in ("usage_preferences", "soft_preferences", "implicit_needs"):
                if values.get(key):
                    patch.replace[key] = values[key]
            for key, value in (values.get("hard_constraints") or {}).items():
                if value not in (None, []):
                    patch.set[f"hard_constraints.{key}"] = value
            patch.evidence = values.get("evidence") or {}
            return patch
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
            if self.rankings:
                return self.rankings.pop(0)
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
    def __init__(self, product_ids: list[str] | None = None) -> None:
        config = load_config()
        self.payload_schema = {
            field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
            for field, schema in config["payload_indexes"].items()
        }
        self.query_kwargs: dict[str, Any] | None = None
        self.created: list[tuple[str, str]] = []
        self.query_count = 0
        self.product_ids = product_ids or ["sku-1"]

    def get_collection(self, _: str) -> Any:
        return SimpleNamespace(payload_schema=self.payload_schema)

    def query_points(self, **kwargs: Any) -> Any:
        self.query_count += 1
        self.query_kwargs = kwargs
        points = [
            SimpleNamespace(
                id=f"point-{index}",
                score=0.91 - index * 0.01,
                payload={
                    "product_id": product_id,
                    "name": f"Tủ lạnh thử nghiệm {index}",
                    "text": "Dung tích 350 lít, có Inverter.",
                    "metadata": {
                        "brand": "Test",
                        "Kiểu dáng chuẩn": "Ngăn đá trên",
                        "Giá gốc vnd": 18_000_000,
                        "Giá khuyến mãi vnd": 15_000_000,
                        "Dung tích sử dụng lít": 350,
                        "Số người sử dụng": "3 - 4 người",
                        "Có inverter": True,
                    },
                },
            )
            for index, product_id in enumerate(self.product_ids, start=1)
        ]
        return SimpleNamespace(points=points)

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
    assert 'metadata."Giá khuyến mãi vnd"' in text
    assert 'metadata."Số người tối thiểu"' in text
    assert "metadata.brand" in text
    assert "Panasonic" in text
    assert 'metadata."Ngang cm"' in text
    assert 'metadata."Có inverter"' in text


def _metadata_key(path: str) -> str:
    prefix = 'metadata."'
    assert path.startswith(prefix) and path.endswith('"')
    return path[len(prefix) : -1]


def test_filter_config_only_uses_standardized_tulanh_metadata_keys() -> None:
    config = load_config()
    fields = config["payload_fields"].values()
    indexes = config["payload_indexes"]

    assert STANDARDIZED_METADATA_KEYS == set(KEY_MAPPING.values())
    quoted_fields = [path for path in fields if path != "metadata.brand"]
    quoted_indexes = [path for path in indexes if path != "metadata.brand"]
    assert {_metadata_key(path) for path in quoted_fields} <= STANDARDIZED_METADATA_KEYS
    assert {_metadata_key(path) for path in quoted_indexes} <= STANDARDIZED_METADATA_KEYS
    assert "metadata.brand" in fields
    assert "metadata.brand" in indexes
    assert set(fields) <= set(indexes)
    assert indexes['metadata."Ngang cm"'] == "float"
    assert indexes['metadata."Cao cm"'] == "float"
    assert indexes['metadata."Sâu cm"'] == "float"


def test_required_brand_is_a_real_indexed_hard_filter() -> None:
    config = load_config()

    query_filter = build_filter(
        {"hard_constraints": {"brands": ["LG", "Samsung"]}},
        config["payload_fields"],
    )

    assert query_filter is not None
    serialized = query_filter.model_dump(mode="json", exclude_none=True)
    assert serialized["must"][0]["key"] == "metadata.brand"
    assert serialized["must"][0]["match"]["any"] == ["LG", "Samsung"]


def test_ingestion_normalizes_dimension_values_as_float() -> None:
    metadata = normalize_metadata(
        {
            "ngang_cm": 59,
            "cao_cm": 170.5,
            "sau_cm": 66,
            "dung_tich_su_dung_lit": 350,
        }
    )

    assert metadata == {
        "Ngang cm": 59.0,
        "Cao cm": 170.5,
        "Sâu cm": 66.0,
        "Dung tích sử dụng lít": 350,
    }
    assert all(
        isinstance(metadata[key], float) for key in ("Ngang cm", "Cao cm", "Sâu cm")
    )


def test_filter_matches_vietnamese_metadata_in_local_qdrant() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="tulanh-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="tulanh-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        "Kiểu dáng chuẩn": "Ngăn đá dưới",
                        "Giá gốc vnd": 18_000_000,
                        "Giá khuyến mãi vnd": 15_000_000,
                        "Dung tích sử dụng lít": 350,
                        "Số người tối thiểu": 3,
                        "Số người tối đa": 4,
                        "Ngang cm": 59.6,
                        "Cao cm": 170.5,
                        "Sâu cm": 66.2,
                        "Có inverter": True,
                    }
                },
            )
        ],
        wait=True,
    )
    query_filter = build_filter(
        {
            "household_size": 4,
            "budget_max_vnd": 20_000_000,
            "hard_constraints": {
                "styles": ["Ngăn đá dưới"],
                "min_capacity_lit": 300,
                "max_width_cm": 60,
                "required_features": ["inverter"],
            },
        }
    )

    result = client.query_points(
        collection_name="tulanh-test",
        query=[1.0, 0.0],
        query_filter=query_filter,
        limit=1,
    )

    assert [point.id for point in result.points] == [1]


def test_candidate_normalization_reads_vietnamese_metadata() -> None:
    candidate = normalize_candidate(
        SimpleNamespace(
            id="point-1",
            score=0.9,
            payload={
                "product_id": "sku-1",
                "name": "Tủ lạnh thử nghiệm",
                "text": "Mô tả sản phẩm.",
                "image_path": "/catalog/public/tu_lanh.jpg",
                "metadata": {
                    "brand": "Test",
                    "Kiểu dáng chuẩn": "Ngăn đá dưới",
                    "Giá gốc vnd": 18_000_000,
                    "Giá khuyến mãi vnd": 15_000_000,
                    "Dung tích sử dụng lít": 350,
                    "Dung tích ngăn đá lít": 92,
                    "Điện năng kWh năm": 381,
                    "Số người sử dụng": "3 - 4 người",
                    "Có inverter": True,
                    "Có lấy nước ngoài": True,
                    "Có chế độ tự động": False,
                    "Ngang cm": 59.6,
                    "Cao cm": 170.5,
                    "Sâu cm": 66.2,
                },
            },
        )
    )

    assert candidate["style"] == "Ngăn đá dưới"
    assert candidate["effective_price_vnd"] == 15_000_000
    assert candidate["image_path"] == "/catalog/public/tu_lanh.jpg"
    assert candidate["capacity_lit"] == 350
    assert candidate["freezer_capacity_lit"] == 92
    assert candidate["annual_energy_kwh"] == 381
    assert candidate["inverter"] is True
    assert candidate["external_water"] is True
    assert candidate["automatic_mode"] is False
    assert candidate["dimensions_cm"] == {
        "width": 59.6,
        "height": 170.5,
        "depth": 66.2,
    }


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
    assert all(item["options"][-1]["option_id"] == "other" for item in payload["questions"])
    assert [name for name, _ in llm.calls] == [
        "TurnAnalysisResult",
        "ProfilePatch",
    ]

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
    assert [name for name, _ in llm.calls] == [
        "TurnAnalysisResult",
        "ProfilePatch",
        "RankingResult",
        "PlainResponse",
    ]


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


def test_unimplemented_intent_returns_placeholder_without_qdrant() -> None:
    llm = FakeLLM(intent=IntentLabel.DISHWASHER)
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy rửa chén")]},
        {"configurable": {"thread_id": "placeholder"}},
    )
    assert "máy sấy quần áo" in result["response"]["answer"]
    assert qdrant.query_kwargs is None


def test_index_setup_is_idempotent() -> None:
    qdrant = FakeQdrant()
    config = load_config()
    removed_field = 'metadata."Ngang cm"'
    del qdrant.payload_schema[removed_field]

    missing = ensure_payload_indexes(
        qdrant, "tulanh", config["payload_indexes"], apply=False
    )
    assert missing == {removed_field: "float"}
    assert qdrant.created == []

    remaining = ensure_payload_indexes(
        qdrant, "tulanh", config["payload_indexes"], apply=True
    )
    assert remaining == {}
    assert qdrant.created == [(removed_field, "float")]
    assert (
        ensure_payload_indexes(
            qdrant, "tulanh", config["payload_indexes"], apply=True
        )
        == {}
    )
    assert qdrant.created == [(removed_field, "float")]


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


def _complete_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"household_size": 4, "budget_max_vnd": 20_000_000},
        replace={"usage_preferences": ["weekly_storage"]},
    )


def test_follow_up_inherits_category_and_reuses_recommendations() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.OTHER,
                category_transition="inherit",
                action="compare",
                scope="current_recommendations",
                referenced_product_ids=["sku-1"],
            ),
        ],
        patches=[_complete_patch()],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "follow-up-reuse"}}

    graph.invoke({"messages": [HumanMessage(content="Tư vấn tủ lạnh")]}, config)
    calls_before = len(llm.calls)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Mẫu nào tiết kiệm điện nhất?")]},
        config,
    )

    assert result["conversation"]["active_category"] == "refrigerator"
    assert result["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 1
    assert [name for name, _ in llm.calls[calls_before:]] == [
        "TurnAnalysisResult",
        "PlainResponse",
    ]
    assert [message.type for message in result["messages"]] == [
        "human",
        "ai",
        "human",
        "ai",
    ]


def test_switching_category_and_returning_restores_refrigerator_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.DISHWASHER,
                category_transition="switch",
                switch_evidence="máy rửa chén",
                action="switch_category",
            ),
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="switch",
                switch_evidence="tủ lạnh",
                action="switch_category",
            ),
        ],
        patches=[_complete_patch()],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "category-return"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn tủ lạnh")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua máy rửa chén")]}, config
    )
    assert switched["conversation"]["active_category"] == "dishwasher"

    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại tủ lạnh")]}, config
    )
    assert restored["conversation"]["active_category"] == "refrigerator"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 1


def test_full_form_other_answer_does_not_trigger_a_second_clarification() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
        ],
        patches=[ProfilePatch()],
        custom=CustomAnswerInterpretation(
            interpretation_status="unresolved",
            raw_answer="Số người sử dụng không cố định",
            confidence=0,
        ),
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "full-form-clarification"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tôi cần tủ lạnh")]}, config
    )
    assert len(first["__interrupt__"][0].value["questions"]) == 3
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "household_size",
                        "option_id": "other",
                        "custom_answer": "Số người sử dụng không cố định",
                    },
                    {"question_id": "budget", "option_id": "10m_20m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "energy_saving",
                    }
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["budget_max_vnd"] == 20_000_000
    assert completed["need_profile"]["custom_answers"]["household_size"][
        "status"
    ] == "unresolved"
    assert sum(name == "TurnAnalysisResult" for name, _ in llm.calls) == 1
    assert not any(name == "ClarificationDecision" for name, _ in llm.calls)


def test_profile_patch_replaces_removes_and_clears_values() -> None:
    profile, changed = apply_profile_patch(
        {
            "budget_max_vnd": 20_000_000,
            "usage_preferences": ["weekly_storage"],
            "hard_constraints": {"brands": ["Panasonic"]},
        },
        ProfilePatch(
            replace={"hard_constraints.brands": ["LG"]},
            remove={"usage_preferences": ["weekly_storage"]},
            add={"usage_preferences": ["energy_saving"]},
            clear=["budget_max_vnd"],
        ),
    )
    assert profile["hard_constraints"]["brands"] == ["LG"]
    assert profile["usage_preferences"] == ["energy_saving"]
    assert "budget_max_vnd" not in profile
    assert set(changed) == {
        "hard_constraints.brands",
        "usage_preferences",
        "budget_max_vnd",
    }


def test_more_options_reranks_unseen_cached_candidates_without_qdrant() -> None:
    first_ranking = RankingResult(
        selected_products=[
            {"product_id": "sku-1", "reason": "Phù hợp.", "trade_off": "A."}
        ]
    )
    second_ranking = RankingResult(
        selected_products=[
            {"product_id": "sku-2", "reason": "Lựa chọn khác.", "trade_off": "B."}
        ]
    )
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.OTHER,
                category_transition="inherit",
                action="more_options",
                scope="current_recommendations",
            ),
        ],
        patches=[_complete_patch()],
        rankings=[first_ranking, second_ranking],
    )
    qdrant = FakeQdrant(product_ids=["sku-1", "sku-2", "sku-3", "sku-4"])
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "cached-more-options"}}
    graph.invoke({"messages": [HumanMessage(content="Tư vấn tủ lạnh")]}, config)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Còn mẫu khác không?")]}, config
    )

    assert result["conversation"]["execution_mode"] == "rerank"
    assert result["ranking"]["selected_products"][0]["product_id"] == "sku-2"
    assert qdrant.query_count == 1


def test_unverified_topic_switch_is_ignored_and_budget_change_retrieves() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.WASHING_MACHINE,
                category_transition="switch",
                switch_evidence="máy giặt",
                action="refine_needs",
                has_profile_update=True,
            ),
        ],
        patches=[_complete_patch(), ProfilePatch(set={"budget_max_vnd": 15_000_000})],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "invalid-switch-hard-change"}}
    graph.invoke({"messages": [HumanMessage(content="Tư vấn tủ lạnh")]}, config)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Giảm ngân sách còn 15 triệu")]},
        config,
    )

    assert result["conversation"]["active_category"] == "refrigerator"
    assert result["conversation"]["execution_mode"] == "retrieve"
    assert result["need_profile"]["budget_max_vnd"] == 15_000_000
    assert qdrant.query_count == 2
