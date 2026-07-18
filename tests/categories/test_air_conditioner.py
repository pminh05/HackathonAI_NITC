"""Behavior and data-contract tests for the air-conditioner category."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from advisor.categories.air_conditioner import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.air_conditioner.filter_builder import (
    STANDARDIZED_METADATA_KEYS,
    build_filter,
)
from advisor.categories.air_conditioner.normalizer import normalize_candidate
from advisor.categories.air_conditioner.schemas import (
    AirConditionerCustomAnswer,
    AirConditionerNeedProfile,
)
from advisor.categories.air_conditioner.setup_indexes import ensure_payload_indexes
from advisor.categories.refrigerator import load_config as load_refrigerator_config
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import (
    IntentLabel,
    ProfilePatch,
    RankingResult,
    TurnAnalysisResult,
)


CHANGE_NAME_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_lanh/changeName.py"
)
KEY_MAPPING = CHANGE_NAME_MODULE["KEY_MAPPING"]
normalize_metadata = CHANGE_NAME_MODULE["normalize_metadata"]


class FakeStructuredModel:
    def __init__(self, owner: FakeLLM, schema: type[Any]) -> None:
        self.owner = owner
        self.schema = schema

    def invoke(self, prompt: str) -> Any:
        self.owner.calls.append((self.schema.__name__, prompt))
        return self.owner.response_for(self.schema, prompt)


class FakeLLM:
    def __init__(
        self,
        *,
        analyses: list[TurnAnalysisResult] | None = None,
        patches: list[ProfilePatch] | None = None,
        custom: AirConditionerCustomAnswer | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.custom = custom
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        category = "tủ lạnh" if "tư vấn tủ lạnh" in prompt.casefold() else "máy lạnh"
        return AIMessage(content=f"Mình đề xuất {category} thử nghiệm phù hợp nhu cầu.")

    def response_for(self, schema: type[Any], prompt: str) -> Any:
        if schema is TurnAnalysisResult:
            if self.analyses:
                return self.analyses.pop(0)
            return TurnAnalysisResult(
                category=IntentLabel.AIR_CONDITIONER,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if schema is ProfilePatch:
            return self.patches.pop(0) if self.patches else ProfilePatch()
        if schema is AirConditionerCustomAnswer and self.custom:
            return self.custom
        if schema is RankingResult:
            product_id = "sku-fridge" if "sku-fridge" in prompt else "sku-ac"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Phù hợp điều kiện đã nêu.",
                        "trade_off": "Một số thông số còn thiếu.",
                    }
                ]
            )
        raise AssertionError(f"No fake response for {schema}")


class FakeQdrant:
    def __init__(self) -> None:
        self.schemas = {
            "maylanh": {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in load_config()["payload_indexes"].items()
            },
            "tulanh": {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in load_refrigerator_config()["payload_indexes"].items()
            },
        }
        self.query_kwargs: dict[str, Any] | None = None
        self.query_count = 0
        self.created: list[tuple[str, str]] = []

    def get_collection(self, collection: str) -> Any:
        return SimpleNamespace(payload_schema=self.schemas[collection])

    def query_points(self, **kwargs: Any) -> Any:
        self.query_count += 1
        self.query_kwargs = kwargs
        if kwargs["collection_name"] == "tulanh":
            payload = {
                "product_id": "sku-fridge",
                "name": "Tủ lạnh thử nghiệm",
                "text": "Tủ lạnh 350 lít có inverter.",
                "image_path": "/public/tu_lanh.jpg",
                "metadata": {
                    "brand": "Test",
                    "Kiểu dáng chuẩn": "Ngăn đá trên",
                    "Giá gốc vnd": 18_000_000,
                    "Giá khuyến mãi vnd": 15_000_000,
                    "Dung tích sử dụng lít": 350,
                    "Có inverter": True,
                },
            }
        else:
            payload = air_conditioner_payload()
        return SimpleNamespace(
            points=[SimpleNamespace(id="point-1", score=0.91, payload=payload)]
        )

    def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema: Any, wait: bool
    ) -> None:
        del wait
        schema = field_schema.value
        self.created.append((field_name, schema))
        self.schemas[collection_name][field_name] = SimpleNamespace(
            data_type=SimpleNamespace(value=schema)
        )


def air_conditioner_payload() -> dict[str, Any]:
    return {
        "product_id": "sku-ac",
        "name": "Máy lạnh thử nghiệm",
        "text": "Máy lạnh inverter cho phòng 15 đến 20 mét vuông.",
        "image_path": "/public/may_lanh.jpg",
        "metadata": {
            "brand": "Daikin",
            "Giá gốc vnd": 14_000_000,
            "Giá khuyến mãi vnd": 12_000_000,
            "Diện tích min m2": 15,
            "Diện tích max m2": 20,
            "Thể tích min m3": 40,
            "Thể tích max m3": 60,
            "Loại máy": "Máy lạnh 1 chiều (chỉ làm lạnh)",
            "Loại Inverter": "Máy lạnh Inverter",
            "Loại Gas": "R-32",
            "Số sao năng lượng": 5,
            "Hiệu suất năng lượng": 5.21,
            "Độ ồn min dB": 22,
            "Độ ồn max dB": 48,
            "Công nghệ làm lạnh": "Powerful",
            "Công nghệ tiết kiệm điện": "Inverter",
            "Tiện ích": "Chế độ ngủ | Điều khiển bằng điện thoại, có Wi-Fi",
            "Bảo hành máy nén năm": 10,
        },
    }


def complete_air_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"room_area_m2": 18, "budget_max_vnd": 15_000_000},
        replace={"usage_preferences": ["energy_saving"]},
    )


def complete_refrigerator_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"household_size": 4, "budget_max_vnd": 20_000_000},
        replace={"usage_preferences": ["weekly_storage"]},
    )


def test_ingestion_normalizes_prices_and_metadata_types() -> None:
    metadata = normalize_metadata(
        {
            "giá gốc": "11690000.0",
            "giá khuyến mãi": "10,490,000",
            "gia_goc_vnd": 116_900_000,
            "gia_khuyen_mai_vnd": 104_900_000,
            "dien_tich_min_m2": 15.0,
            "hieu_suat_nang_luong": 5,
            "cong_suat_lanh_btu_h": 9000.0,
        }
    )
    assert metadata["Giá gốc vnd"] == 11_690_000
    assert metadata["Giá khuyến mãi vnd"] == 10_490_000
    assert metadata["Diện tích min m2"] == 15
    assert metadata["Hiệu suất năng lượng"] == 5.0
    assert metadata["Công suất lạnh BTU/h"] == 9000


def test_category_spec_and_config_match_ingestion_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = spec.config
    assert spec.name == config["category"] == "air_conditioner"
    assert config["collection"] == "maylanh"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert STANDARDIZED_METADATA_KEYS <= set(KEY_MAPPING.values()) | {
        "Loại máy",
        "Loại Inverter",
        "Loại Gas",
    }
    assert config["payload_indexes"]['metadata."Diện tích min m2"'] == "integer"


def test_profile_schema_and_missing_questions_are_category_specific() -> None:
    profile = AirConditionerNeedProfile(
        room_area_m2=18,
        budget_max_vnd=15_000_000,
        usage_preferences=["quiet_sleep"],
    ).model_dump(exclude_none=True)
    assert get_missing_profile_fields(profile) == []
    assert get_missing_profile_fields({}) == [
        "room_size",
        "budget",
        "usage_preferences",
    ]
    with pytest.raises(ValidationError):
        AirConditionerNeedProfile(room_area_m2=-1)


def test_filter_contains_only_indexed_hard_constraints() -> None:
    config = load_config()
    query_filter = build_filter(
        {
            "room_area_m2": 18,
            "budget_max_vnd": 15_000_000,
            "soft_preferences": ["wifi", "quiet_operation"],
            "hard_constraints": {
                "brands": ["Daikin"],
                "machine_types": ["one_way"],
                "inverter": True,
                "gas_types": ["r32"],
            },
        },
        config["payload_fields"],
    )
    dumped = query_filter.model_dump(mode="json", exclude_none=True)
    text = str(dumped)
    assert 'metadata."Diện tích min m2"' in text
    assert 'metadata."Giá khuyến mãi vnd"' in text
    assert "Máy lạnh 1 chiều (chỉ làm lạnh)" in text
    assert "Máy lạnh Inverter" in text
    assert "R-32" in text
    assert "wifi" not in text and "quiet_operation" not in text
    assert build_filter({"soft_preferences": ["wifi"]}) is None


def test_filter_matches_normalized_payload_in_local_qdrant() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="maylanh-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="maylanh-test",
        points=[
            models.PointStruct(id=1, vector=[1.0, 0.0], payload=air_conditioner_payload()),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    **air_conditioner_payload(),
                    "metadata": {
                        **air_conditioner_payload()["metadata"],
                        "Diện tích min m2": 30,
                        "Diện tích max m2": 40,
                    },
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="maylanh-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "room_area_m2": 18,
                "budget_max_vnd": 15_000_000,
                "hard_constraints": {"inverter": True, "gas_types": ["r32"]},
            }
        ),
        limit=2,
    )
    assert [point.id for point in result.points] == [1]


def test_candidate_normalizer_whitelists_air_conditioner_fields() -> None:
    candidate = normalize_candidate(
        SimpleNamespace(id="point-1", score=0.9, payload=air_conditioner_payload())
    )
    assert candidate["product_id"] == "sku-ac"
    assert candidate["effective_price_vnd"] == 12_000_000
    assert candidate["image_path"] == "/public/may_lanh.jpg"
    assert candidate["room_area_range_m2"] == {"min": 15.0, "max": 20.0}
    assert candidate["inverter_type"] == "Máy lạnh Inverter"
    assert candidate["cooling_capacity_btu_h"] is None
    assert candidate["compressor_warranty_years"] == 10.0
    assert "khuyến mãi quà" not in candidate


def test_profile_patch_operations_use_air_conditioner_allowlist() -> None:
    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "room_area_m2": 18,
            "usage_preferences": ["quiet_sleep", "energy_saving"],
            "hard_constraints": {"brands": ["LG"]},
        },
        ProfilePatch(
            set={"room_area_m2": 20, "household_size": 4},
            replace={"hard_constraints.brands": ["Daikin"]},
            remove={"usage_preferences": ["quiet_sleep"]},
            add={"usage_preferences": ["smart_control"]},
        ),
        spec,
    )
    assert profile["room_area_m2"] == 20
    assert profile["usage_preferences"] == ["energy_saving", "smart_control"]
    assert profile["hard_constraints"]["brands"] == ["Daikin"]
    assert "household_size" not in profile
    assert set(changed) == {
        "room_area_m2",
        "hard_constraints.brands",
        "usage_preferences",
    }


def test_index_setup_is_idempotent() -> None:
    qdrant = FakeQdrant()
    config = load_config()
    removed = 'metadata."Diện tích min m2"'
    del qdrant.schemas["maylanh"][removed]
    assert ensure_payload_indexes(
        qdrant, "maylanh", config["payload_indexes"], apply=False
    ) == {removed: "integer"}
    assert ensure_payload_indexes(
        qdrant, "maylanh", config["payload_indexes"], apply=True
    ) == {}
    assert qdrant.created == [(removed, "integer")]
    assert ensure_payload_indexes(
        qdrant, "maylanh", config["payload_indexes"], apply=True
    ) == {}


def test_prompts_and_no_match_text_do_not_leak_refrigerator_rules() -> None:
    spec = get_category_spec()
    extraction = spec.build_need_extraction_prompt("Phòng 18 m²", {})
    ranking = spec.build_ranking_prompt(
        {"need_profile": {}, "hard_constraints": {}, "candidates": []}
    )
    response = spec.build_response_prompt(
        {"need_profile": {}, "selected_products": []}
    )
    assert "máy lạnh" in extraction.casefold()
    assert "room_area_m2" in extraction
    assert "BTU" in extraction
    assert "diện tích" in ranking.casefold()
    assert "máy lạnh" in response.casefold()
    assert "tủ lạnh" not in (extraction + ranking + response).casefold()
    assert "máy lạnh" in no_match_answer({"room_area_m2": 18}).casefold()


def test_generic_air_conditioner_query_interrupts_and_resumes_to_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "air-conditioner-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi máy lạnh")]}, config
    )
    payload = interrupted["__interrupt__"][0].value
    assert payload["category"] == "air_conditioner"
    assert [question["question_id"] for question in payload["questions"]] == [
        "room_size",
        "budget",
        "usage_preferences",
    ]
    assert all(question["options"][-1]["option_id"] == "other" for question in payload["questions"])

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "room_size", "option_id": "15_20"},
                    {"question_id": "budget", "option_id": "10m_15m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "energy_saving",
                    },
                ]
            }
        ),
        config,
    )
    assert completed["control"]["stage"] == "completed"
    assert completed["need_profile"]["room_area_m2"] == 20
    assert completed["need_profile"]["budget_max_vnd"] == 15_000_000
    assert completed["ranking"]["selected_products"][0]["product_id"] == "sku-ac"
    assert qdrant.query_kwargs["collection_name"] == "maylanh"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_room_size_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=AirConditionerCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Phòng 18 mét vuông",
            room_area_m2=18,
            confidence=0.99,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "air-conditioner-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Tôi cần máy lạnh")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "room_size",
                        "option_id": "other",
                        "custom_answer": "Phòng 18 mét vuông",
                    },
                    {"question_id": "budget", "option_id": "10m_15m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "quiet_sleep",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["room_area_m2"] == 18
    assert completed["need_profile"]["custom_answers"]["room_size"]["raw_answer"] == "Phòng 18 mét vuông"


def test_switch_air_conditioner_to_refrigerator_and_back_restores_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.AIR_CONDITIONER,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="switch",
                switch_evidence="tủ lạnh",
                action="switch_category",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.AIR_CONDITIONER,
                category_transition="switch",
                switch_evidence="máy lạnh",
                action="switch_category",
            ),
        ],
        patches=[complete_air_patch(), complete_refrigerator_patch()],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "air-refrigerator-air"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy lạnh")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua tủ lạnh")]}, config
    )
    assert switched["conversation"]["active_category"] == "refrigerator"
    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại máy lạnh")]}, config
    )
    assert restored["conversation"]["active_category"] == "air_conditioner"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2

