"""Behavior and data-contract tests for the water-heater category."""

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

from advisor.categories.refrigerator import load_config as load_refrigerator_config
from advisor.categories.water_heater import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.water_heater.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.water_heater.normalizer import normalize_candidate
from advisor.categories.water_heater.schemas import (
    WaterHeaterCustomAnswer,
    WaterHeaterNeedProfile,
)
from advisor.categories.water_heater.setup_indexes import ensure_payload_indexes
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


INGESTION_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_nuoc_nong/changeName.py"
)
normalize_metadata = INGESTION_MODULE["normalize_metadata"]
normalize_water_pressure = INGESTION_MODULE["normalize_water_pressure"]
QDRANT_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_nuoc_nong/qdrant.py"
)


class FakeStructuredModel:
    def __init__(self, owner: FakeLLM, schema: type[Any]) -> None:
        self.owner = owner
        self.schema = schema

    def invoke(self, prompt: str) -> Any:
        self.owner.calls.append((self.schema.__name__, prompt))
        if self.schema is TurnAnalysisResult:
            if self.owner.analyses:
                return self.owner.analyses.pop(0)
            return TurnAnalysisResult(
                category=IntentLabel.WATER_HEATER,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            product_id = "wh-1" if "wh-1" in prompt else "sku-rf"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Phù hợp loại máy và nguồn nước đã xác nhận.",
                        "trade_off": "Một số điều kiện lắp đặt cần kiểm tra thực tế.",
                    }
                ]
            )
        if self.schema is WaterHeaterCustomAnswer:
            return self.owner.custom or WaterHeaterCustomAnswer(
                interpretation_status="unresolved",
                raw_answer="không rõ",
                confidence=0,
            )
        raise AssertionError(self.schema)


class FakeLLM:
    def __init__(
        self,
        *,
        analyses: list[TurnAnalysisResult] | None = None,
        patches: list[ProfilePatch] | None = None,
        custom: WaterHeaterCustomAnswer | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.custom = custom
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        return AIMessage(
            content="Mình đề xuất mẫu máy nước nóng thử nghiệm theo nhu cầu của bạn."
        )


class FakeQdrant:
    def __init__(self) -> None:
        self.payload_schemas = {
            config["collection"]: {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in config["payload_indexes"].items()
            }
            for config in (load_config(), load_refrigerator_config())
        }
        self.created: list[tuple[str, str]] = []
        self.query_kwargs: dict[str, Any] | None = None
        self.query_count = 0

    def get_collection(self, collection: str) -> Any:
        return SimpleNamespace(payload_schema=self.payload_schemas[collection])

    def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema: Any, **_: Any
    ) -> None:
        value = getattr(field_schema, "value", str(field_schema))
        self.payload_schemas[collection_name][field_name] = SimpleNamespace(
            data_type=SimpleNamespace(value=value)
        )
        self.created.append((field_name, str(value)))

    def query_points(self, **kwargs: Any) -> Any:
        self.query_count += 1
        self.query_kwargs = kwargs
        if kwargs["collection_name"] == "tulanh":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="point-rf",
                        score=0.9,
                        payload={
                            "product_id": "sku-rf",
                            "name": "Tủ lạnh thử nghiệm",
                            "text": "Tủ lạnh 350 lít.",
                            "image_path": "/public/tu_lanh.jpg",
                            "metadata": {
                                "brand": "Test",
                                "Giá gốc vnd": 18_000_000,
                                "Giá khuyến mãi vnd": 15_000_000,
                                "Dung tích sử dụng lít": 350,
                            },
                        },
                    )
                ]
            )
        return SimpleNamespace(points=[water_heater_point()])


def water_heater_point() -> Any:
    return SimpleNamespace(
        id="point-wh-1",
        score=0.93,
        payload={
            "product_id": "wh-1",
            "name": "Máy nước nóng Stiebel Eltron Test",
            "text": "Máy trực tiếp 4500 W có bơm trợ lực và ELCB.",
            "image_path": "/public/may_nuoc_nong.jpg",
            "metadata": {
                "category_scope": "water_heater",
                "brand": "Stiebel Eltron",
                "model_code": "WH Test",
                "product_type": "direct",
                "Loại máy": "Làm nóng trực tiếp",
                "original_price_vnd": 4_790_000,
                "promotional_price_vnd": 3_990_000,
                "capacity_l": 0,
                "power_w": 4500,
                "heating_time_min_minutes": 0.0,
                "heating_time_max_minutes": 0.0,
                "max_temperature_c": 55,
                "has_booster_pump": True,
                "includes_shower": True,
                "water_pressure_min_mpa": 0.015,
                "water_pressure_max_mpa": 0.5,
                "ip_rating": "IP25",
                "safety_tags": ["elcb", "flow_sensor", "waterproof"],
                "feature_tags": ["low_pressure_compatible"],
                "Tính năng an toàn": "ELCB | Cảm biến lưu lượng nước",
                "Tùy chỉnh nhiệt độ": "Chỉnh nhiệt độ vô cấp",
                "Tiện ích": "Hoạt động với áp lực nước thấp",
                "Vòi sen": "Có kèm theo vòi sen 5 chế độ phun",
                "people_min": 2,
                "people_max": 4,
                "width_cm": 21.0,
                "height_cm": 34.0,
                "depth_cm": 7.0,
                "weight_kg": 2.0,
                "Sản xuất tại": "Thái Lan",
            },
        },
    )


def test_ingestion_normalizes_faulty_price_pressure_type_and_safety() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Stiebel Eltron",
            "Loại máy": "Làm nóng trực tiếp",
            "giá gốc": "4790000.0",
            "giá khuyến mãi": "3917455.0",
            "gia_goc_vnd": 47_900_000,
            "gia_khuyen_mai_vnd": 39_174_550,
            "dung_tich_lit": 0,
            "cong_suat_w": 4500,
            "co_bom_tro_luc": True,
            "co_kem_voi_sen": True,
            "Áp lực nước hoạt động": "Tối thiểu 0.015 Mpa - Tối đa 0.5 Mpa",
            "chi_so_chong_nuoc_ip": "IP25",
            "Tính năng an toàn": (
                "Cầu dao chống rò điện ELCB | Cảm biến lưu lượng nước | "
                "Vỏ chống thấm nước IP25"
            ),
        }
    )
    assert metadata["category_scope"] == "water_heater"
    assert metadata["product_type"] == "direct"
    assert metadata["original_price_vnd"] == 4_790_000
    assert metadata["promotional_price_vnd"] == 3_917_455
    assert "gia_goc_vnd" not in metadata
    assert "gia_khuyen_mai_vnd" not in metadata
    assert metadata["water_pressure_min_mpa"] == 0.015
    assert metadata["water_pressure_max_mpa"] == 0.5
    assert metadata["has_booster_pump"] is True
    assert metadata["safety_tags"] == ["elcb", "flow_sensor", "waterproof"]
    assert normalize_water_pressure(
        "Tối thiểu 0.1 Bar - Tối đa 6.0 Bar"
    ) == (0.01, 0.6, [])
    assert normalize_water_pressure(
        "Tối thiểu 30 kPa - Tối đa 300 kPa"
    ) == (0.03, 0.3, [])


def test_ingestion_point_shape_uses_stable_uuid_and_payload() -> None:
    product = {
        "id": "sku-wh",
        "name": "Máy nước nóng Test",
        "text": "Máy trực tiếp 4500 W.",
        "image_path": "/public/may_nuoc_nong.jpg",
        "metadata": {"category_scope": "water_heater"},
    }
    first = next(QDRANT_MODULE["point_generator"]([product]))
    second = next(QDRANT_MODULE["point_generator"]([product]))
    assert first.id == second.id
    assert isinstance(first.vector, models.Document)
    assert first.vector.model == "intfloat/multilingual-e5-small"
    assert first.vector.text.startswith("passage: ")
    assert set(first.payload) == {
        "product_id",
        "name",
        "text",
        "image_path",
        "metadata",
    }


def test_category_spec_and_config_match_qdrant_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "water_heater"
    assert spec.display_name == "Máy nước nóng"
    assert config["collection"] == "maynuocnong"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS


def test_profile_schema_and_missing_questions_are_water_heater_specific() -> None:
    profile = WaterHeaterNeedProfile(
        budget_max_vnd=5_000_000,
        water_supply="low_pressure",
        hard_constraints={"product_types": ["direct"]},
        usage_preferences=["low_pressure"],
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == ["heater_type", "water_supply", "budget"]
    with pytest.raises(ValidationError):
        WaterHeaterNeedProfile(water_supply="weekly_storage")
    with pytest.raises(ValidationError):
        WaterHeaterNeedProfile(hard_constraints={"product_types": ["heat_pump"]})


def test_filter_contains_only_indexed_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "budget_max_vnd": 5_000_000,
            "water_supply": "low_pressure",
            "soft_preferences": ["low_pressure_compatible"],
            "hard_constraints": {
                "brands": ["Stiebel Eltron"],
                "product_types": ["direct"],
                "max_power_w": 4500,
                "max_width_cm": 25,
                "required_features": ["booster_pump"],
                "required_safety_features": ["elcb"],
                "ip_ratings": ["IP25"],
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.category_scope",
        "metadata.brand",
        "metadata.product_type",
        "metadata.power_w",
        "metadata.has_booster_pump",
        "metadata.safety_tags",
        "metadata.ip_rating",
    ):
        assert path in text
    assert "low_pressure_compatible" not in text
    assert "water_supply" not in text


def test_filter_matches_payload_and_rejects_unknown_price() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="water-heater-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    eligible = {
        "category_scope": "water_heater",
        "brand": "Stiebel Eltron",
        "product_type": "direct",
        "power_w": 4500,
        "capacity_l": 0,
        "heating_time_max_minutes": 0.0,
        "has_booster_pump": True,
        "includes_shower": True,
        "width_cm": 21.0,
        "height_cm": 34.0,
        "depth_cm": 7.0,
        "ip_rating": "IP25",
        "safety_tags": ["elcb", "waterproof"],
    }
    client.upsert(
        collection_name="water-heater-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **eligible,
                        "original_price_vnd": 4_790_000,
                        "promotional_price_vnd": 3_990_000,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={"metadata": eligible},
            ),
            models.PointStruct(
                id=3,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **eligible,
                        "product_type": "indirect",
                        "original_price_vnd": 4_500_000,
                    }
                },
            ),
            models.PointStruct(
                id=4,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **eligible,
                        "original_price_vnd": 6_500_000,
                        "promotional_price_vnd": 6_000_000,
                    }
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="water-heater-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "budget_max_vnd": 5_000_000,
                "hard_constraints": {
                    "product_types": ["direct"],
                    "max_power_w": 4500,
                    "required_features": ["booster_pump"],
                    "required_safety_features": ["elcb"],
                },
            }
        ),
        limit=10,
    )
    assert [point.id for point in result.points] == [1]


def test_candidate_normalizer_whitelists_water_heater_fields() -> None:
    candidate = normalize_candidate(water_heater_point())
    assert candidate["product_id"] == "wh-1"
    assert candidate["effective_price_vnd"] == 3_990_000
    assert candidate["product_type"] == "direct"
    assert candidate["capacity_l"] == 0
    assert candidate["has_booster_pump"] is True
    assert candidate["water_pressure_mpa"] == {"min": 0.015, "max": 0.5}
    assert candidate["safety_tags"] == ["elcb", "flow_sensor", "waterproof"]
    assert candidate["dimensions_cm"] == {
        "width": 21.0,
        "height": 34.0,
        "depth": 7.0,
    }
    assert "metadata" not in candidate


def test_profile_patch_operations_use_water_heater_allowlist() -> None:
    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "budget_max_vnd": 5_000_000,
            "usage_preferences": ["instant_heating"],
            "hard_constraints": {"brands": ["Ariston"]},
        },
        ProfilePatch(
            replace={"hard_constraints.brands": ["Stiebel Eltron"]},
            remove={"usage_preferences": ["instant_heating"]},
            add={"usage_preferences": ["low_pressure"]},
            clear=["budget_max_vnd"],
            set={"capacity_segment": "large"},
        ),
        spec,
    )
    assert profile["hard_constraints"]["brands"] == ["Stiebel Eltron"]
    assert profile["usage_preferences"] == ["low_pressure"]
    assert "budget_max_vnd" not in profile
    assert "capacity_segment" not in profile
    assert set(changed) == {
        "hard_constraints.brands",
        "usage_preferences",
        "budget_max_vnd",
    }


def test_index_setup_is_idempotent() -> None:
    client = FakeQdrant()
    config = load_config()
    removed = "metadata.has_booster_pump"
    del client.payload_schemas["maynuocnong"][removed]
    assert ensure_payload_indexes(
        client, "maynuocnong", config["payload_indexes"], apply=False
    ) == {removed: "bool"}
    assert client.created == []
    assert ensure_payload_indexes(
        client, "maynuocnong", config["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "bool")]
    assert ensure_payload_indexes(
        client, "maynuocnong", config["payload_indexes"], apply=True
    ) == {}


def test_prompts_search_and_no_match_are_category_specific() -> None:
    spec = get_category_spec()
    extraction = spec.build_need_extraction_prompt("Cần máy nước nóng", {})
    ranking = spec.build_ranking_prompt(
        {"need_profile": {}, "hard_constraints": {}, "candidates": []}
    )
    response = spec.build_response_prompt(
        {"need_profile": {}, "selected_products": []}
    )
    search = spec.build_search_text(
        {
            "water_supply": "low_pressure",
            "hard_constraints": {"product_types": ["direct"]},
        },
        "Tìm máy phù hợp",
    )
    combined = " ".join((extraction, ranking, response, search, no_match_answer({})))
    assert "máy nước nóng" in combined.casefold()
    assert "nguồn nước yếu" in combined.casefold()
    assert "làm nóng trực tiếp" in combined.casefold()
    assert "tủ lạnh" not in combined.casefold()
    assert "bộ châu âu" not in combined.casefold()


def test_generic_query_interrupts_and_resumes_to_water_heater_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "water-heater-flow"}}

    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi máy nước nóng")]},
        config,
    )
    assert [
        item["question_id"]
        for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["heater_type", "water_supply", "budget"]

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "heater_type", "option_id": "direct"},
                    {"question_id": "water_supply", "option_id": "low_pressure"},
                    {"question_id": "budget", "option_id": "3m_5m"},
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["conversation"]["active_category"] == "water_heater"
    assert completed["need_profile"]["water_supply"] == "low_pressure"
    assert completed["need_profile"]["budget_max_vnd"] == 5_000_000
    assert completed["need_profile"]["hard_constraints"]["product_types"] == [
        "direct"
    ]
    assert completed["ranking"]["selected_products"][0]["product_id"] == "wh-1"
    assert completed["response"]["answer"].startswith("Mình đề xuất")
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "maynuocnong"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_water_supply_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=WaterHeaterCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Nước nhà tôi khá yếu",
            water_supply="low_pressure",
            usage_preferences=["low_pressure"],
            soft_preferences=["low_pressure_compatible"],
            confidence=1,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "water-heater-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua máy nước nóng")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "heater_type", "option_id": "direct"},
                    {
                        "question_id": "water_supply",
                        "option_id": "other",
                        "custom_answer": "Nước nhà tôi khá yếu",
                    },
                    {"question_id": "budget", "option_id": "5m_9m"},
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["water_supply"] == "low_pressure"
    assert completed["need_profile"]["custom_answers"]["water_supply"][
        "raw_answer"
    ] == "Nước nhà tôi khá yếu"


def test_switch_refrigerator_to_water_heater_and_back_restores_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.WATER_HEATER,
                category_transition="switch",
                switch_evidence="máy nước nóng",
                action="switch_category",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="switch",
                switch_evidence="tủ lạnh",
                action="switch_category",
            ),
        ],
        patches=[
            ProfilePatch(
                set={"household_size": 4, "budget_max_vnd": 20_000_000},
                replace={"usage_preferences": ["weekly_storage"]},
            ),
            ProfilePatch(
                set={"budget_max_vnd": 5_000_000, "water_supply": "stable"},
                replace={"hard_constraints.product_types": ["direct"]},
            ),
        ],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "refrigerator-water-heater"}}

    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn tủ lạnh")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua máy nước nóng")]}, config
    )
    assert switched["conversation"]["active_category"] == "water_heater"
    assert switched["ranking"]["selected_products"][0]["product_id"] == "wh-1"

    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại tủ lạnh")]}, config
    )
    assert restored["conversation"]["active_category"] == "refrigerator"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2
