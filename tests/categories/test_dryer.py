"""Behavior and data-contract tests for the clothes-dryer category."""

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

from advisor.categories.dryer import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.dryer.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.dryer.normalizer import normalize_candidate
from advisor.categories.dryer.schemas import DryerCustomAnswer, DryerNeedProfile
from advisor.categories.dryer.setup_indexes import ensure_payload_indexes
from advisor.categories.refrigerator import load_config as load_refrigerator_config
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


CHANGE_NAME_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_say_quan_ao/changeName.py"
)
normalize_metadata = CHANGE_NAME_MODULE["normalize_metadata"]
PROCESSING_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_say_quan_ao/processing.py"
)
QDRANT_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_say_quan_ao/qdrant.py"
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
                category=IntentLabel.DRYER,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            product_id = "fridge-1" if "fridge-1" in prompt else "dryer-1"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Phù hợp quy mô gia đình và nhu cầu sử dụng.",
                        "trade_off": "Một số thông tin tiện ích chưa đầy đủ.",
                    }
                ]
            )
        if self.schema is DryerCustomAnswer:
            return self.owner.custom or DryerCustomAnswer(
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
        custom: DryerCustomAnswer | None = None,
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
            content="Mình đề xuất mẫu máy sấy thử nghiệm theo nhu cầu của bạn."
        )


class FakeQdrant:
    def __init__(self) -> None:
        self.schemas = {
            "maysayquanao": {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in load_config()["payload_indexes"].items()
            },
            "tulanh": {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in load_refrigerator_config()[
                    "payload_indexes"
                ].items()
            },
        }
        self.created: list[tuple[str, str]] = []
        self.query_kwargs: dict[str, Any] | None = None
        self.query_count = 0

    def get_collection(self, collection: str) -> Any:
        return SimpleNamespace(payload_schema=self.schemas[collection])

    def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema: Any, **_: Any
    ) -> None:
        value = getattr(field_schema, "value", str(field_schema))
        self.schemas[collection_name][field_name] = SimpleNamespace(
            data_type=SimpleNamespace(value=value)
        )
        self.created.append((field_name, str(value)))

    def query_points(self, **kwargs: Any) -> Any:
        self.query_kwargs = kwargs
        self.query_count += 1
        if kwargs["collection_name"] == "tulanh":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="point-fridge-1",
                        score=0.9,
                        payload={
                            "product_id": "fridge-1",
                            "name": "Tủ lạnh Test",
                            "text": "Tủ lạnh cho gia đình.",
                            "image_path": "/public/tu_lanh.jpg",
                            "metadata": {
                                "brand": "Test",
                                "Giá gốc vnd": 15_000_000,
                                "Dung tích sử dụng lít": 300,
                            },
                        },
                    )
                ]
            )
        return SimpleNamespace(points=[dryer_point()])


def dryer_point() -> Any:
    return SimpleNamespace(
        id="point-dryer-1",
        score=0.93,
        payload={
            "product_id": "dryer-1",
            "name": "Máy sấy Test 9 kg",
            "text": "Máy sấy bơm nhiệt 9 kg có inverter và cảm biến.",
            "image_path": "/public/may_say_quan_ao.jpg",
            "metadata": {
                "brand": "Electrolux",
                "dryer_type": "heat_pump",
                "Loại sản phẩm": "Sấy bơm nhiệt",
                "original_price_vnd": 17_000_000,
                "promotional_price_vnd": 15_000_000,
                "dry_capacity_kg": 9.0,
                "people_min": 3,
                "people_max": 5,
                "Số người sử dụng": "Từ 3 - 5 người (8 - 9 kg)",
                "has_inverter": True,
                "has_sensor": True,
                "power_min_w": 700,
                "power_max_w": 700,
                "temperature_min_c": 55.0,
                "temperature_max_c": 60.0,
                "width_cm": 59.0,
                "height_cm": 85.0,
                "depth_cm": 66.0,
                "Công nghệ": "Cảm biến độ ẩm và sấy đảo chiều",
                "Công nghệ tiết kiệm điện": "Inverter",
                "Tiện ích": "Chống nhăn | Khóa trẻ em",
                "Sản xuất tại": "Thái Lan",
            },
        },
    )


def complete_dryer_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"household_size": 5, "budget_max_vnd": 20_000_000},
        replace={"usage_preferences": ["energy_saving"]},
    )


def complete_refrigerator_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"household_size": 4, "budget_max_vnd": 20_000_000},
        replace={"usage_preferences": ["weekly_storage"]},
    )


def test_ingestion_normalizes_price_type_dimensions_and_capabilities() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Electrolux",
            "Loại sản phẩm": "Sấy bơm nhiệt",
            "Số người sử dụng": "Trên 7 người (Trên 10 kg)",
            "khoi_luong_say_kg": 10.5,
            "ngang_cm": 59,
            "cao_cm": 85,
            "sau_cm": 63.5,
            "cong_suat_min_w": 700,
            "cong_suat_max_w": 800,
            "Công nghệ tiết kiệm điện": "Dual Inverter",
            "co_cam_bien": True,
            "giá gốc": "18640000.0",
            "giá khuyến mãi": "13190000.0",
            "gia_goc_vnd": 186_400_000,
            "gia_khuyen_mai_vnd": 131_900_000,
        }
    )
    assert metadata["original_price_vnd"] == 18_640_000
    assert metadata["promotional_price_vnd"] == 13_190_000
    assert "gia_goc_vnd" not in metadata
    assert metadata["dryer_type"] == "heat_pump"
    assert metadata["dry_capacity_kg"] == 10.5
    assert metadata["people_min"] == 8
    assert "people_max" not in metadata
    assert metadata["has_inverter"] is True
    assert metadata["has_sensor"] is True
    assert metadata["width_cm"] == 59.0
    assert metadata["height_cm"] == 85.0
    assert metadata["depth_cm"] == 63.5
    assert metadata["power_max_w"] == 800


def test_semantic_text_uses_real_dryer_technology_field() -> None:
    text = PROCESSING_MODULE["create_semantic_text"](
        {
            "brand": "LG",
            "model_code": "DVHP09",
            "Loại sản phẩm": "Sấy bơm nhiệt",
            "khoi_luong_say_kg": 9,
            "Công nghệ": "Sấy đảo chiều và cảm biến độ ẩm",
        },
        "sku-dryer",
    )
    assert "Sấy đảo chiều và cảm biến độ ẩm" in text
    assert "khối lượng sấy (kg): 9" in text


def test_ingestion_point_shape_uses_canonical_cloud_inference_payload() -> None:
    product = {
        "id": "sku-dryer",
        "name": "Máy sấy Test sku-dryer",
        "text": "Máy sấy bơm nhiệt 9 kg.",
        "image_path": "/public/may_say_quan_ao.jpg",
        "metadata": {
            "category_scope": "dryer",
            "dryer_type": "heat_pump",
            "dry_capacity_kg": 9.0,
        },
    }
    first = next(QDRANT_MODULE["point_generator"]([product]))
    second = next(QDRANT_MODULE["point_generator"]([product]))
    assert first.id == second.id
    assert isinstance(first.vector, models.Document)
    assert first.vector.model == "intfloat/multilingual-e5-small"
    assert first.vector.text == "passage: Máy sấy bơm nhiệt 9 kg."
    assert set(first.payload) == {
        "product_id",
        "name",
        "text",
        "image_path",
        "metadata",
    }

    with pytest.raises(ValueError, match="metadata canonical"):
        next(
            QDRANT_MODULE["point_generator"](
                [
                    {
                        **product,
                        "metadata": {"category": "máy sấy quần áo"},
                    }
                ]
            )
        )


def test_category_spec_and_config_match_qdrant_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "dryer"
    assert spec.display_name == "Máy sấy quần áo"
    assert config["collection"] == "maysayquanao"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS


def test_profile_schema_and_missing_questions_are_dryer_specific() -> None:
    profile = DryerNeedProfile(
        household_size=5,
        budget_max_vnd=20_000_000,
        usage_preferences=["energy_saving"],
        hard_constraints={
            "dryer_types": ["heat_pump"],
            "min_dry_capacity_kg": 9,
            "inverter": True,
        },
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == [
        "household_size",
        "budget",
        "usage_preferences",
    ]
    with pytest.raises(ValidationError):
        DryerNeedProfile(usage_preferences=["daily_laundry"])
    with pytest.raises(ValidationError):
        DryerNeedProfile(
            hard_constraints={
                "min_dry_capacity_kg": 10,
                "max_dry_capacity_kg": 8,
            }
        )


def test_filter_contains_only_indexed_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "household_size": 5,
            "budget_max_vnd": 16_000_000,
            "soft_preferences": ["anti_wrinkle", "wifi"],
            "hard_constraints": {
                "brands": ["Electrolux"],
                "dryer_types": ["heat_pump"],
                "min_dry_capacity_kg": 9,
                "max_width_cm": 60,
                "max_power_w": 1000,
                "inverter": True,
                "sensor": True,
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.brand",
        "metadata.dryer_type",
        "metadata.dry_capacity_kg",
        "metadata.power_max_w",
        "metadata.has_inverter",
        "metadata.has_sensor",
    ):
        assert path in text
    assert "anti_wrinkle" not in text
    assert "wifi" not in text


def test_filter_matches_normalized_payload_and_price_fallback() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="dryer-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )

    def point(point_id: int, metadata: dict[str, Any]) -> models.PointStruct:
        return models.PointStruct(
            id=point_id,
            vector=[1.0, 0.0],
            payload={"metadata": metadata},
        )

    base = {
        "brand": "Electrolux",
        "dryer_type": "heat_pump",
        "dry_capacity_kg": 9.0,
        "people_min": 3,
        "people_max": 5,
        "has_inverter": True,
        "has_sensor": True,
        "power_max_w": 800,
        "width_cm": 59.0,
    }
    client.upsert(
        collection_name="dryer-test",
        points=[
            point(
                1,
                {
                    **base,
                    "original_price_vnd": 17_000_000,
                    "promotional_price_vnd": 15_000_000,
                },
            ),
            point(
                2,
                {
                    **base,
                    "original_price_vnd": 14_000_000,
                    "promotional_price_vnd": 20_000_000,
                },
            ),
            point(3, base),
            point(
                4,
                {
                    **base,
                    "dryer_type": "vented",
                    "original_price_vnd": 8_000_000,
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="dryer-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "household_size": 5,
                "budget_max_vnd": 16_000_000,
                "hard_constraints": {
                    "brands": ["Electrolux"],
                    "dryer_types": ["heat_pump"],
                    "min_dry_capacity_kg": 9,
                    "max_width_cm": 60,
                    "max_power_w": 1000,
                    "inverter": True,
                    "sensor": True,
                },
            }
        ),
        limit=5,
    )
    assert [point.id for point in result.points] == [1]


def test_candidate_normalizer_whitelists_dryer_fields() -> None:
    candidate = normalize_candidate(dryer_point())
    assert candidate["product_id"] == "dryer-1"
    assert candidate["effective_price_vnd"] == 15_000_000
    assert candidate["dryer_type"] == "heat_pump"
    assert candidate["dry_capacity_kg"] == 9.0
    assert candidate["has_inverter"] is True
    assert candidate["has_sensor"] is True
    assert candidate["dimensions_cm"] == {
        "width": 59.0,
        "height": 85.0,
        "depth": 66.0,
    }
    assert "metadata" not in candidate


def test_profile_patch_operations_use_dryer_allowlist() -> None:
    profile, changed = apply_profile_patch(
        {
            "budget_max_vnd": 20_000_000,
            "usage_preferences": ["rainy_season"],
            "hard_constraints": {"brands": ["Panasonic"]},
        },
        ProfilePatch(
            replace={"hard_constraints.brands": ["Electrolux"]},
            remove={"usage_preferences": ["rainy_season"]},
            add={"usage_preferences": ["energy_saving"]},
            clear=["budget_max_vnd"],
            set={"room_area_m2": 20},
        ),
        get_category_spec(),
    )
    assert profile["hard_constraints"]["brands"] == ["Electrolux"]
    assert profile["usage_preferences"] == ["energy_saving"]
    assert "budget_max_vnd" not in profile
    assert "room_area_m2" not in profile
    assert set(changed) == {
        "hard_constraints.brands",
        "usage_preferences",
        "budget_max_vnd",
    }


def test_index_setup_is_idempotent() -> None:
    client = FakeQdrant()
    config = load_config()
    removed = "metadata.dry_capacity_kg"
    del client.schemas["maysayquanao"][removed]
    assert ensure_payload_indexes(
        client, "maysayquanao", config["payload_indexes"], apply=False
    ) == {removed: "float"}
    assert client.created == []
    assert ensure_payload_indexes(
        client, "maysayquanao", config["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "float")]
    assert ensure_payload_indexes(
        client, "maysayquanao", config["payload_indexes"], apply=True
    ) == {}


def test_prompts_and_no_match_text_are_dryer_specific() -> None:
    spec = get_category_spec()
    extraction = spec.build_need_extraction_prompt("Cần máy sấy 9 kg", {})
    ranking = spec.build_ranking_prompt(
        {"need_profile": {}, "hard_constraints": {}, "candidates": []}
    )
    response = spec.build_response_prompt(
        {"need_profile": {}, "selected_products": []}
    )
    combined = " ".join((extraction, ranking, response, no_match_answer({})))
    assert "máy sấy" in combined.casefold()
    assert "tải sấy" in combined.casefold()
    assert "tủ lạnh" not in combined.casefold()
    assert "btu" not in combined.casefold()


def test_generic_dryer_query_interrupts_and_resumes_to_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "dryer-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi máy sấy quần áo")]},
        config,
    )
    payload = interrupted["__interrupt__"][0].value
    assert payload["category"] == "dryer"
    assert [question["question_id"] for question in payload["questions"]] == [
        "household_size",
        "budget",
        "usage_preferences",
    ]

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "household_size", "option_id": "three_five"},
                    {"question_id": "budget", "option_id": "15m_20m"},
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
    assert completed["conversation"]["active_category"] == "dryer"
    assert completed["need_profile"]["household_size"] == 5
    assert completed["need_profile"]["budget_max_vnd"] == 20_000_000
    assert completed["ranking"]["selected_products"][0]["product_id"] == "dryer-1"
    assert completed["response"]["answer"].startswith("Mình đề xuất")
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "maysayquanao"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_household_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=DryerCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Nhà có 9 người",
            household_size=9,
            confidence=1,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "dryer-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua máy sấy quần áo")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "household_size",
                        "option_id": "other",
                        "custom_answer": "Nhà có 9 người",
                    },
                    {"question_id": "budget", "option_id": "15m_20m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "rainy_season",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["household_size"] == 9
    assert completed["need_profile"]["custom_answers"]["household_size"][
        "raw_answer"
    ] == "Nhà có 9 người"


def test_switch_dryer_to_refrigerator_and_back_restores_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.DRYER,
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
                category=IntentLabel.DRYER,
                category_transition="switch",
                switch_evidence="máy sấy",
                action="switch_category",
            ),
        ],
        patches=[complete_dryer_patch(), complete_refrigerator_patch()],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "dryer-refrigerator-dryer"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy sấy quần áo")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua tủ lạnh")]}, config
    )
    assert switched["conversation"]["active_category"] == "refrigerator"
    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại máy sấy")]}, config
    )
    assert restored["conversation"]["active_category"] == "dryer"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2
