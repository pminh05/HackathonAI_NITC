"""Behavior and data-contract tests for the dishwasher category."""

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

from advisor.categories.dishwasher import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.dishwasher.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.dishwasher.normalizer import normalize_candidate
from advisor.categories.dishwasher.schemas import (
    DishwasherCustomAnswer,
    DishwasherNeedProfile,
)
from advisor.categories.dishwasher.setup_indexes import ensure_payload_indexes
from advisor.categories.refrigerator import load_config as load_refrigerator_config
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


INGESTION_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_rua_chen/changeName.py"
)
normalize_metadata = INGESTION_MODULE["normalize_metadata"]
QDRANT_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_rua_chen/qdrant.py"
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
                category=IntentLabel.DISHWASHER,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            if self.owner.rankings:
                return self.owner.rankings.pop(0)
            product_id = "dw-1" if "dw-1" in prompt else "sku-rf"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Phù hợp nhu cầu đã xác nhận.",
                        "trade_off": "Một số tiện ích chưa có dữ liệu.",
                    }
                ]
            )
        if self.schema is DishwasherCustomAnswer:
            return self.owner.custom or DishwasherCustomAnswer(
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
        rankings: list[RankingResult] | None = None,
        custom: DishwasherCustomAnswer | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.rankings = list(rankings or [])
        self.custom = custom
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        return AIMessage(
            content="Mình đề xuất mẫu máy rửa chén thử nghiệm theo nhu cầu của bạn."
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
        return SimpleNamespace(points=[dishwasher_point()])


def dishwasher_point() -> Any:
    return SimpleNamespace(
        id="point-dw-1",
        score=0.93,
        payload={
            "product_id": "dw-1",
            "name": "Máy rửa chén Bosch SMS Test",
            "text": "Máy độc lập 14 bộ, độ ồn 44 dB.",
            "image_path": "/public/may_rua_chen.jpg",
            "metadata": {
                "category_scope": "dishwasher",
                "brand": "Bosch",
                "model_code": "SMS Test",
                "product_type": "freestanding",
                "Loại sản phẩm": "Máy rửa chén độc lập",
                "original_price_vnd": 19_000_000,
                "promotional_price_vnd": 16_500_000,
                "vietnamese_meals_min": 3,
                "vietnamese_meals_max": 4,
                "place_settings_min": 14,
                "place_settings_max": 14,
                "water_min_l_per_cycle": 9.5,
                "water_max_l_per_cycle": 9.5,
                "noise_db": 44.0,
                "width_cm": 60.0,
                "height_cm": 84.0,
                "depth_cm": 60.0,
                "power_min_w": 2000,
                "power_max_w": 2000,
                "program_count": 8,
                "Chương trình": "Rửa nhanh | Rửa tiết kiệm ECO",
                "Công nghệ": "Rửa bằng nước nóng",
                "Công nghệ sấy": "Sấy ngưng tụ",
                "Tiện ích": "Aquastop - chống tràn nước",
            },
        },
    )


def test_ingestion_normalizes_faulty_prices_types_and_units() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Bosch",
            "Loại sản phẩm": "Máy rửa chén độc lập",
            "giá gốc": "19990000.0",
            "giá khuyến mãi": "14990000.0",
            "gia_goc_vnd": 199_900_000,
            "gia_khuyen_mai_vnd": 149_900_000,
            "bua_an_viet_min": 3,
            "bua_an_viet_max": 4,
            "bo_chau_au_min": 13,
            "bo_chau_au_max": 13,
            "tieu_thu_nuoc_min_lit_lan": 4,
            "tieu_thu_nuoc_max_lit_lan": 14,
            "do_on_db": 46,
            "ngang_cm": 60,
            "cao_cm": 84,
            "sau_cm": 60,
        }
    )
    assert metadata["original_price_vnd"] == 19_990_000
    assert metadata["promotional_price_vnd"] == 14_990_000
    assert metadata["product_type"] == "freestanding"
    assert metadata["category_scope"] == "dishwasher"
    assert metadata["place_settings_max"] == 13
    assert metadata["water_max_l_per_cycle"] == 14.0
    assert "gia_goc_vnd" not in metadata
    assert "gia_khuyen_mai_vnd" not in metadata

    dryer = normalize_metadata({"Loại sản phẩm": "Máy sấy chén"})
    assert dryer["product_type"] == "dish_dryer"
    assert dryer["category_scope"] == "dish_dryer"


def test_ingestion_point_shape_uses_cloud_inference_and_stable_payload() -> None:
    product = {
        "id": "sku-dw",
        "name": "Máy rửa chén Test",
        "text": "Máy rửa chén độc lập 14 bộ.",
        "image_path": "/public/may_rua_chen.jpg",
        "metadata": {"category_scope": "dishwasher"},
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
    assert spec.name == config["category"] == "dishwasher"
    assert spec.display_name == "Máy rửa chén"
    assert config["collection"] == "mayruachen"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS


def test_profile_schema_and_missing_questions_are_dishwasher_specific() -> None:
    profile = DishwasherNeedProfile(
        budget_max_vnd=18_000_000,
        capacity_segment="standard",
        hard_constraints={"product_types": ["freestanding"]},
        usage_preferences=["quiet_night"],
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == ["installation", "capacity", "budget"]
    with pytest.raises(ValidationError):
        DishwasherNeedProfile(usage_preferences=["frozen_storage"])
    with pytest.raises(ValidationError):
        DishwasherNeedProfile(hard_constraints={"product_types": ["dish_dryer"]})


def test_filter_contains_only_indexed_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "budget_max_vnd": 18_000_000,
            "capacity_segment": "large",
            "soft_preferences": ["auto_door", "wifi"],
            "hard_constraints": {
                "brands": ["Bosch"],
                "product_types": ["freestanding"],
                "min_place_settings": 14,
                "max_width_cm": 60,
                "max_noise_db": 45,
                "max_water_l_per_cycle": 10,
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.category_scope",
        "metadata.brand",
        "metadata.product_type",
        "metadata.place_settings_max",
        "metadata.noise_db",
        "metadata.water_max_l_per_cycle",
    ):
        assert path in text
    assert "auto_door" not in text
    assert "wifi" not in text
    assert "large" not in text


def test_filter_matches_payload_and_handles_missing_price_in_local_qdrant() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="dishwasher-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    base = {
        "category_scope": "dishwasher",
        "brand": "Bosch",
        "product_type": "freestanding",
        "place_settings_max": 14,
        "vietnamese_meals_max": 4,
        "water_max_l_per_cycle": 9.5,
        "noise_db": 44.0,
        "width_cm": 60.0,
        "height_cm": 84.0,
        "depth_cm": 60.0,
    }
    client.upsert(
        collection_name="dishwasher-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "original_price_vnd": 19_000_000,
                        "promotional_price_vnd": 16_500_000,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={"metadata": {**base, "category_scope": "dish_dryer"}},
            ),
            models.PointStruct(
                id=3,
                vector=[1.0, 0.0],
                payload={"metadata": base},
            ),
            models.PointStruct(
                id=4,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "original_price_vnd": 22_000_000,
                        "promotional_price_vnd": 20_000_000,
                    }
                },
            ),
        ],
        wait=True,
    )
    strict = client.query_points(
        collection_name="dishwasher-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "budget_max_vnd": 18_000_000,
                "hard_constraints": {
                    "brands": ["Bosch"],
                    "product_types": ["freestanding"],
                    "min_place_settings": 14,
                    "max_noise_db": 45,
                    "max_water_l_per_cycle": 10,
                },
            }
        ),
        limit=10,
    )
    assert [point.id for point in strict.points] == [1]

    without_budget = client.query_points(
        collection_name="dishwasher-test",
        query=[1.0, 0.0],
        query_filter=build_filter({}),
        limit=10,
    )
    assert {point.id for point in without_budget.points} == {1, 3, 4}


def test_candidate_normalizer_whitelists_dishwasher_fields() -> None:
    candidate = normalize_candidate(dishwasher_point())
    assert candidate["product_id"] == "dw-1"
    assert candidate["effective_price_vnd"] == 16_500_000
    assert candidate["product_type"] == "freestanding"
    assert candidate["capacity"]["place_settings_max"] == 14
    assert candidate["water_l_per_cycle"]["max"] == 9.5
    assert candidate["noise_db"] == 44.0
    assert candidate["dimensions_cm"] == {
        "width": 60.0,
        "height": 84.0,
        "depth": 60.0,
    }
    assert "metadata" not in candidate


def test_profile_patch_operations_use_dishwasher_allowlist() -> None:
    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "budget_max_vnd": 25_000_000,
            "usage_preferences": ["quick_cycle"],
            "hard_constraints": {"brands": ["Bosch"]},
        },
        ProfilePatch(
            replace={"hard_constraints.brands": ["Electrolux"]},
            remove={"usage_preferences": ["quick_cycle"]},
            add={"usage_preferences": ["quiet_night"]},
            clear=["budget_max_vnd"],
            set={"household_size": 4},
        ),
        spec,
    )
    assert profile["hard_constraints"]["brands"] == ["Electrolux"]
    assert profile["usage_preferences"] == ["quiet_night"]
    assert "budget_max_vnd" not in profile
    assert "household_size" not in profile
    assert set(changed) == {
        "hard_constraints.brands",
        "usage_preferences",
        "budget_max_vnd",
    }


def test_index_setup_is_idempotent() -> None:
    client = FakeQdrant()
    config = load_config()
    removed = "metadata.place_settings_max"
    del client.payload_schemas["mayruachen"][removed]
    assert ensure_payload_indexes(
        client, "mayruachen", config["payload_indexes"], apply=False
    ) == {removed: "integer"}
    assert client.created == []
    assert ensure_payload_indexes(
        client, "mayruachen", config["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "integer")]
    assert ensure_payload_indexes(
        client, "mayruachen", config["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "integer")]


def test_prompts_search_and_no_match_are_category_specific() -> None:
    spec = get_category_spec()
    extraction = spec.build_need_extraction_prompt("Cần máy rửa chén", {})
    ranking = spec.build_ranking_prompt(
        {"need_profile": {}, "hard_constraints": {}, "candidates": []}
    )
    response = spec.build_response_prompt(
        {"need_profile": {}, "selected_products": []}
    )
    search = spec.build_search_text(
        {
            "capacity_segment": "large",
            "hard_constraints": {"product_types": ["built_in"]},
        },
        "Tìm máy phù hợp",
    )
    combined = " ".join((extraction, ranking, response, search, no_match_answer({})))
    assert "máy rửa chén" in combined.casefold()
    assert "14 bộ" in combined.casefold()
    assert "âm tủ" in combined.casefold()
    assert "tủ lạnh" not in combined.casefold()
    assert "khối lượng giặt" not in combined.casefold()


def test_generic_query_interrupts_and_resumes_to_dishwasher_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "dishwasher-flow"}}

    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi máy rửa chén")]},
        config,
    )
    assert [
        item["question_id"]
        for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["installation", "capacity", "budget"]

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "installation", "option_id": "freestanding"},
                    {"question_id": "capacity", "option_id": "standard"},
                    {"question_id": "budget", "option_id": "12m_18m"},
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["conversation"]["active_category"] == "dishwasher"
    assert completed["need_profile"]["capacity_segment"] == "standard"
    assert completed["need_profile"]["budget_max_vnd"] == 18_000_000
    assert completed["need_profile"]["hard_constraints"]["product_types"] == [
        "freestanding"
    ]
    assert completed["ranking"]["selected_products"][0]["product_id"] == "dw-1"
    assert completed["response"]["answer"].startswith("Mình đề xuất")
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "mayruachen"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_capacity_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=DishwasherCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Khoảng 14 bộ châu Âu",
            capacity_segment="large",
            hard_constraints={"min_place_settings": 14},
            confidence=1,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "dishwasher-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua máy rửa chén")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "installation", "option_id": "freestanding"},
                    {
                        "question_id": "capacity",
                        "option_id": "other",
                        "custom_answer": "Khoảng 14 bộ châu Âu",
                    },
                    {"question_id": "budget", "option_id": "18m_25m"},
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["capacity_segment"] == "large"
    assert completed["need_profile"]["hard_constraints"][
        "min_place_settings"
    ] == 14
    assert completed["need_profile"]["custom_answers"]["capacity"][
        "raw_answer"
    ] == "Khoảng 14 bộ châu Âu"


def test_switch_refrigerator_to_dishwasher_and_back_restores_context() -> None:
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
                set={"budget_max_vnd": 18_000_000, "capacity_segment": "standard"},
                replace={"hard_constraints.product_types": ["freestanding"]},
            ),
        ],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "refrigerator-dishwasher"}}

    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn tủ lạnh")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua máy rửa chén")]}, config
    )
    assert switched["conversation"]["active_category"] == "dishwasher"
    assert switched["ranking"]["selected_products"][0]["product_id"] == "dw-1"

    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại tủ lạnh")]}, config
    )
    assert restored["conversation"]["active_category"] == "refrigerator"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2
