"""Behavior and data-contract tests for the cooler/freezer category."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from advisor.categories.cooler_freezer import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.cooler_freezer.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.cooler_freezer.normalizer import normalize_candidate
from advisor.categories.cooler_freezer.schemas import (
    CoolerFreezerCustomAnswer,
    CoolerFreezerNeedProfile,
)
from advisor.categories.cooler_freezer.setup_indexes import ensure_payload_indexes
from advisor.categories.refrigerator import load_config as load_refrigerator_config
from advisor.categories.registry import build_default_registry
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


ROOT = Path(__file__).parents[2]
CHANGE_NAME_MODULE = runpy.run_path(
    ROOT / "ingestion/tu_mat_tu_dong/changeName.py"
)
normalize_metadata = CHANGE_NAME_MODULE["normalize_metadata"]
normalize_temperature_range = CHANGE_NAME_MODULE["normalize_temperature_range"]
PROCESSING_MODULE = runpy.run_path(
    ROOT / "ingestion/tu_mat_tu_dong/processing.py"
)
QDRANT_MODULE = runpy.run_path(ROOT / "ingestion/tu_mat_tu_dong/qdrant.py")


class FakeStructuredModel:
    def __init__(self, owner: "FakeLLM", schema: type[Any]) -> None:
        self.owner = owner
        self.schema = schema

    def invoke(self, prompt: str) -> Any:
        self.owner.calls.append((self.schema.__name__, prompt))
        if self.schema is TurnAnalysisResult:
            if self.owner.analyses:
                return self.owner.analyses.pop(0)
            return TurnAnalysisResult(
                category=IntentLabel.COOLER_FREEZER,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            product_id = "fridge-1" if "fridge-1" in prompt else "freezer-1"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Đúng loại tủ, dung tích và ngân sách yêu cầu.",
                        "trade_off": "Một số thông tin điện năng chưa đầy đủ.",
                    }
                ]
            )
        if self.schema is CoolerFreezerCustomAnswer:
            return self.owner.custom or CoolerFreezerCustomAnswer(
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
        custom: CoolerFreezerCustomAnswer | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.custom = custom
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        return AIMessage(content="Mình đề xuất mẫu tủ đông thử nghiệm phù hợp.")


class FakeQdrant:
    def __init__(self) -> None:
        self.schemas = {
            "tumattudong": {
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
        return SimpleNamespace(points=[freezer_point()])


def freezer_point() -> Any:
    return SimpleNamespace(
        id="point-freezer-1",
        score=0.94,
        payload={
            "product_id": "freezer-1",
            "name": "Tủ đông Test 200 lít",
            "text": "Tủ đông mini 200 lít đạt -25°C, có Inverter.",
            "image_path": "/public/tu_mat_tu_dong.jpg",
            "metadata": {
                "category_scope": "cooler_freezer",
                "brand": "Sanaky",
                "model_code": "CF200",
                "product_family": "freezer",
                "product_type": "freezer_mini",
                "is_mini": True,
                "original_price_vnd": 8_000_000,
                "promotional_price_vnd": 7_500_000,
                "effective_price_vnd": 7_500_000,
                "total_capacity_lit": 200,
                "compartment_count": 1,
                "freezer_compartment_count": 1,
                "cooler_compartment_count": 0,
                "temperature_min_c": -25.0,
                "temperature_max_c": -18.0,
                "has_inverter": True,
                "energy_kwh_day": 0.8,
                "noise_max_db": 42.0,
                "door_count": 1,
                "width_cm": 55.0,
                "height_cm": 85.0,
                "depth_cm": 60.0,
                "weight_kg": 35.0,
                "gas_type": "R600a",
                "feature_tags": ["fast_freeze", "lock", "wheels", "drain"],
                "Công nghệ": "Làm đông nhanh",
                "Tiện ích": "Khóa cửa | Bánh xe | Lỗ thoát nước",
                "Sản xuất tại": "Việt Nam",
                "nam_ra_mat": 2025,
            },
        },
    )


def complete_cooler_freezer_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"product_family": "freezer", "budget_max_vnd": 15_000_000},
        replace={"usage_preferences": ["bulk_frozen_storage"]},
    )


def complete_refrigerator_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"household_size": 4, "budget_max_vnd": 20_000_000},
        replace={"usage_preferences": ["weekly_storage"]},
    )


def test_ingestion_normalizes_typed_metadata_and_features() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Sanaky",
            "model_code": "CF200",
            "Loại sản phẩm": "Tủ đông mini",
            "dung_tich_tong_lit": 200,
            "tong_so_ngan": 1,
            "so_ngan_dong": 1,
            "so_ngan_mat": 0,
            "Nhiệt độ ngăn đông (độ C)": "10 - -25°C",
            "Công nghệ tiết kiệm điện": "Inverter",
            "Tiện ích": "Khóa cửa tủ | Bánh xe | Lỗ thoát nước | Làm đông nhanh",
            "Chất liệu mặt": "Kính cường lực",
            "ngang_cm": 55,
            "cao_cm": 85,
            "sau_cm": 60.5,
            "loai_gas_chuan": "R600a",
            "giá gốc": "8000000",
            "giá khuyến mãi": "7500000",
        }
    )
    assert metadata["category_scope"] == "cooler_freezer"
    assert metadata["product_family"] == "freezer"
    assert metadata["product_type"] == "freezer_mini"
    assert metadata["is_mini"] is True
    assert metadata["original_price_vnd"] == 8_000_000
    assert metadata["promotional_price_vnd"] == 7_500_000
    assert metadata["effective_price_vnd"] == 7_500_000
    assert metadata["temperature_min_c"] == -25.0
    assert metadata["temperature_max_c"] == 10.0
    assert metadata["has_inverter"] is True
    assert metadata["width_cm"] == 55.0
    assert metadata["depth_cm"] == 60.5
    assert metadata["feature_tags"] == [
        "drain",
        "fast_freeze",
        "glass_door",
        "lock",
        "wheels",
    ]


def test_temperature_parser_distinguishes_range_separator_and_unary_minus() -> None:
    assert normalize_temperature_range(
        {"Nhiệt độ ngăn đông (độ C)": "0 - 10°C"}, "cooler"
    ) == (0.0, 10.0)
    assert normalize_temperature_range(
        {"Nhiệt độ ngăn đông (độ C)": "≤ - 18℃"}, "freezer"
    ) == (-18.0, -18.0)
    assert normalize_temperature_range(
        {"Nhiệt độ ngăn đông (độ C)": "Dưới 18℃"}, "freezer"
    ) == (None, None)


def test_semantic_text_is_specific_to_each_family() -> None:
    cooler = PROCESSING_MODULE["create_semantic_text"](
        {
            "brand": "Hoà Phát",
            "model_code": "HPC300",
            "Loại sản phẩm": "Tủ mát",
            "dung_tich_tong_lit": 300,
            "Nhiệt độ ngăn đông (độ C)": "0 - 10°C",
            "Công nghệ": "Quạt đối lưu | Làm lạnh trực tiếp",
        },
        "sku-cooler",
    )
    freezer = PROCESSING_MODULE["create_semantic_text"](
        {
            "brand": "Sanaky",
            "model_code": "CF200",
            "Loại sản phẩm": "Tủ đông mini",
            "dung_tich_tong_lit": 200,
            "Nhiệt độ ngăn đông (độ C)": "-18°C",
        },
        "sku-freezer",
    )
    assert "Tủ mát Hoà Phát" in cooler
    assert "Khoảng nhiệt độ làm mát" in cooler
    assert "Tủ đông mini Sanaky" in freezer
    assert "Nhiệt độ vận hành" in freezer


def test_source_dataset_has_valid_unique_skus_and_product_families() -> None:
    payload = json.loads(
        (ROOT / "ingestion/tu_mat_tu_dong/data/tu_mat_tu_dong.json").read_text(
            encoding="utf-8-sig"
        )
    )
    products = payload["data_clean"]
    skus = [str(item.get("sku") or "").strip() for item in products]
    assert len(products) == 222
    assert len(set(skus)) == len(skus)
    assert all(skus)
    assert all(item.get("loai_san_pham_chuan") for item in products)


def test_ingestion_point_shape_is_stable_and_uses_384_dimensions() -> None:
    product = {
        "id": "sku-freezer",
        "name": "Tủ đông Test sku-freezer",
        "text": "Tủ đông 200 lít đạt -25°C.",
        "image_path": "/public/tu_mat_tu_dong.jpg",
        "metadata": {
            "category_scope": "cooler_freezer",
            "product_family": "freezer",
        },
        "vector": [0.0] * 384,
    }
    first = next(QDRANT_MODULE["point_generator"]([product]))
    second = next(QDRANT_MODULE["point_generator"]([product]))
    assert first.id == second.id
    assert len(first.vector) == 384
    assert set(first.payload) == {
        "product_id",
        "name",
        "text",
        "image_path",
        "metadata",
    }


def test_payload_refresh_reuses_existing_vector_and_point_id() -> None:
    product = {
        "id": "sku-freezer",
        "name": "Tủ đông Test sku-freezer",
        "text": "Tủ đông 200 lít đạt -25°C.",
        "image_path": "/public/tu_mat_tu_dong.jpg",
        "metadata": {
            "category_scope": "cooler_freezer",
            "product_family": "freezer",
            "effective_price_vnd": 7_500_000,
        },
    }
    legacy = SimpleNamespace(
        id="legacy-point-id",
        vector=[0.25] * 384,
        payload={"product_id": "sku-freezer", "metadata": {"old": True}},
    )
    refreshed = next(
        QDRANT_MODULE["payload_refresh_generator"]([product], [legacy])
    )
    assert refreshed.id == "legacy-point-id"
    assert refreshed.vector == [0.25] * 384
    assert refreshed.payload["metadata"]["product_family"] == "freezer"
    assert refreshed.payload["metadata"]["effective_price_vnd"] == 7_500_000


def test_category_spec_config_and_registry_match_qdrant_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    registry = build_default_registry()
    registry.validate_all()
    assert spec.name == config["category"] == "cooler_freezer"
    assert spec.display_name == "Tủ mát, tủ đông"
    assert config["collection"] == "tumattudong"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert registry.get("cooler_freezer").implemented is True


def test_profile_schema_and_missing_questions_are_category_specific() -> None:
    profile = CoolerFreezerNeedProfile(
        product_family="freezer",
        budget_max_vnd=15_000_000,
        usage_preferences=["bulk_frozen_storage"],
        hard_constraints={
            "size_variants": ["mini"],
            "min_capacity_lit": 150,
            "required_temperature_c": -24,
            "inverter": True,
        },
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == [
        "product_family",
        "budget",
        "usage_preferences",
    ]
    with pytest.raises(ValidationError):
        CoolerFreezerNeedProfile(product_family="refrigerator")
    with pytest.raises(ValidationError):
        CoolerFreezerNeedProfile(
            hard_constraints={"min_capacity_lit": 500, "max_capacity_lit": 300}
        )


def test_filter_contains_only_indexed_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "product_family": "freezer",
            "budget_max_vnd": 8_000_000,
            "soft_preferences": ["low_noise", "large_capacity"],
            "hard_constraints": {
                "brands": ["Sanaky"],
                "size_variants": ["mini"],
                "min_capacity_lit": 150,
                "required_temperature_c": -24,
                "max_width_cm": 60,
                "inverter": True,
                "gas_types": ["R600a"],
                "required_features": ["lock", "wheels"],
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.product_family",
        "metadata.effective_price_vnd",
        "metadata.is_mini",
        "metadata.total_capacity_lit",
        "metadata.temperature_min_c",
        "metadata.feature_tags",
    ):
        assert path in text
    assert "low_noise" not in text
    assert "large_capacity" not in text


def test_filter_matches_canonical_payload_and_rejects_missing_price() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="cooler-freezer-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )

    def point(point_id: int, metadata: dict[str, Any]) -> models.PointStruct:
        return models.PointStruct(
            id=point_id,
            vector=[1.0, 0.0],
            payload={"metadata": metadata},
        )

    base = {
        "product_family": "freezer",
        "product_type": "freezer_mini",
        "is_mini": True,
        "brand": "Sanaky",
        "total_capacity_lit": 200,
        "temperature_min_c": -25.0,
        "has_inverter": True,
        "gas_type": "R600a",
        "feature_tags": ["lock", "wheels"],
        "width_cm": 55.0,
    }
    client.upsert(
        collection_name="cooler-freezer-test",
        points=[
            point(1, {**base, "effective_price_vnd": 7_500_000}),
            point(2, {**base, "effective_price_vnd": 9_000_000}),
            point(3, base),
            point(
                4,
                {
                    **base,
                    "product_family": "cooler",
                    "product_type": "cooler_mini",
                    "effective_price_vnd": 7_000_000,
                },
            ),
            point(
                5,
                {
                    **base,
                    "feature_tags": ["lock"],
                    "effective_price_vnd": 7_000_000,
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="cooler-freezer-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "product_family": "freezer",
                "budget_max_vnd": 8_000_000,
                "hard_constraints": {
                    "brands": ["Sanaky"],
                    "size_variants": ["mini"],
                    "min_capacity_lit": 150,
                    "required_temperature_c": -24,
                    "max_width_cm": 60,
                    "inverter": True,
                    "gas_types": ["R600a"],
                    "required_features": ["lock", "wheels"],
                },
            }
        ),
        limit=10,
    )
    assert [item.id for item in result.points] == [1]


def test_candidate_normalizer_whitelists_cooler_freezer_fields() -> None:
    candidate = normalize_candidate(freezer_point())
    assert candidate["product_id"] == "freezer-1"
    assert candidate["effective_price_vnd"] == 7_500_000
    assert candidate["product_family"] == "freezer"
    assert candidate["product_type_label"] == "Tủ đông mini"
    assert candidate["total_capacity_lit"] == 200
    assert candidate["temperature_range_c"] == {"min": -25.0, "max": -18.0}
    assert candidate["has_inverter"] is True
    assert candidate["dimensions_cm"] == {
        "width": 55.0,
        "height": 85.0,
        "depth": 60.0,
    }
    assert "metadata" not in candidate


def test_profile_patch_operations_use_category_allowlist() -> None:
    profile, changed = apply_profile_patch(
        {
            "budget_max_vnd": 15_000_000,
            "usage_preferences": ["commercial_storage"],
            "hard_constraints": {"brands": ["Kangaroo"]},
        },
        ProfilePatch(
            replace={"hard_constraints.brands": ["Sanaky"]},
            remove={"usage_preferences": ["commercial_storage"]},
            add={"usage_preferences": ["bulk_frozen_storage"]},
            clear=["budget_max_vnd"],
            set={"household_size": 4},
        ),
        get_category_spec(),
    )
    assert profile["hard_constraints"]["brands"] == ["Sanaky"]
    assert profile["usage_preferences"] == ["bulk_frozen_storage"]
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
    removed = "metadata.total_capacity_lit"
    del client.schemas["tumattudong"][removed]
    assert ensure_payload_indexes(
        client, "tumattudong", config["payload_indexes"], apply=False
    ) == {removed: "integer"}
    assert client.created == []
    assert ensure_payload_indexes(
        client, "tumattudong", config["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "integer")]
    assert ensure_payload_indexes(
        client, "tumattudong", config["payload_indexes"], apply=True
    ) == {}


def test_prompts_and_no_match_text_are_cooler_freezer_specific() -> None:
    spec = get_category_spec()
    extraction = spec.build_need_extraction_prompt("Cần tủ đông đạt -24°C", {})
    ranking = spec.build_ranking_prompt(
        {"need_profile": {}, "hard_constraints": {}, "candidates": []}
    )
    response = spec.build_response_prompt(
        {"need_profile": {}, "selected_products": []}
    )
    combined = " ".join((extraction, ranking, response, no_match_answer({})))
    assert "tủ mát" in combined.casefold()
    assert "tủ đông" in combined.casefold()
    assert "nhiệt độ" in combined.casefold()
    assert "tủ lạnh" not in combined.casefold()
    assert "btu" not in combined.casefold()


def test_generic_query_interrupts_and_resumes_to_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "cooler-freezer-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi tủ mát hoặc tủ đông")]},
        config,
    )
    payload = interrupted["__interrupt__"][0].value
    assert payload["category"] == "cooler_freezer"
    assert [question["question_id"] for question in payload["questions"]] == [
        "product_family",
        "budget",
        "usage_preferences",
    ]

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "product_family", "option_id": "freezer"},
                    {"question_id": "budget", "option_id": "8m_15m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "bulk_frozen_storage",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["conversation"]["active_category"] == "cooler_freezer"
    assert completed["need_profile"]["product_family"] == "freezer"
    assert completed["need_profile"]["budget_max_vnd"] == 15_000_000
    assert completed["ranking"]["selected_products"][0]["product_id"] == "freezer-1"
    assert completed["response"]["answer"].startswith("Mình đề xuất")
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "tumattudong"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_product_family_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=CoolerFreezerCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Tôi cần tủ đông mini",
            product_family="freezer",
            hard_constraints={"size_variants": ["mini"]},
            confidence=1,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "cooler-freezer-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua tủ bảo quản")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "product_family",
                        "option_id": "other",
                        "custom_answer": "Tôi cần tủ đông mini",
                    },
                    {"question_id": "budget", "option_id": "8m_15m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "bulk_frozen_storage",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["product_family"] == "freezer"
    assert completed["need_profile"]["hard_constraints"]["size_variants"] == [
        "mini"
    ]
    assert completed["need_profile"]["custom_answers"]["product_family"][
        "raw_answer"
    ] == "Tôi cần tủ đông mini"


def test_switch_cooler_freezer_to_refrigerator_and_back_restores_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.COOLER_FREEZER,
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
                category=IntentLabel.COOLER_FREEZER,
                category_transition="switch",
                switch_evidence="tủ đông",
                action="switch_category",
            ),
        ],
        patches=[complete_cooler_freezer_patch(), complete_refrigerator_patch()],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "freezer-refrigerator-freezer"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn tủ đông")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua tủ lạnh")]}, config
    )
    assert switched["conversation"]["active_category"] == "refrigerator"
    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại tủ đông")]}, config
    )
    assert restored["conversation"]["active_category"] == "cooler_freezer"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2
