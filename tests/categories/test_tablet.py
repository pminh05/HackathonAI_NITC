"""Behavior and data-contract tests for the tablet category."""

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

from advisor.categories.registry import build_default_registry
from advisor.categories.tablet import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.tablet.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.tablet.normalizer import normalize_candidate
from advisor.categories.tablet.schemas import TabletNeedProfile
from advisor.categories.tablet.setup_indexes import ensure_payload_indexes
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


ROOT = Path(__file__).parents[2]
INGESTION = runpy.run_path(ROOT / "ingestion/may_tinh_bang/changeName.py")
QDRANT = runpy.run_path(ROOT / "ingestion/may_tinh_bang/qdrant.py")
normalize_metadata = INGESTION["normalize_metadata"]


def tablet_point() -> Any:
    return SimpleNamespace(
        id="point-tab-1",
        score=0.94,
        payload={
            "product_id": "tab-1",
            "name": "Máy tính bảng Samsung Test",
            "text": "Máy tính bảng Android 8 GB RAM, 5G.",
            "image_path": "/public/may_tinh_bang.jpg",
            "metadata": {
                "category_scope": "tablet",
                "brand": "Samsung",
                "model_code": "TAB-X",
                "original_price_vnd": 18_000_000,
                "promotional_price_vnd": 15_000_000,
                "os_family": "android",
                "Hệ điều hành": "Android 15",
                "ram_gb": 8,
                "storage_gb": 256,
                "available_storage_gb": 230.5,
                "screen_size_inch": 11.0,
                "display_family": "oled_amoled",
                "Màn hình hiển thị": "Dynamic AMOLED 2X",
                "weight_g": 500,
                "connectivity_class": "cellular",
                "max_mobile_generation": 5,
                "supports_calls": True,
                "supports_memory_card": True,
                "battery_mah": 8000,
                "max_charging_w": 45.0,
                "Chip xử lý (CPU)": "Snapdragon Test",
                "feature_tags": ["stylus", "keyboard"],
            },
        },
    )


class FakeStructuredModel:
    def __init__(self, schema: type[Any]) -> None:
        self.schema = schema

    def invoke(self, _: str) -> Any:
        if self.schema is TurnAnalysisResult:
            return TurnAnalysisResult(
                category=IntentLabel.TABLET,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return ProfilePatch()
        if self.schema is RankingResult:
            return RankingResult(
                selected_products=[
                    {
                        "product_id": "tab-1",
                        "reason": "Đúng nhu cầu kết nối và hiệu năng.",
                        "trade_off": "Khối lượng tương đối cao.",
                    }
                ]
            )
        raise AssertionError(self.schema)


class FakeLLM:
    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(schema)

    def invoke(self, _: str) -> AIMessage:
        return AIMessage(content="Mình đề xuất máy tính bảng phù hợp nhu cầu của bạn.")


class FakeQdrant:
    def __init__(self) -> None:
        config = load_config()
        self.payload_schema = {
            field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
            for field, schema in config["payload_indexes"].items()
        }
        self.created: list[tuple[str, str]] = []
        self.query_kwargs: dict[str, Any] | None = None

    def get_collection(self, _: str) -> Any:
        return SimpleNamespace(payload_schema=self.payload_schema)

    def create_payload_index(
        self, *, field_name: str, field_schema: Any, **_: Any
    ) -> None:
        value = getattr(field_schema, "value", str(field_schema))
        self.payload_schema[field_name] = SimpleNamespace(
            data_type=SimpleNamespace(value=value)
        )
        self.created.append((field_name, str(value)))

    def query_points(self, **kwargs: Any) -> Any:
        self.query_kwargs = kwargs
        return SimpleNamespace(points=[tablet_point()])


def test_ingestion_normalizes_prices_and_tablet_fields() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Samsung",
            "giá gốc": "15990000.0",
            "giá khuyến mãi": "14.990.000 đ",
            "gia_goc_vnd": 159_900_000,
            "gia_khuyen_mai_vnd": 149_900_000,
            "Hệ điều hành": "Android 15",
            "Màn hình hiển thị": "Dynamic AMOLED 2X",
            "ram_gb": 8,
            "bo_nho_luu_tru_gb": 256,
            "kich_thuoc_man_hinh_inch": 11,
            "khoi_luong_g": 500,
            "Mạng di động": "Hỗ trợ 5G",
            "Thực hiện cuộc gọi": "Có",
            "Thẻ nhớ": "Micro SD, hỗ trợ tối đa 1 TB",
            "Tính năng đặc biệt": "Kết nối bút S Pen | Kết nối bàn phím rời",
        }
    )
    assert metadata["original_price_vnd"] == 15_990_000
    assert metadata["promotional_price_vnd"] == 14_990_000
    assert "gia_goc_vnd" not in metadata
    assert metadata["os_family"] == "android"
    assert metadata["display_family"] == "oled_amoled"
    assert metadata["connectivity_class"] == "cellular"
    assert metadata["max_mobile_generation"] == 5
    assert metadata["supports_calls"] is True
    assert metadata["supports_memory_card"] is True
    assert metadata["feature_tags"] == ["stylus", "keyboard"]


def test_ingestion_point_shape_uses_cloud_inference_and_stable_id() -> None:
    product = {
        "id": "tab-sku",
        "name": "Máy tính bảng Test",
        "text": "Máy tính bảng Android 8 GB RAM.",
        "image_path": "/public/may_tinh_bang.jpg",
        "metadata": {"category_scope": "tablet"},
    }
    first = next(QDRANT["point_generator"]([product]))
    second = next(QDRANT["point_generator"]([product]))
    assert first.id == second.id
    assert isinstance(first.vector, models.Document)
    assert first.vector.model == "intfloat/multilingual-e5-small"
    assert first.vector.text.startswith("passage: ")


def test_spec_profile_and_registry_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "tablet"
    assert spec.display_name == "Máy tính bảng"
    assert config["collection"] == "maytinhbang"
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert build_default_registry().get_spec("tablet").name == "tablet"
    profile = TabletNeedProfile(
        budget_max_vnd=20_000_000,
        connectivity_segment="cellular_5g",
        usage_preferences=["study_work"],
        hard_constraints={"min_ram_gb": 8},
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == ["primary_usage", "budget", "connectivity"]
    with pytest.raises(ValidationError):
        TabletNeedProfile(usage_preferences=["frozen_storage"])


def test_filter_enforces_hard_fields_and_not_soft_preferences() -> None:
    query_filter = build_filter(
        {
            "budget_max_vnd": 20_000_000,
            "connectivity_segment": "cellular_5g",
            "soft_preferences": ["stylus", "keyboard"],
            "hard_constraints": {
                "brands": ["Samsung"],
                "os_families": ["android"],
                "display_families": ["oled_amoled"],
                "min_ram_gb": 8,
                "min_storage_gb": 256,
                "max_weight_g": 600,
                "requires_memory_card": True,
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.category_scope",
        "metadata.brand",
        "metadata.os_family",
        "metadata.max_mobile_generation",
        "metadata.ram_gb",
        "metadata.storage_gb",
        "metadata.weight_g",
        "metadata.supports_memory_card",
    ):
        assert path in text
    assert "stylus" not in text
    assert "keyboard" not in text


def test_filter_matches_only_qualified_local_point() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="tablet-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    base = {
        "category_scope": "tablet",
        "brand": "Samsung",
        "os_family": "android",
        "ram_gb": 8,
        "storage_gb": 256,
        "screen_size_inch": 11.0,
        "weight_g": 500,
        "connectivity_class": "cellular",
        "max_mobile_generation": 5,
        "supports_calls": True,
        "supports_memory_card": True,
        "display_family": "oled_amoled",
    }
    client.upsert(
        collection_name="tablet-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "original_price_vnd": 18_000_000,
                        "promotional_price_vnd": 15_000_000,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={"metadata": {**base, "ram_gb": 4, "original_price_vnd": 9_000_000}},
            ),
            models.PointStruct(id=3, vector=[1.0, 0.0], payload={"metadata": base}),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="tablet-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "budget_max_vnd": 20_000_000,
                "connectivity_segment": "cellular_5g",
                "hard_constraints": {"min_ram_gb": 8, "min_storage_gb": 256},
            }
        ),
        limit=10,
    )
    assert [point.id for point in result.points] == [1]


def test_normalizer_patch_indexes_and_prompts() -> None:
    candidate = normalize_candidate(tablet_point())
    assert candidate["effective_price_vnd"] == 15_000_000
    assert candidate["ram_gb"] == 8
    assert candidate["feature_tags"] == ["stylus", "keyboard"]
    assert "metadata" not in candidate
    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {"usage_preferences": ["study_work"], "hard_constraints": {"brands": []}},
        ProfilePatch(
            add={"hard_constraints.brands": ["Samsung"]},
            replace={"usage_preferences": ["gaming"]},
            set={"household_size": 4},
        ),
        spec,
    )
    assert profile["usage_preferences"] == ["gaming"]
    assert profile["hard_constraints"]["brands"] == ["Samsung"]
    assert "household_size" not in profile
    assert set(changed) == {"usage_preferences", "hard_constraints.brands"}
    fake = FakeQdrant()
    removed = "metadata.ram_gb"
    del fake.payload_schema[removed]
    assert ensure_payload_indexes(fake, "maytinhbang", load_config()["payload_indexes"]) == {removed: "integer"}
    assert ensure_payload_indexes(fake, "maytinhbang", load_config()["payload_indexes"], apply=True) == {}
    combined = " ".join(
        (
            spec.build_need_extraction_prompt("Cần tablet", {}),
            spec.build_ranking_prompt({"need_profile": {}, "hard_constraints": {}, "candidates": []}),
            spec.build_response_prompt({"need_profile": {}, "selected_products": []}),
            no_match_answer({}),
        )
    ).casefold()
    assert "máy tính bảng" in combined
    assert "tủ lạnh" not in combined


def test_graph_interrupts_and_resumes_to_tablet_collection() -> None:
    qdrant = FakeQdrant()
    graph = build_graph(llm=FakeLLM(), qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "tablet-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy tính bảng")]}, config
    )
    assert [
        item["question_id"] for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["primary_usage", "budget", "connectivity"]
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "primary_usage", "option_id": "study_work"},
                    {"question_id": "budget", "option_id": "10m_20m"},
                    {"question_id": "connectivity", "option_id": "cellular_5g"},
                ]
            }
        ),
        config,
    )
    assert completed["conversation"]["active_category"] == "tablet"
    assert completed["need_profile"]["budget_max_vnd"] == 20_000_000
    assert completed["ranking"]["selected_products"][0]["product_id"] == "tab-1"
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "maytinhbang"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"
