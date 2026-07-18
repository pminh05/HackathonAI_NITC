"""Behavior and data-contract tests for the desktop-computer category."""

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

from advisor.categories.desktop import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.desktop.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.desktop.normalizer import normalize_candidate
from advisor.categories.desktop.schemas import (
    DesktopCustomAnswer,
    DesktopNeedProfile,
)
from advisor.categories.desktop.setup_indexes import ensure_payload_indexes
from advisor.categories.registry import build_default_registry
from advisor.categories.refrigerator import load_config as load_refrigerator_config
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


ROOT = Path(__file__).parents[2]
INGESTION = runpy.run_path(ROOT / "ingestion/may_tinh_de_ban/changeName.py")
QDRANT = runpy.run_path(ROOT / "ingestion/may_tinh_de_ban/qdrant.py")
normalize_metadata = INGESTION["normalize_metadata"]


def desktop_point() -> Any:
    return SimpleNamespace(
        id="point-desktop-1",
        score=0.95,
        payload={
            "product_id": "desktop-1",
            "name": "Máy tính để bàn Rosa Test",
            "text": "Máy tính để bàn AMD, RAM 32 GB, SSD NVMe và card rời.",
            "image_path": "/public/may_tinh_de_ban.jpg",
            "metadata": {
                "category_scope": "desktop",
                "brand": "Rosa",
                "brand_key": "rosa",
                "model_code": "DESK-X",
                "original_price_vnd": 35_000_000,
                "promotional_price_vnd": 28_000_000,
                "desktop_form": "separate_unit",
                "cpu_vendor": "amd",
                "Công nghệ CPU": "AMD Ryzen 7",
                "Loại CPU": "7700",
                "toc_do_cpu_co_ban_ghz": 3.8,
                "toc_do_cpu_toi_da_ghz": 5.3,
                "so_nhan_cpu": 8,
                "so_luong_cpu_luong": 16,
                "ram_gb": 32,
                "ram_max_gb": 128,
                "ram_slots": 4,
                "Loại RAM": "DDR5",
                "storage_total_gb": 1024,
                "storage_type_tags": ["nvme", "ssd"],
                "Ổ cứng": "SSD 1 TB M.2 NVMe",
                "gpu_type": "discrete",
                "Chip đồ họa (GPU)": "NVIDIA GeForce RTX 4060",
                "vram_gpu_gb": 8,
                "os_family_tags": ["windows"],
                "Hệ điều hành": "Windows 11 Pro",
                "has_wifi": True,
                "has_bluetooth": True,
                "Wifi": "Wi-Fi 6 | Bluetooth 5.2",
                "Cổng kết nối": "HDMI | DisplayPort | USB-C",
                "dai_mm": 400,
                "rong_mm": 200,
                "day_mm": 450,
                "weight_kg": 8.5,
                "nguon_dien_max_w": 650,
                "release_year": 2025,
            },
        },
    )


class FakeStructuredModel:
    def __init__(self, owner: "FakeLLM", schema: type[Any]) -> None:
        self.owner = owner
        self.schema = schema

    def invoke(self, _: str) -> Any:
        if self.schema is TurnAnalysisResult:
            if self.owner.analyses:
                return self.owner.analyses.pop(0)
            return TurnAnalysisResult(
                category=IntentLabel.DESKTOP,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            product_id = "desktop-1" if "desktop-1" in _ else "sku-rf"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Đủ RAM, SSD và GPU cho nhu cầu.",
                        "trade_off": "Bộ máy không kèm màn hình.",
                    }
                ]
            )
        if self.schema is DesktopCustomAnswer:
            return self.owner.custom or DesktopCustomAnswer(
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
        custom: DesktopCustomAnswer | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.custom = custom

    def with_structured_output(
        self, schema: type[Any], **_: Any
    ) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, _: str) -> AIMessage:
        return AIMessage(
            content="Mình đề xuất máy tính để bàn phù hợp nhu cầu của bạn."
        )


class FakeQdrant:
    def __init__(self) -> None:
        config = load_config()
        self.payload_schema = {
            field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
            for field, schema in config["payload_indexes"].items()
        }
        refrigerator_config = load_refrigerator_config()
        self.payload_schemas = {
            config["collection"]: self.payload_schema,
            refrigerator_config["collection"]: {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in refrigerator_config["payload_indexes"].items()
            },
        }
        self.created: list[tuple[str, str]] = []
        self.query_kwargs: dict[str, Any] | None = None
        self.query_count = 0

    def get_collection(self, collection: str) -> Any:
        return SimpleNamespace(payload_schema=self.payload_schemas[collection])

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: Any,
        **_: Any,
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
        return SimpleNamespace(points=[desktop_point()])


def test_ingestion_normalizes_desktop_prices_storage_and_capabilities() -> None:
    metadata = normalize_metadata(
        {
            "brand": "ASUS",
            "giá gốc": "25000000.0",
            "giá khuyến mãi": "19.990.000 đ",
            "gia_goc_vnd": 250_000_000,
            "gia_khuyen_mai_vnd": 199_900_000,
            "Công nghệ CPU": "Intel Core i7 Raptor Lake",
            "Hệ điều hành": "Windows 11 Pro",
            "ram_gb": 32,
            "ram_toi_da_gb": 128,
            "Ổ cứng": "128 GB SSD M.2 NVMe + 1 TB HDD",
            "dung_luong_o_cung_gb": 128,
            "Thiết kế card": "Card đồ hoạ rời",
            "Wifi": "Wi-Fi 6 | Bluetooth 5.2",
            "Kích thước màn hình": "24 inch",
            "kich_thuoc_man_hinh_inch": 24,
            "nam_ra_mat": 2025,
        }
    )
    assert metadata["category_scope"] == "desktop"
    assert metadata["brand_key"] == "asus"
    assert metadata["original_price_vnd"] == 25_000_000
    assert metadata["promotional_price_vnd"] == 19_990_000
    assert "gia_goc_vnd" not in metadata
    assert metadata["desktop_form"] == "all_in_one"
    assert metadata["has_integrated_display"] is True
    assert metadata["cpu_vendor"] == "intel"
    assert metadata["os_family_tags"] == ["windows"]
    assert metadata["storage_total_gb"] == 1152
    assert metadata["storage_type_tags"] == ["nvme", "ssd", "hdd"]
    assert metadata["gpu_type"] == "discrete"
    assert metadata["has_wifi"] is True
    assert metadata["has_bluetooth"] is True
    assert "generated_price_ignored" in metadata["data_quality_flags"]

    separate = normalize_metadata({"brand": "Dell", "RAM": "16 GB"})
    assert separate["desktop_form"] == "separate_unit"
    assert separate["has_integrated_display"] is False
    assert "gpu_type" not in separate
    assert "has_wifi" not in separate


def test_ingestion_point_shape_uses_cloud_inference_and_stable_id() -> None:
    product = {
        "id": "desktop-sku",
        "name": "Máy tính để bàn Test",
        "text": "Máy tính để bàn RAM 16 GB, SSD 512 GB.",
        "image_path": "/public/may_tinh_de_ban.jpg",
        "metadata": {"category_scope": "desktop"},
    }
    first = next(QDRANT["point_generator"]([product]))
    second = next(QDRANT["point_generator"]([product]))
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


def test_spec_profile_questions_and_registry_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "desktop"
    assert spec.display_name == "Máy tính để bàn"
    assert config["collection"] == "maytinhdeban"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert build_default_registry().get_spec("desktop").name == "desktop"

    profile = DesktopNeedProfile(
        budget_max_vnd=30_000_000,
        form_preference="separate_unit",
        usage_preferences=["programming_multitasking"],
        hard_constraints={"min_ram_gb": 16, "storage_types": ["nvme"]},
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == [
        "primary_usage",
        "budget",
        "desktop_form",
    ]
    with pytest.raises(ValidationError):
        DesktopNeedProfile(usage_preferences=["daily_laundry"])


def test_filter_enforces_only_indexed_desktop_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "budget_max_vnd": 30_000_000,
            "form_preference": "separate_unit",
            "soft_preferences": ["quiet_operation", "many_ports"],
            "hard_constraints": {
                "brands": ["ASUS"],
                "cpu_vendors": ["amd"],
                "os_families": ["windows"],
                "storage_types": ["nvme"],
                "gpu_types": ["discrete"],
                "min_ram_gb": 16,
                "min_supported_ram_gb": 64,
                "min_storage_gb": 512,
                "requires_wifi": True,
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in load_config()["payload_fields"].values():
        if path != "metadata.screen_size_inch":
            assert path in text
    assert "quiet_operation" not in text
    assert "many_ports" not in text


def test_filter_matches_only_qualified_local_point() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="desktop-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    base = {
        "category_scope": "desktop",
        "brand_key": "rosa",
        "desktop_form": "separate_unit",
        "cpu_vendor": "amd",
        "os_family_tags": ["windows"],
        "ram_gb": 32,
        "ram_max_gb": 128,
        "storage_total_gb": 1024,
        "storage_type_tags": ["nvme", "ssd"],
        "gpu_type": "discrete",
        "has_wifi": True,
    }
    client.upsert(
        collection_name="desktop-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "original_price_vnd": 35_000_000,
                        "promotional_price_vnd": 28_000_000,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "ram_gb": 8,
                        "original_price_vnd": 20_000_000,
                    }
                },
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
                        "desktop_form": "all_in_one",
                        "original_price_vnd": 25_000_000,
                    }
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="desktop-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "budget_max_vnd": 30_000_000,
                "form_preference": "separate_unit",
                "hard_constraints": {
                    "brands": ["Rosa"],
                    "cpu_vendors": ["amd"],
                    "os_families": ["windows"],
                    "storage_types": ["nvme"],
                    "gpu_types": ["discrete"],
                    "min_ram_gb": 16,
                    "min_supported_ram_gb": 64,
                    "min_storage_gb": 512,
                    "requires_wifi": True,
                },
            }
        ),
        limit=10,
    )
    assert [point.id for point in result.points] == [1]


def test_normalizer_patch_indexes_prompts_and_no_match_are_desktop_specific() -> None:
    candidate = normalize_candidate(desktop_point())
    assert candidate["product_id"] == "desktop-1"
    assert candidate["effective_price_vnd"] == 28_000_000
    assert candidate["desktop_form"] == "separate_unit"
    assert candidate["cpu_vendor"] == "amd"
    assert candidate["ram_gb"] == 32
    assert candidate["storage_total_gb"] == 1024
    assert candidate["gpu_type"] == "discrete"
    assert candidate["has_wifi"] is True
    assert "metadata" not in candidate

    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "usage_preferences": ["office_study"],
            "hard_constraints": {"brands": []},
        },
        ProfilePatch(
            replace={"usage_preferences": ["gaming"]},
            add={"hard_constraints.brands": ["Rosa"]},
            set={"household_size": 5},
        ),
        spec,
    )
    assert profile["usage_preferences"] == ["gaming"]
    assert profile["hard_constraints"]["brands"] == ["Rosa"]
    assert "household_size" not in profile
    assert set(changed) == {"usage_preferences", "hard_constraints.brands"}

    fake = FakeQdrant()
    removed = "metadata.ram_gb"
    del fake.payload_schema[removed]
    assert ensure_payload_indexes(
        fake, "maytinhdeban", load_config()["payload_indexes"]
    ) == {removed: "integer"}
    assert ensure_payload_indexes(
        fake, "maytinhdeban", load_config()["payload_indexes"], apply=True
    ) == {}
    assert ensure_payload_indexes(
        fake, "maytinhdeban", load_config()["payload_indexes"], apply=True
    ) == {}
    assert fake.created == [(removed, "integer")]

    combined = " ".join(
        (
            spec.build_need_extraction_prompt("Cần desktop", {}),
            spec.build_ranking_prompt(
                {"need_profile": {}, "hard_constraints": {}, "candidates": []}
            ),
            spec.build_response_prompt(
                {"need_profile": {}, "selected_products": []}
            ),
            no_match_answer({}),
        )
    ).casefold()
    assert "máy tính để bàn" in combined
    assert "cpu" in combined
    assert "máy giặt" not in combined
    assert "khối lượng giặt" not in combined
    assert "btu" not in combined


def test_generic_desktop_query_interrupts_and_resumes_to_advice() -> None:
    qdrant = FakeQdrant()
    graph = build_graph(llm=FakeLLM(), qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "desktop-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy tính để bàn")]}, config
    )
    assert [
        item["question_id"]
        for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["primary_usage", "budget", "desktop_form"]

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "primary_usage",
                        "option_id": "programming_multitasking",
                    },
                    {"question_id": "budget", "option_id": "15m_30m"},
                    {
                        "question_id": "desktop_form",
                        "option_id": "separate_unit",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["conversation"]["active_category"] == "desktop"
    assert completed["need_profile"]["budget_max_vnd"] == 30_000_000
    assert completed["need_profile"]["form_preference"] == "separate_unit"
    assert completed["ranking"]["selected_products"][0]["product_id"] == "desktop-1"
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "maytinhdeban"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_usage_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=DesktopCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Tôi chủ yếu lập trình và chạy nhiều máy ảo",
            usage_preferences=["programming_multitasking"],
            soft_preferences=["high_ram", "ram_upgrade"],
            confidence=1,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "desktop-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua desktop")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "primary_usage",
                        "option_id": "other",
                        "custom_answer": "Tôi chủ yếu lập trình và chạy nhiều máy ảo",
                    },
                    {"question_id": "budget", "option_id": "15m_30m"},
                    {"question_id": "desktop_form", "option_id": "flexible"},
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["usage_preferences"] == [
        "programming_multitasking"
    ]
    assert completed["need_profile"]["custom_answers"]["primary_usage"][
        "raw_answer"
    ] == "Tôi chủ yếu lập trình và chạy nhiều máy ảo"


def test_switch_desktop_refrigerator_and_back_restores_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.DESKTOP,
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
                category=IntentLabel.DESKTOP,
                category_transition="switch",
                switch_evidence="máy tính để bàn",
                action="switch_category",
            ),
        ],
        patches=[
            ProfilePatch(
                set={
                    "budget_max_vnd": 30_000_000,
                    "form_preference": "separate_unit",
                },
                replace={"usage_preferences": ["programming_multitasking"]},
            ),
            ProfilePatch(
                set={"household_size": 4, "budget_max_vnd": 20_000_000},
                replace={"usage_preferences": ["weekly_storage"]},
            ),
        ],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "desktop-refrigerator-desktop"}}

    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy tính để bàn")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua tủ lạnh")]}, config
    )
    assert switched["conversation"]["active_category"] == "refrigerator"
    assert switched["ranking"]["selected_products"][0]["product_id"] == "sku-rf"

    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại máy tính để bàn")]}, config
    )
    assert restored["conversation"]["active_category"] == "desktop"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2
