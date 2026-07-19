"""Behavior and data-contract tests for the computer-monitor category."""

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

from advisor.categories.monitor import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.monitor.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.monitor.normalizer import normalize_candidate
from advisor.categories.monitor.schemas import MonitorCustomAnswer, MonitorNeedProfile
from advisor.categories.monitor.setup_indexes import ensure_payload_indexes
from advisor.categories.registry import build_default_registry
from advisor.categories.washing_machine import load_config as load_washing_config
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


INGESTION_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/man_hinh_may_tinh/changeName.py"
)
normalize_metadata = INGESTION_MODULE["normalize_metadata"]
QDRANT_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/man_hinh_may_tinh/qdrant.py"
)


def monitor_point() -> SimpleNamespace:
    return SimpleNamespace(
        id="point-monitor-1",
        score=0.94,
        payload={
            "product_id": "monitor-1",
            "name": "Màn hình Test 27 inch QHD",
            "text": "Màn hình 27 inch QHD, tấm nền IPS, có HDMI và USB-C.",
            "image_path": "/public/man_hinh_may_tinh.jpg",
            "metadata": {
                "category_scope": "monitor",
                "brand": "Test",
                "brand_key": "test",
                "original_price_vnd": 6_490_000,
                "promotional_price_vnd": 5_490_000,
                "screen_size_inch": 27.0,
                "Độ phân giải": "QHD (2560 x 1440)",
                "resolution_key": "2560x1440",
                "resolution_width_px": 2560,
                "resolution_height_px": 1440,
                "panel_family": "ips",
                "panel_variant": "Fast IPS",
                "screen_shape": "flat",
                "response_time_ms": 1.0,
                "response_time_metric": "gtg",
                "brightness_nits": 350.0,
                "srgb_coverage_pct": 100.0,
                "dci_p3_coverage_pct": 95.0,
                "connection_tags": ["hdmi", "displayport", "usb_c"],
                "Kết nối": "HDMI 2.0 | DisplayPort 1.4 | USB Type-C",
                "feature_tags": ["freesync", "flicker_free", "low_blue_light"],
                "Màn hình hiển thị": "AMD FreeSync | Flicker-free",
                "Tiện ích": "Điều chỉnh được độ cao",
                "has_speakers": True,
                "has_vesa_mount": True,
                "has_touch": False,
                "width_mm": 614.0,
                "height_min_mm": 390.0,
                "height_max_mm": 530.0,
                "depth_mm": 220.0,
                "weight_kg": 5.2,
                "power_consumption_w": 28.0,
            },
        },
    )


def washing_point() -> SimpleNamespace:
    return SimpleNamespace(
        id="point-wm-switch",
        score=0.9,
        payload={
            "product_id": "wm-switch",
            "name": "Máy giặt chuyển ngành",
            "text": "Máy giặt 10 kg cửa trước.",
            "image_path": "/public/may_giat.jpg",
            "metadata": {
                "brand": "Test",
                "product_type": "front_load",
                "drum_type": "horizontal",
                "original_price_vnd": 10_000_000,
                "wash_capacity_kg": 10.0,
                "people_min": 3,
                "people_max": 5,
                "has_inverter": True,
                "has_dryer": False,
            },
        },
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
                category=IntentLabel.MONITOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            product_id = "wm-switch" if "wm-switch" in prompt else "monitor-1"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Phù hợp kích thước, độ phân giải và nhu cầu.",
                        "trade_off": "Chưa có dữ liệu tần số quét để xác nhận.",
                    }
                ]
            )
        if self.schema is MonitorCustomAnswer:
            return self.owner.custom or MonitorCustomAnswer(
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
        custom: MonitorCustomAnswer | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.custom = custom
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        return AIMessage(content="Mình đề xuất màn hình phù hợp nhu cầu của bạn.")


class FakeQdrant:
    def __init__(self) -> None:
        monitor_config = load_config()
        washing_config = load_washing_config()
        self.payload_schema = {
            field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
            for field, schema in monitor_config["payload_indexes"].items()
        }
        self.payload_schemas = {
            monitor_config["collection"]: self.payload_schema,
            washing_config["collection"]: {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in washing_config["payload_indexes"].items()
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
        point = (
            washing_point()
            if kwargs["collection_name"] == "maygiat"
            else monitor_point()
        )
        return SimpleNamespace(points=[point])


def test_ingestion_normalizes_monitor_prices_panel_capabilities_and_dimensions() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Gigabyte",
            "giá gốc": "6490000.0",
            "giá khuyến mãi": "5.490.000 đ",
            "gia_goc_vnd": 64_900_000,
            "gia_khuyen_mai_vnd": 54_900_000,
            "kich_thuoc_man_hinh_inch": 27,
            "do_phan_giai_ngang_px": 2560,
            "do_phan_giai_doc_px": 1440,
            "Tấm nền": "Rapid IPS",
            "Loại màn hình": "Cong",
            "thoi_gian_dap_ung_ms": 1,
            "loai_thoi_gian_dap_ung": "MPRT",
            "do_sang_cd_m2": 350,
            "do_phu_srgb_pct": 105,
            "do_phu_dci_p3_pct": 95,
            "Kết nối": (
                "2 x HDMI 2.0 | 1 x DisplayPort 1.4 | USB Type-C | "
                "Thunderbolt 4 | Jack tai nghe"
            ),
            "Màn hình hiển thị": "AMD FreeSync | Anti-Flicker | HDR10",
            "Tiện ích": "Điều chỉnh độ cao | Webcam",
            "Loa": "Không có",
            "Vesa": "Có",
            "Màn hình cảm ứng": "Không cảm ứng",
            "ngang_mm": 614,
            "cao_min_mm": 390,
            "cao_max_mm": 530,
            "day_mm": 220,
            "khoi_luong_kg": 5.2,
            "dien_nang_tieu_thu_w": 28,
        }
    )
    assert metadata["category_scope"] == "monitor"
    assert metadata["brand_key"] == "gigabyte"
    assert metadata["original_price_vnd"] == 6_490_000
    assert metadata["promotional_price_vnd"] == 5_490_000
    assert "gia_goc_vnd" not in metadata
    assert metadata["resolution_key"] == "2560x1440"
    assert metadata["panel_family"] == "ips"
    assert metadata["panel_variant"] == "Rapid IPS"
    assert metadata["screen_shape"] == "curved"
    assert metadata["response_time_metric"] == "mprt"
    assert metadata["connection_tags"] == [
        "hdmi",
        "displayport",
        "usb_c",
        "thunderbolt",
        "audio_out",
    ]
    assert set(metadata["feature_tags"]) >= {
        "freesync",
        "flicker_free",
        "hdr",
        "height_adjust",
        "webcam",
    }
    assert metadata["has_speakers"] is False
    assert metadata["has_vesa_mount"] is True
    assert metadata["has_touch"] is False
    assert "refresh_rate_hz" not in metadata
    assert "generated_price_ignored" in metadata["data_quality_flags"]


def test_ingestion_point_shape_uses_cloud_inference_and_stable_payload() -> None:
    product = {
        "id": "sku-monitor",
        "name": "Màn hình Test",
        "text": "Màn hình 27 inch QHD tấm nền IPS.",
        "image_path": "/public/man_hinh_may_tinh.jpg",
        "metadata": {"category_scope": "monitor"},
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


def test_spec_profile_questions_and_registry_are_monitor_specific() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "monitor"
    assert spec.display_name == "Màn hình máy tính"
    assert config["collection"] == "manhinhmaytinh"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert build_default_registry().get_spec("monitor").name == "monitor"

    profile = MonitorNeedProfile(
        budget_segment="open",
        screen_size_preference="standard",
        usage_preferences=["programming_multitasking"],
        hard_constraints={
            "panel_families": ["ips"],
            "required_connections": ["hdmi", "usb_c"],
        },
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == ["primary_usage", "budget", "screen_size"]
    with pytest.raises(ValidationError):
        MonitorNeedProfile(usage_preferences=["daily_laundry"])


def test_filter_contains_only_indexed_monitor_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "budget_max_vnd": 8_000_000,
            "soft_preferences": ["high_refresh_rate", "color_accuracy"],
            "hard_constraints": {
                "brands": ["LG"],
                "panel_families": ["ips"],
                "screen_shapes": ["flat"],
                "resolution_keys": ["2560x1440"],
                "required_connections": ["hdmi", "usb_c"],
                "required_features": ["flicker_free"],
                "response_time_metrics": ["gtg"],
                "min_screen_size_inch": 27,
                "min_resolution_width_px": 2560,
                "min_resolution_height_px": 1440,
                "max_response_time_ms": 2,
                "min_brightness_nits": 300,
                "min_srgb_coverage_pct": 99,
                "requires_vesa": True,
                "max_width_mm": 650,
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.category_scope",
        "metadata.panel_family",
        "metadata.resolution_key",
        "metadata.connection_tags",
        "metadata.response_time_ms",
        "metadata.has_vesa_mount",
    ):
        assert path in text
    assert text.count("metadata.connection_tags") == 2
    assert "high_refresh_rate" not in text
    assert "refresh_rate" not in text


def test_filter_matches_only_fully_qualified_local_point() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="monitor-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    base = {
        "category_scope": "monitor",
        "brand_key": "test",
        "screen_size_inch": 27.0,
        "resolution_key": "2560x1440",
        "resolution_width_px": 2560,
        "resolution_height_px": 1440,
        "panel_family": "ips",
        "screen_shape": "flat",
        "response_time_ms": 1.0,
        "response_time_metric": "gtg",
        "brightness_nits": 350.0,
        "srgb_coverage_pct": 100.0,
        "connection_tags": ["hdmi", "displayport", "usb_c"],
        "feature_tags": ["flicker_free"],
        "has_vesa_mount": True,
        "width_mm": 614.0,
    }
    client.upsert(
        collection_name="monitor-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "original_price_vnd": 6_490_000,
                        "promotional_price_vnd": 5_490_000,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "connection_tags": ["hdmi"],
                        "original_price_vnd": 5_000_000,
                    }
                },
            ),
            models.PointStruct(id=3, vector=[1.0, 0.0], payload={"metadata": base}),
            models.PointStruct(
                id=4,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "category_scope": "desktop",
                        "original_price_vnd": 5_000_000,
                    }
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="monitor-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "budget_max_vnd": 6_000_000,
                "hard_constraints": {
                    "brands": ["Test"],
                    "panel_families": ["ips"],
                    "resolution_keys": ["2560x1440"],
                    "required_connections": ["hdmi", "usb_c"],
                    "required_features": ["flicker_free"],
                    "max_response_time_ms": 2,
                    "requires_vesa": True,
                },
            }
        ),
        limit=10,
    )
    assert [point.id for point in result.points] == [1]


def test_normalizer_patch_indexes_prompts_and_no_match_are_monitor_specific() -> None:
    candidate = normalize_candidate(monitor_point())
    assert candidate["product_id"] == "monitor-1"
    assert candidate["effective_price_vnd"] == 5_490_000
    assert candidate["screen_size_inch"] == 27.0
    assert candidate["resolution_key"] == "2560x1440"
    assert candidate["panel_family"] == "ips"
    assert candidate["connection_tags"] == ["hdmi", "displayport", "usb_c"]
    assert candidate["has_vesa_mount"] is True
    assert candidate["dimensions_mm"]["width"] == 614.0
    assert "metadata" not in candidate

    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "usage_preferences": ["office_study"],
            "hard_constraints": {"brands": []},
        },
        ProfilePatch(
            replace={"usage_preferences": ["gaming"]},
            add={"hard_constraints.brands": ["LG"]},
            set={"household_size": 5},
        ),
        spec,
    )
    assert profile["usage_preferences"] == ["gaming"]
    assert profile["hard_constraints"]["brands"] == ["LG"]
    assert "household_size" not in profile
    assert set(changed) == {"usage_preferences", "hard_constraints.brands"}

    fake = FakeQdrant()
    removed = "metadata.panel_family"
    del fake.payload_schema[removed]
    assert ensure_payload_indexes(
        fake, "manhinhmaytinh", load_config()["payload_indexes"]
    ) == {removed: "keyword"}
    assert ensure_payload_indexes(
        fake, "manhinhmaytinh", load_config()["payload_indexes"], apply=True
    ) == {}
    assert ensure_payload_indexes(
        fake, "manhinhmaytinh", load_config()["payload_indexes"], apply=True
    ) == {}
    assert fake.created == [(removed, "keyword")]

    combined = " ".join(
        (
            spec.build_need_extraction_prompt("Cần màn hình 27 inch", {}),
            spec.build_ranking_prompt(
                {"need_profile": {}, "hard_constraints": {}, "candidates": []}
            ),
            spec.build_response_prompt(
                {"need_profile": {}, "selected_products": []}
            ),
            no_match_answer({"budget_max_vnd": 6_000_000}),
        )
    ).casefold()
    assert "màn hình máy tính" in combined
    assert "độ phân giải" in combined
    assert "tần số quét" in combined
    assert "khối lượng giặt" not in combined
    assert "btu" not in combined
    assert "chưa có giá xác minh" in combined


def test_generic_monitor_query_interrupts_and_resumes_to_advice() -> None:
    qdrant = FakeQdrant()
    graph = build_graph(llm=FakeLLM(), qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "monitor-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn màn hình máy tính")]}, config
    )
    assert [
        item["question_id"]
        for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["primary_usage", "budget", "screen_size"]

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "primary_usage", "option_id": "gaming"},
                    {"question_id": "budget", "option_id": "3m_6m"},
                    {"question_id": "screen_size", "option_id": "standard"},
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["conversation"]["active_category"] == "monitor"
    assert completed["need_profile"]["usage_preferences"] == ["gaming"]
    assert completed["need_profile"]["budget_max_vnd"] == 6_000_000
    assert completed["need_profile"]["screen_size_preference"] == "standard"
    assert completed["ranking"]["selected_products"][0]["product_id"] == "monitor-1"
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "manhinhmaytinh"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_screen_size_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=MonitorCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Tôi muốn màn 29 inch",
            preferred_screen_size_inch=29,
            confidence=1,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "monitor-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua màn hình")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "primary_usage", "option_id": "office_study"},
                    {"question_id": "budget", "option_id": "open"},
                    {
                        "question_id": "screen_size",
                        "option_id": "other",
                        "custom_answer": "Tôi muốn màn 29 inch",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["preferred_screen_size_inch"] == 29
    assert completed["need_profile"]["custom_answers"]["screen_size"][
        "raw_answer"
    ] == "Tôi muốn màn 29 inch"


def test_switch_monitor_washing_machine_and_back_restores_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.MONITOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.WASHING_MACHINE,
                category_transition="switch",
                switch_evidence="máy giặt",
                action="switch_category",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.MONITOR,
                category_transition="switch",
                switch_evidence="màn hình máy tính",
                action="switch_category",
            ),
        ],
        patches=[
            ProfilePatch(
                set={
                    "budget_segment": "open",
                    "screen_size_preference": "standard",
                },
                replace={"usage_preferences": ["programming_multitasking"]},
            ),
            ProfilePatch(
                set={"household_size": 5, "budget_max_vnd": 12_000_000},
                replace={"usage_preferences": ["daily_laundry"]},
            ),
        ],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "monitor-washing-monitor"}}

    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn màn hình máy tính")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua máy giặt")]}, config
    )
    assert switched["conversation"]["active_category"] == "washing_machine"
    assert switched["ranking"]["selected_products"][0]["product_id"] == "wm-switch"

    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại màn hình máy tính")]}, config
    )
    assert restored["conversation"]["active_category"] == "monitor"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2
