"""Behavior and data-contract tests for the printer category."""

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

from advisor.categories.printer import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.printer.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.printer.normalizer import normalize_candidate
from advisor.categories.printer.schemas import PrinterNeedProfile
from advisor.categories.printer.setup_indexes import ensure_payload_indexes
from advisor.categories.registry import build_default_registry
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


ROOT = Path(__file__).parents[2]
INGESTION = runpy.run_path(ROOT / "ingestion/may_in/changeName.py")
QDRANT = runpy.run_path(ROOT / "ingestion/may_in/qdrant.py")
normalize_metadata = INGESTION["normalize_metadata"]


def printer_point() -> Any:
    return SimpleNamespace(
        id="point-pr-1",
        score=0.93,
        payload={
            "product_id": "pr-1",
            "name": "Máy in HP Test",
            "text": "Máy in laser màu, Wi-Fi, công suất 2.000 trang/tháng.",
            "image_path": "/public/may_in.jpg",
            "metadata": {
                "category_scope": "printer",
                "brand": "Hp",
                "model_code": "PR-X",
                "original_price_vnd": 8_000_000,
                "promotional_price_vnd": 7_000_000,
                "print_technology": "laser",
                "color_mode": "color",
                "Loại sản phẩm": "In laser màu",
                "print_speed_ppm": 25.0,
                "monthly_volume_min_pages": 500,
                "monthly_volume_max_pages": 2000,
                "resolution_horizontal_dpi": 1200,
                "resolution_vertical_dpi": 1200,
                "connection_tags": ["wifi", "lan", "usb", "mobile_print"],
                "paper_size_tags": ["a4", "a5"],
                "os_tags": ["windows", "macos"],
                "supports_duplex": True,
                "paper_input_sheets": 250,
                "toner_yield_min_pages": 1500,
                "toner_yield_max_pages": 2500,
                "width_mm": 420,
                "height_mm": 250,
                "depth_mm": 360,
                "Loại mực in": "Mực thử nghiệm",
                "Tương thích": "Windows | macOS",
            },
        },
    )


class FakeStructuredModel:
    def __init__(self, schema: type[Any]) -> None:
        self.schema = schema

    def invoke(self, _: str) -> Any:
        if self.schema is TurnAnalysisResult:
            return TurnAnalysisResult(
                category=IntentLabel.PRINTER,
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
                        "product_id": "pr-1",
                        "reason": "Đúng nhu cầu in màu và khối lượng.",
                        "trade_off": "Chi phí đầu tư cao hơn máy đơn sắc.",
                    }
                ]
            )
        raise AssertionError(self.schema)


class FakeLLM:
    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(schema)

    def invoke(self, _: str) -> AIMessage:
        return AIMessage(content="Mình đề xuất máy in phù hợp nhu cầu của bạn.")


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
        return SimpleNamespace(points=[printer_point()])


def test_ingestion_normalizes_prices_types_and_tags() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Hp",
            "giá gốc": "8190000.0",
            "giá khuyến mãi": "7.590.000 đ",
            "gia_goc_vnd": 81_900_000,
            "gia_khuyen_mai_vnd": 75_900_000,
            "Loại sản phẩm": "In laser màu",
            "Kết nối": "Wi-Fi Direct | In bằng điện thoại",
            "Cổng kết nối": "USB 2.0 | LAN",
            "Công nghệ": "Apple AirPrint",
            "Kích thước phụ kiện": "A4 | A5 | Letter",
            "Tương thích": "Windows | macOS | Android",
            "Loại giấy in 2 mặt": "Giấy thường",
            "cong_suat_thang_min_trang": 500,
            "cong_suat_thang_max_trang": 2000,
            "toc_do_in_trang_phut": 25,
            "dai_mm": 420,
            "rong_mm": 360,
            "cao_mm": 250,
        }
    )
    assert metadata["original_price_vnd"] == 8_190_000
    assert metadata["promotional_price_vnd"] == 7_590_000
    assert "gia_goc_vnd" not in metadata
    assert metadata["print_technology"] == "laser"
    assert metadata["color_mode"] == "color"
    assert metadata["connection_tags"] == [
        "wifi",
        "wifi_direct",
        "lan",
        "usb",
        "mobile_print",
    ]
    assert metadata["paper_size_tags"] == ["a4", "a5", "letter"]
    assert metadata["os_tags"] == ["windows", "macos", "android"]
    assert metadata["supports_duplex"] is True


def test_ingestion_point_shape_uses_cloud_inference_and_stable_id() -> None:
    product = {
        "id": "pr-sku",
        "name": "Máy in Test",
        "text": "Máy in laser màu có Wi-Fi.",
        "image_path": "/public/may_in.jpg",
        "metadata": {"category_scope": "printer"},
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
    assert spec.name == config["category"] == "printer"
    assert spec.display_name == "Máy in"
    assert config["collection"] == "mayin"
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert build_default_registry().get_spec("printer").name == "printer"
    profile = PrinterNeedProfile(
        budget_max_vnd=10_000_000,
        monthly_volume_segment="office",
        usage_preferences=["color_documents"],
        hard_constraints={"color_modes": ["color"]},
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == ["print_purpose", "monthly_volume", "budget"]
    with pytest.raises(ValidationError):
        PrinterNeedProfile(usage_preferences=["weekly_storage"])


def test_filter_enforces_hard_fields_and_ands_required_tags() -> None:
    query_filter = build_filter(
        {
            "budget_max_vnd": 10_000_000,
            "monthly_pages_estimate": 1000,
            "soft_preferences": ["scan", "copy"],
            "hard_constraints": {
                "brands": ["Hp"],
                "technologies": ["laser"],
                "color_modes": ["color"],
                "min_print_speed_ppm": 20,
                "required_connections": ["wifi", "lan"],
                "required_paper_sizes": ["a4"],
                "requires_duplex": True,
            },
        }
    )
    dumped = query_filter.model_dump(mode="json", exclude_none=True)
    text = str(dumped)
    for path in (
        "metadata.category_scope",
        "metadata.brand",
        "metadata.print_technology",
        "metadata.color_mode",
        "metadata.monthly_volume_max_pages",
        "metadata.print_speed_ppm",
        "metadata.connection_tags",
        "metadata.paper_size_tags",
        "metadata.supports_duplex",
    ):
        assert path in text
    assert text.count("metadata.connection_tags") == 2
    assert "scan" not in text
    assert "copy" not in text


def test_filter_matches_only_qualified_local_point() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="printer-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    base = {
        "category_scope": "printer",
        "brand": "Hp",
        "print_technology": "laser",
        "color_mode": "color",
        "monthly_volume_max_pages": 2000,
        "print_speed_ppm": 25.0,
        "connection_tags": ["wifi", "lan", "usb"],
        "paper_size_tags": ["a4", "a5"],
        "supports_duplex": True,
        "width_mm": 420,
        "height_mm": 250,
        "depth_mm": 360,
    }
    client.upsert(
        collection_name="printer-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "original_price_vnd": 8_000_000,
                        "promotional_price_vnd": 7_000_000,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={"metadata": {**base, "connection_tags": ["wifi"], "original_price_vnd": 7_000_000}},
            ),
            models.PointStruct(id=3, vector=[1.0, 0.0], payload={"metadata": base}),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="printer-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "budget_max_vnd": 10_000_000,
                "monthly_pages_estimate": 1000,
                "hard_constraints": {
                    "technologies": ["laser"],
                    "color_modes": ["color"],
                    "required_connections": ["wifi", "lan"],
                    "requires_duplex": True,
                },
            }
        ),
        limit=10,
    )
    assert [point.id for point in result.points] == [1]


def test_normalizer_patch_indexes_and_prompts() -> None:
    candidate = normalize_candidate(printer_point())
    assert candidate["effective_price_vnd"] == 7_000_000
    assert candidate["print_technology"] == "laser"
    assert candidate["monthly_volume_pages"]["max"] == 2000
    assert candidate["connection_tags"] == ["wifi", "lan", "usb", "mobile_print"]
    assert "metadata" not in candidate
    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {"usage_preferences": ["mono_documents"], "hard_constraints": {"required_connections": []}},
        ProfilePatch(
            add={"hard_constraints.required_connections": ["wifi"]},
            replace={"usage_preferences": ["color_documents"]},
            set={"household_size": 4},
        ),
        spec,
    )
    assert profile["usage_preferences"] == ["color_documents"]
    assert profile["hard_constraints"]["required_connections"] == ["wifi"]
    assert "household_size" not in profile
    assert set(changed) == {"usage_preferences", "hard_constraints.required_connections"}
    fake = FakeQdrant()
    removed = "metadata.print_speed_ppm"
    del fake.payload_schema[removed]
    assert ensure_payload_indexes(fake, "mayin", load_config()["payload_indexes"]) == {removed: "float"}
    assert ensure_payload_indexes(fake, "mayin", load_config()["payload_indexes"], apply=True) == {}
    combined = " ".join(
        (
            spec.build_need_extraction_prompt("Cần máy in", {}),
            spec.build_ranking_prompt({"need_profile": {}, "hard_constraints": {}, "candidates": []}),
            spec.build_response_prompt({"need_profile": {}, "selected_products": []}),
            no_match_answer({}),
        )
    ).casefold()
    assert "máy in" in combined
    assert "tủ lạnh" not in combined
    assert "công nghệ làm lạnh" not in combined


def test_graph_interrupts_and_resumes_to_printer_collection() -> None:
    qdrant = FakeQdrant()
    graph = build_graph(llm=FakeLLM(), qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "printer-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn máy in")]}, config
    )
    assert [
        item["question_id"] for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["print_purpose", "monthly_volume", "budget"]
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "print_purpose", "option_id": "color_documents"},
                    {"question_id": "monthly_volume", "option_id": "office"},
                    {"question_id": "budget", "option_id": "5m_10m"},
                ]
            }
        ),
        config,
    )
    assert completed["conversation"]["active_category"] == "printer"
    assert completed["need_profile"]["budget_max_vnd"] == 10_000_000
    assert completed["need_profile"]["hard_constraints"]["color_modes"] == ["color"]
    assert completed["ranking"]["selected_products"][0]["product_id"] == "pr-1"
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "mayin"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"
