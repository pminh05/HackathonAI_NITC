"""Behavior and data-contract tests for the smartwatch category."""

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
from advisor.categories.smartwatch import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.smartwatch.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.smartwatch.normalizer import normalize_candidate
from advisor.categories.smartwatch.schemas import SmartwatchNeedProfile
from advisor.categories.smartwatch.setup_indexes import ensure_payload_indexes
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


ROOT = Path(__file__).parents[2]
INGESTION = runpy.run_path(ROOT / "ingestion/dong_ho_thong_minh/changeName.py")
QDRANT = runpy.run_path(ROOT / "ingestion/dong_ho_thong_minh/qdrant.py")
normalize_metadata = INGESTION["normalize_metadata"]
normalize_typical_battery_hours = INGESTION["normalize_typical_battery_hours"]
normalize_water_resistance_atm = INGESTION["normalize_water_resistance_atm"]


def smartwatch_point() -> Any:
    return SimpleNamespace(
        id="point-watch-1",
        score=0.95,
        payload={
            "product_id": "watch-1",
            "name": "Đồng hồ thông minh Garmin Test",
            "text": "Đồng hồ GPS, pin 14 ngày và theo dõi sức khỏe.",
            "image_path": "/public/dong_ho_thong_minh.jpg",
            "metadata": {
                "category_scope": "smartwatch",
                "brand": "Garmin",
                "brand_key": "garmin",
                "model_code": "WATCH-X",
                "original_price_vnd": 12_000_000,
                "promotional_price_vnd": 9_500_000,
                "compatible_platforms": ["ios", "android"],
                "Tương thích": "iOS 14 trở lên | Android 9 trở lên",
                "display_family": "amoled_oled",
                "Màn hình hiển thị": "AMOLED",
                "screen_size_inch": 1.3,
                "Độ phân giải": "454 x 454 pixels",
                "case_length_mm": 45.0,
                "case_width_mm": 45.0,
                "case_thickness_mm": 11.0,
                "weight_g": 48.0,
                "wrist_min_cm": 13.0,
                "wrist_max_cm": 21.0,
                "strap_material_family": "silicone",
                "Chất liệu dây": "Silicone",
                "typical_battery_hours": 336.0,
                "battery_mah": 450,
                "Thời gian sử dụng": "Khoảng 14 ngày",
                "charging_time_hours": 2.0,
                "water_resistance_atm": 5.0,
                "Chuẩn chống nước, bụi": "Kháng nước 5 ATM (Bơi vùng nước nông)",
                "call_mode": "on_wrist",
                "Thực hiện cuộc gọi": "Nghe gọi ngay trên đồng hồ",
                "has_cellular": False,
                "SIM": "Không có",
                "has_gps": True,
                "Định vị": "Galileo | GLONASS | GPS",
                "has_notifications": True,
                "Hiển thị thông báo": "Tin nhắn | Cuộc gọi",
                "swim_ready": True,
                "has_sos": False,
                "Theo dõi sức khoẻ": "Đo nhịp tim | SpO2 | Theo dõi giấc ngủ",
                "Môn thể thao": "Chạy bộ | Bơi",
                "health_feature_tags": [
                    "heart_rate",
                    "spo2",
                    "sleep",
                ],
                "Hệ điều hành": "Garmin OS",
                "Tiện ích khác": "Báo thức | Đồng hồ bấm giờ",
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
                category=IntentLabel.SMARTWATCH,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            if self.owner.patches:
                return self.owner.patches.pop(0)
            return ProfilePatch()
        if self.schema is RankingResult:
            return RankingResult(
                selected_products=[
                    {
                        "product_id": "watch-1",
                        "reason": "Có GPS, pin dài và đúng hệ điện thoại.",
                        "trade_off": "Không có SIM độc lập.",
                    }
                ]
            )
        raise AssertionError(self.schema)


class FakeLLM:
    def __init__(
        self,
        *,
        analyses: list[TurnAnalysisResult] | None = None,
        patches: list[ProfilePatch] | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])

    def with_structured_output(
        self, schema: type[Any], **_: Any
    ) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, _: str) -> AIMessage:
        return AIMessage(
            content="Mình đề xuất đồng hồ thông minh phù hợp nhu cầu của bạn."
        )


class FakeQdrant:
    def __init__(self) -> None:
        config = load_config()
        self.payload_schema = {
            field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
            for field, schema in config["payload_indexes"].items()
        }
        self.created: list[tuple[str, str]] = []
        self.query_kwargs: dict[str, Any] | None = None
        self.query_count = 0

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
        self.query_count += 1
        return SimpleNamespace(points=[smartwatch_point()])


def test_ingestion_normalizes_typed_smartwatch_metadata() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Garmin",
            "giá gốc": "12.000.000 đ",
            "giá khuyến mãi": "9.500.000 đ",
            "gia_goc_vnd": 120_000_000,
            "gia_khuyen_mai_vnd": 95_000_000,
            "Tương thích": "iOS 14 trở lên | Android 9 trở lên",
            "Màn hình hiển thị": "AMOLED",
            "Chất liệu dây": "Silicone",
            "kich_thuoc_man_hinh_inch": 1.3,
            "dai_mm": 45,
            "ngang_mm": 44,
            "day_mm": 11,
            "khoi_luong_g": 48,
            "chu_vi_co_tay_min_cm": 13,
            "chu_vi_co_tay_max_cm": 21,
            "dung_luong_pin_mah": 450,
            "thoi_gian_sac_gio": 2,
            "Thời gian sử dụng": (
                "Khoảng 14 ngày (ở chế độ đồng hồ thông minh) | "
                "Khoảng 30 giờ khi sử dụng GPS"
            ),
            "Chuẩn chống nước, bụi": "Kháng nước 5 ATM (Bơi vùng nước nông)",
            "Định vị": "Galileo | GLONASS | GPS",
            "Thực hiện cuộc gọi": "Nghe gọi độc lập",
            "SIM": "eSIM",
            "Hiển thị thông báo": "Tin nhắn | Cuộc gọi",
            "Theo dõi sức khoẻ": (
                "Đo nhịp tim | SpO2 | Theo dõi giấc ngủ | Đo huyết áp"
            ),
            "Tiện ích khác": "Cuộc gọi khẩn cấp SOS",
        }
    )
    assert metadata["category_scope"] == "smartwatch"
    assert metadata["brand_key"] == "garmin"
    assert metadata["original_price_vnd"] == 12_000_000
    assert metadata["promotional_price_vnd"] == 9_500_000
    assert metadata["compatible_platforms"] == ["ios", "android"]
    assert metadata["display_family"] == "amoled_oled"
    assert metadata["strap_material_family"] == "silicone"
    assert metadata["case_width_mm"] == 44.0
    assert metadata["typical_battery_hours"] == 336.0
    assert metadata["water_resistance_atm"] == 5.0
    assert metadata["call_mode"] == "standalone"
    assert metadata["has_cellular"] is True
    assert metadata["has_gps"] is True
    assert metadata["has_notifications"] is True
    assert metadata["swim_ready"] is True
    assert metadata["has_sos"] is True
    assert {
        "heart_rate",
        "spo2",
        "sleep",
        "blood_pressure",
    } <= set(metadata["health_feature_tags"])
    assert "gia_goc_vnd" not in metadata


def test_ingestion_does_not_infer_incomparable_battery_or_atm() -> None:
    assert (
        normalize_typical_battery_hours(
            "Khoảng 30 giờ khi sử dụng GPS | 40 ngày ở chế độ tiết kiệm pin"
        )
        is None
    )
    assert normalize_water_resistance_atm("Kháng nước IP68") is None
    unknown = normalize_metadata({"brand": "Unknown"})
    assert unknown["compatible_platforms"] == []
    assert unknown["health_feature_tags"] == []
    for capability in (
        "call_mode",
        "has_cellular",
        "has_gps",
        "has_notifications",
        "swim_ready",
    ):
        assert capability not in unknown


def test_ingestion_point_shape_uses_cloud_inference_and_rejects_stale_payload() -> None:
    product = {
        "id": "watch-sku",
        "name": "Đồng hồ thông minh Test",
        "text": "Đồng hồ GPS và theo dõi sức khỏe.",
        "image_path": "/public/dong_ho_thong_minh.jpg",
        "metadata": {
            "category_scope": "smartwatch",
            "brand_key": "garmin",
            "compatible_platforms": ["ios", "android"],
            "health_feature_tags": [],
            "has_gps": True,
        },
    }
    first = next(QDRANT["point_generator"]([product]))
    second = next(QDRANT["point_generator"]([product]))
    assert first.id == second.id
    assert isinstance(first.vector, models.Document)
    assert first.vector.model == "intfloat/multilingual-e5-small"
    assert first.vector.text.startswith("passage: ")
    stale = {**product, "id": "stale", "metadata": {"brand": "Garmin"}}
    with pytest.raises(ValueError, match="metadata canonical"):
        next(QDRANT["point_generator"]([stale]))


def test_spec_profile_registry_and_missing_field_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "smartwatch"
    assert spec.display_name == "Đồng hồ thông minh"
    assert config["collection"] == "donghothongminh"
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert build_default_registry().get_spec("smartwatch").name == "smartwatch"
    profile = SmartwatchNeedProfile(
        budget_max_vnd=10_000_000,
        phone_platform="ios",
        usage_preferences=["outdoor_navigation"],
        hard_constraints={"requires_gps": True},
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == [
        "primary_usage",
        "budget",
        "phone_platform",
    ]
    with pytest.raises(ValidationError):
        SmartwatchNeedProfile(usage_preferences=["weekly_storage"])


def test_filter_enforces_hard_fields_and_not_soft_preferences() -> None:
    query_filter = build_filter(
        {
            "budget_max_vnd": 10_000_000,
            "phone_platform": "ios",
            "soft_preferences": ["premium_design", "long_battery"],
            "hard_constraints": {
                "brands": ["Garmin"],
                "display_families": ["amoled_oled"],
                "strap_material_families": ["silicone"],
                "min_screen_size_inch": 1.2,
                "max_screen_size_inch": 1.5,
                "max_case_width_mm": 46,
                "max_weight_g": 60,
                "wrist_circumference_cm": 16,
                "min_typical_battery_hours": 168,
                "min_water_resistance_atm": 5,
                "call_requirement": "on_wrist",
                "requires_cellular": True,
                "requires_gps": True,
                "requires_notifications": True,
                "requires_swimming": True,
                "required_health_features": ["spo2"],
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.category_scope",
        "metadata.brand_key",
        "metadata.compatible_platforms",
        "metadata.display_family",
        "metadata.strap_material_family",
        "metadata.screen_size_inch",
        "metadata.case_width_mm",
        "metadata.weight_g",
        "metadata.wrist_min_cm",
        "metadata.wrist_max_cm",
        "metadata.typical_battery_hours",
        "metadata.water_resistance_atm",
        "metadata.call_mode",
        "metadata.has_cellular",
        "metadata.has_gps",
        "metadata.has_notifications",
        "metadata.swim_ready",
        "metadata.health_feature_tags",
    ):
        assert path in text
    assert "premium_design" not in text
    assert "long_battery" not in text


def test_filter_matches_only_fully_qualified_local_points_and_call_modes() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="smartwatch-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    qualified = {
        "category_scope": "smartwatch",
        "brand_key": "garmin",
        "compatible_platforms": ["ios", "android"],
        "display_family": "amoled_oled",
        "strap_material_family": "silicone",
        "screen_size_inch": 1.3,
        "case_width_mm": 45.0,
        "weight_g": 48.0,
        "wrist_min_cm": 13.0,
        "wrist_max_cm": 21.0,
        "typical_battery_hours": 336.0,
        "water_resistance_atm": 5.0,
        "call_mode": "on_wrist",
        "has_cellular": False,
        "has_gps": True,
        "has_notifications": True,
        "swim_ready": True,
        "health_feature_tags": ["heart_rate", "spo2", "sleep"],
        "original_price_vnd": 12_000_000,
        "promotional_price_vnd": 9_500_000,
    }
    client.upsert(
        collection_name="smartwatch-test",
        points=[
            models.PointStruct(
                id=1, vector=[1.0, 0.0], payload={"metadata": qualified}
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **qualified,
                        "compatible_platforms": ["android"],
                    }
                },
            ),
            models.PointStruct(
                id=3,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **qualified,
                        "health_feature_tags": ["heart_rate"],
                    }
                },
            ),
            models.PointStruct(
                id=4,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        key: value
                        for key, value in qualified.items()
                        if key not in {"original_price_vnd", "promotional_price_vnd"}
                    }
                },
            ),
            models.PointStruct(
                id=5,
                vector=[1.0, 0.0],
                payload={"metadata": {**qualified, "call_mode": "standalone"}},
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="smartwatch-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "budget_max_vnd": 10_000_000,
                "phone_platform": "ios",
                "hard_constraints": {
                    "brands": ["Garmin"],
                    "wrist_circumference_cm": 16,
                    "min_typical_battery_hours": 168,
                    "min_water_resistance_atm": 5,
                    "call_requirement": "on_wrist",
                    "requires_gps": True,
                    "requires_notifications": True,
                    "requires_swimming": True,
                    "required_health_features": ["spo2"],
                },
            }
        ),
        limit=10,
    )
    assert sorted(point.id for point in result.points) == [1, 5]

    standalone = client.query_points(
        collection_name="smartwatch-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {"hard_constraints": {"call_requirement": "standalone"}}
        ),
        limit=10,
    )
    assert [point.id for point in standalone.points] == [5]


def test_normalizer_patch_indexes_prompts_and_no_match() -> None:
    candidate = normalize_candidate(smartwatch_point())
    assert candidate["effective_price_vnd"] == 9_500_000
    assert candidate["typical_battery_hours"] == 336.0
    assert candidate["compatible_platforms"] == ["ios", "android"]
    assert candidate["has_gps"] is True
    assert candidate["swim_ready"] is True
    assert candidate["health_feature_tags"][-1] == "sleep"
    assert "metadata" not in candidate

    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "usage_preferences": ["health_monitoring"],
            "hard_constraints": {"brands": []},
        },
        ProfilePatch(
            add={"hard_constraints.brands": ["Garmin"]},
            replace={"usage_preferences": ["outdoor_navigation"]},
            set={"household_size": 4},
        ),
        spec,
    )
    assert profile["usage_preferences"] == ["outdoor_navigation"]
    assert profile["hard_constraints"]["brands"] == ["Garmin"]
    assert "household_size" not in profile
    assert set(changed) == {"usage_preferences", "hard_constraints.brands"}

    fake = FakeQdrant()
    removed = "metadata.weight_g"
    del fake.payload_schema[removed]
    assert ensure_payload_indexes(
        fake, "donghothongminh", load_config()["payload_indexes"]
    ) == {removed: "float"}
    assert ensure_payload_indexes(
        fake,
        "donghothongminh",
        load_config()["payload_indexes"],
        apply=True,
    ) == {}

    combined = " ".join(
        (
            spec.build_need_extraction_prompt("Cần smartwatch", {}),
            spec.build_ranking_prompt(
                {"need_profile": {}, "hard_constraints": {}, "candidates": []}
            ),
            spec.build_response_prompt(
                {"need_profile": {}, "selected_products": []}
            ),
            no_match_answer({}),
        )
    ).casefold()
    assert "đồng hồ thông minh" in combined
    assert "tủ lạnh" not in combined


def test_graph_interrupts_and_resumes_to_smartwatch_collection() -> None:
    qdrant = FakeQdrant()
    graph = build_graph(llm=FakeLLM(), qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "smartwatch-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn đồng hồ thông minh")]}, config
    )
    assert [
        item["question_id"]
        for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["primary_usage", "budget", "phone_platform"]
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "primary_usage",
                        "option_id": "outdoor_navigation",
                    },
                    {"question_id": "budget", "option_id": "5m_10m"},
                    {"question_id": "phone_platform", "option_id": "ios"},
                ]
            }
        ),
        config,
    )
    assert completed["conversation"]["active_category"] == "smartwatch"
    assert completed["need_profile"]["budget_max_vnd"] == 10_000_000
    assert completed["need_profile"]["phone_platform"] == "ios"
    assert completed["need_profile"]["hard_constraints"]["requires_gps"] is True
    assert completed["ranking"]["selected_products"][0]["product_id"] == "watch-1"
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "donghothongminh"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def _complete_smartwatch_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"budget_max_vnd": 10_000_000, "phone_platform": "ios"},
        replace={"usage_preferences": ["outdoor_navigation"]},
    )


def test_follow_up_reuses_smartwatch_recommendations_without_qdrant() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.SMARTWATCH,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.OTHER,
                category_transition="inherit",
                action="compare",
                scope="current_recommendations",
                referenced_product_ids=["watch-1"],
            ),
        ],
        patches=[_complete_smartwatch_patch()],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "smartwatch-follow-up"}}
    graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn đồng hồ chạy bộ")]}, config
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="Mẫu này dùng với iPhone thế nào?")]},
        config,
    )
    assert result["conversation"]["active_category"] == "smartwatch"
    assert result["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 1


def test_returning_to_smartwatch_restores_category_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.SMARTWATCH,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            ),
            TurnAnalysisResult(
                category=IntentLabel.PHONE_RECORDING_MICROPHONE,
                category_transition="switch",
                switch_evidence="micro thu âm điện thoại",
                action="switch_category",
            ),
            TurnAnalysisResult(
                category=IntentLabel.SMARTWATCH,
                category_transition="switch",
                switch_evidence="đồng hồ thông minh",
                action="switch_category",
            ),
        ],
        patches=[_complete_smartwatch_patch()],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "smartwatch-category-return"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn đồng hồ thông minh")]}, config
    )
    first_profile = first["need_profile"]
    graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua micro thu âm điện thoại")]},
        config,
    )
    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại đồng hồ thông minh")]}, config
    )
    assert restored["conversation"]["active_category"] == "smartwatch"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 1
