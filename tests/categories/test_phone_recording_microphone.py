"""Behavior and data-contract tests for the recording-microphone category."""

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

from advisor.categories.phone_recording_microphone import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.phone_recording_microphone.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.phone_recording_microphone.normalizer import normalize_candidate
from advisor.categories.phone_recording_microphone.schemas import (
    PhoneRecordingMicrophoneCustomAnswer,
    PhoneRecordingMicrophoneNeedProfile,
)
from advisor.categories.phone_recording_microphone.setup_indexes import (
    ensure_payload_indexes,
)
from advisor.categories.registry import build_default_registry
from advisor.categories.washing_machine import load_config as load_washing_config
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


ROOT = Path(__file__).parents[2]
INGESTION_DIR = ROOT / "ingestion/micro_thu_am_dien_thoai"
INGESTION_MODULE = runpy.run_path(INGESTION_DIR / "changeName.py")
normalize_metadata = INGESTION_MODULE["normalize_metadata"]
QDRANT_MODULE = runpy.run_path(INGESTION_DIR / "qdrant.py")


def microphone_point() -> Any:
    return SimpleNamespace(
        id="point-mic-1",
        score=0.93,
        payload={
            "product_id": "mic-1",
            "name": "Micro thu âm Boya Test",
            "text": "Micro không dây dùng USB-C, gồm hai bộ phát.",
            "image_path": "/public/micro_thu_am_dien_thoai.jpg",
            "metadata": {
                "category_scope": "phone_recording_microphone",
                "brand": "Boya",
                "brand_key": "boya",
                "model_code": "BY-Test",
                "product_type": "wireless_recording",
                "original_price_vnd": 1_560_000,
                "promotional_price_vnd": 1_490_000,
                "compatibility_tags": ["ios", "android"],
                "connector_tags": ["usb_c"],
                "feature_tags": ["noise_reduction", "auto_connect"],
                "pickup_pattern": "omnidirectional",
                "wireless_band": "2_4_ghz",
                "transmitter_count": 2,
                "receiver_count": 1,
                "runtime_min_hours": 6.0,
                "runtime_max_hours": 6.0,
                "transmission_range_m": 100.0,
                "manufacture_year": 2024,
                "origin": "Trung Quốc",
            },
        },
    )


def washing_point() -> Any:
    return SimpleNamespace(
        id="point-wm-1",
        score=0.9,
        payload={
            "product_id": "wm-1",
            "name": "Máy giặt Test",
            "text": "Máy giặt cửa trước 10 kg.",
            "image_path": "/public/may_giat.jpg",
            "metadata": {
                "brand": "Test",
                "product_type": "front_load",
                "original_price_vnd": 12_000_000,
                "promotional_price_vnd": 11_000_000,
                "wash_capacity_kg": 10.0,
                "people_min": 3,
                "people_max": 5,
                "has_inverter": True,
                "has_dryer": False,
            },
        },
    )


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
                category=IntentLabel.PHONE_RECORDING_MICROPHONE,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            product_id = "wm-1" if "wm-1" in prompt else "mic-1"
            return RankingResult(
                selected_products=[
                    {
                        "product_id": product_id,
                        "reason": "Phù hợp nhu cầu đã xác nhận.",
                        "trade_off": "Một số thông số chưa được công bố.",
                    }
                ]
            )
        if self.schema is PhoneRecordingMicrophoneCustomAnswer:
            return self.owner.custom or PhoneRecordingMicrophoneCustomAnswer(
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
        custom: PhoneRecordingMicrophoneCustomAnswer | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.custom = custom
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(("PlainResponse", prompt))
        return AIMessage(content="Mình đề xuất micro thu âm phù hợp nhu cầu của bạn.")


class FakeQdrant:
    def __init__(self) -> None:
        self.payload_schemas: dict[str, dict[str, Any]] = {}
        for config in (load_config(), load_washing_config()):
            self.payload_schemas[config["collection"]] = {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in config["payload_indexes"].items()
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
        self.query_kwargs = kwargs
        self.query_count += 1
        point = washing_point() if kwargs["collection_name"] == "maygiat" else microphone_point()
        return SimpleNamespace(points=[point])


def complete_microphone_patch() -> ProfilePatch:
    return ProfilePatch(
        set={"recording_setup": "android_usb_c", "budget_max_vnd": 3_000_000},
        replace={"usage_preferences": ["two_person_interview"]},
    )


def test_ingestion_normalizes_category_specific_metadata_and_ignores_bad_values() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Boya",
            "model_code": "BY-Test",
            "Loại sản phẩm": "Micro thu âm không dây",
            "Tương thích": "iOS (iPhone) | Android",
            "Cổng tai nghe, headphone": "Lightning, Type C, 3.5mm",
            "Tính năng cơ bản": "Tự động kết nối | Lọc tiếng ồn | Ngắt micro tạm thời",
            "Hướng thu âm": "Đa hướng",
            "Băng tần": "2.4 Ghz",
            "Thời gian sử dụng": "9-10 tiếng",
            "Khoảng cách truyền": "Tối đa 100 m",
            "Phụ kiện đi kèm": "2 mic phát | 1 mic thu | Cáp sạc",
            "giá gốc": "1560000.0",
            "giá khuyến mãi": "1,490,000",
            "thoi_gian_su_dung_min_gio": -10,
            "sac_day_bo_thu_phut": 454,
        }
    )
    assert metadata["category_scope"] == "phone_recording_microphone"
    assert metadata["brand_key"] == "boya"
    assert metadata["product_type"] == "wireless_recording"
    assert metadata["compatibility_tags"] == ["ios", "android"]
    assert metadata["connector_tags"] == ["lightning", "usb_c", "3_5mm"]
    assert metadata["feature_tags"] == ["noise_reduction", "auto_connect", "mute"]
    assert metadata["pickup_pattern"] == "omnidirectional"
    assert metadata["transmitter_count"] == 2
    assert metadata["runtime_min_hours"] == 9.0
    assert metadata["runtime_max_hours"] == 10.0
    assert metadata["original_price_vnd"] == 1_560_000
    assert metadata["promotional_price_vnd"] == 1_490_000
    assert "thoi_gian_su_dung_min_gio" not in metadata
    assert "sac_day_bo_thu_phut" not in metadata
    assert "ignored_invalid_generated_runtime" in metadata["data_quality_flags"]


def test_source_catalog_normalizes_all_points_with_cloud_inference() -> None:
    source = json.loads(
        (INGESTION_DIR / "data/micro_thu_am_dien_thoai.json").read_text(
            encoding="utf-8-sig"
        )
    )["data_clean"]
    normalized = [normalize_metadata(item) for item in source]
    assert len(normalized) == 33
    assert all(item["category_scope"] == "phone_recording_microphone" for item in normalized)
    products = [
        {
            "id": item["sku"],
            "name": f"Micro thu âm {item['brand']}",
            "text": "Mô tả micro thu âm.",
            "image_path": "/public/micro_thu_am_dien_thoai.jpg",
            "metadata": item,
        }
        for item in normalized
    ]
    points = list(QDRANT_MODULE["point_generator"](products))
    assert len(points) == len({point.id for point in points}) == 33
    assert all(isinstance(point.vector, models.Document) for point in points)
    assert all(point.vector.model == "intfloat/multilingual-e5-small" for point in points)


def test_category_spec_config_schema_and_questions_are_microphone_specific() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "phone_recording_microphone"
    assert spec.display_name == "Micro thu âm"
    assert config["collection"] == "microthuamdienthoai"
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert build_default_registry().get_spec("phone_recording_microphone").name == spec.name
    assert get_missing_profile_fields({}) == [
        "recording_setup",
        "budget",
        "usage_preferences",
    ]
    profile = PhoneRecordingMicrophoneNeedProfile(
        recording_setup="android_usb_c",
        budget_max_vnd=3_000_000,
        usage_preferences=["outdoor_mobile"],
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    with pytest.raises(ValidationError):
        PhoneRecordingMicrophoneNeedProfile(usage_preferences=["daily_laundry"])


def test_filter_contains_only_indexed_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "recording_setup": "android_usb_c",
            "budget_max_vnd": 3_000_000,
            "soft_preferences": ["portability", "vocal_focus"],
            "hard_constraints": {
                "brands": ["Boya"],
                "product_types": ["wireless_recording"],
                "required_compatibility_tags": ["ios", "android"],
                "connector_types": ["usb_c"],
                "min_transmitter_count": 2,
                "min_runtime_hours": 6,
                "min_transmission_range_m": 50,
                "pickup_patterns": ["omnidirectional"],
                "wireless_bands": ["2_4_ghz"],
                "required_features": ["noise_reduction"],
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in STANDARDIZED_METADATA_PATHS:
        assert path in text
    assert "portability" not in text and "vocal_focus" not in text
    assert build_filter({"soft_preferences": ["portability"]}) is None


def test_filter_matches_only_verified_compatible_payload_locally() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="micro-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    matching = microphone_point().payload
    client.upsert(
        collection_name="micro-test",
        points=[
            models.PointStruct(id=1, vector=[1.0, 0.0], payload=matching),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    **matching,
                    "metadata": {
                        **matching["metadata"],
                        "connector_tags": ["lightning"],
                        "promotional_price_vnd": None,
                        "original_price_vnd": None,
                    },
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="micro-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "recording_setup": "android_usb_c",
                "budget_max_vnd": 2_000_000,
                "hard_constraints": {
                    "min_transmitter_count": 2,
                    "required_features": ["noise_reduction"],
                },
            }
        ),
        limit=5,
    )
    assert [point.id for point in result.points] == [1]


def test_normalizer_patch_indexes_prompts_and_no_match_are_category_specific() -> None:
    candidate = normalize_candidate(microphone_point())
    assert candidate["effective_price_vnd"] == 1_490_000
    assert candidate["compatibility_tags"] == ["ios", "android"]
    assert candidate["transmitter_count"] == 2
    assert candidate["runtime_hours"] == {"min": 6.0, "max": 6.0}
    assert "metadata" not in candidate

    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "recording_setup": "android_usb_c",
            "usage_preferences": ["solo_content"],
            "hard_constraints": {"brands": ["boya"]},
        },
        ProfilePatch(
            replace={"hard_constraints.brands": ["dji"]},
            remove={"usage_preferences": ["solo_content"]},
            add={"usage_preferences": ["outdoor_mobile"]},
            set={"household_size": 4},
        ),
        spec,
    )
    assert profile["hard_constraints"]["brands"] == ["dji"]
    assert profile["usage_preferences"] == ["outdoor_mobile"]
    assert "household_size" not in profile
    assert set(changed) == {"hard_constraints.brands", "usage_preferences"}

    client = FakeQdrant()
    removed = "metadata.runtime_min_hours"
    del client.payload_schemas["microthuamdienthoai"][removed]
    assert ensure_payload_indexes(
        client, "microthuamdienthoai", load_config()["payload_indexes"], apply=False
    ) == {removed: "float"}
    assert ensure_payload_indexes(
        client, "microthuamdienthoai", load_config()["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "float")]

    prompts = " ".join(
        (
            spec.build_need_extraction_prompt("Cần micro thu âm", {}),
            spec.build_ranking_prompt(
                {"need_profile": {}, "hard_constraints": {}, "candidates": []}
            ),
            spec.build_response_prompt({"need_profile": {}, "selected_products": []}),
            no_match_answer({"recording_setup": "android_usb_c"}),
        )
    ).casefold()
    assert "micro thu âm" in prompts and "usb-c" in prompts
    assert "máy giặt" not in prompts and "tải giặt" not in prompts


def test_graph_interrupts_and_resumes_to_microphone_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "microphone-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi micro thu âm")]}, config
    )
    payload = interrupted["__interrupt__"][0].value
    assert payload["category"] == "phone_recording_microphone"
    assert [item["question_id"] for item in payload["questions"]] == [
        "recording_setup",
        "budget",
        "usage_preferences",
    ]
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "recording_setup", "option_id": "android_usb_c"},
                    {"question_id": "budget", "option_id": "1m_3m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "two_person_interview",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["recording_setup"] == "android_usb_c"
    assert completed["need_profile"]["budget_max_vnd"] == 3_000_000
    assert completed["need_profile"]["hard_constraints"]["min_transmitter_count"] == 2
    assert completed["ranking"]["selected_products"][0]["product_id"] == "mic-1"
    assert qdrant.query_kwargs["collection_name"] == "microthuamdienthoai"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_setup_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=PhoneRecordingMicrophoneCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Android cổng USB-C",
            recording_setup="android_usb_c",
            confidence=0.99,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "microphone-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua micro thu âm")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "recording_setup",
                        "option_id": "other",
                        "custom_answer": "Android cổng USB-C",
                    },
                    {"question_id": "budget", "option_id": "under_1m"},
                    {"question_id": "usage_preferences", "option_id": "solo_content"},
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["recording_setup"] == "android_usb_c"
    assert completed["need_profile"]["custom_answers"]["recording_setup"]["raw_answer"] == "Android cổng USB-C"


def test_switch_microphone_washing_machine_and_back_restores_context() -> None:
    llm = FakeLLM(
        analyses=[
            TurnAnalysisResult(
                category=IntentLabel.PHONE_RECORDING_MICROPHONE,
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
                category=IntentLabel.PHONE_RECORDING_MICROPHONE,
                category_transition="switch",
                switch_evidence="micro thu âm",
                action="switch_category",
            ),
        ],
        patches=[
            complete_microphone_patch(),
            ProfilePatch(
                set={"household_size": 5, "budget_max_vnd": 12_000_000},
                replace={"usage_preferences": ["energy_saving"]},
            ),
        ],
    )
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "micro-washing-micro"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn micro thu âm")]}, config
    )
    first_profile = first["need_profile"]
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua máy giặt")]}, config
    )
    assert switched["conversation"]["active_category"] == "washing_machine"
    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại micro thu âm")]}, config
    )
    assert restored["conversation"]["active_category"] == "phone_recording_microphone"
    assert restored["need_profile"] == first_profile
    assert restored["conversation"]["execution_mode"] == "reuse"
    assert qdrant.query_count == 2
