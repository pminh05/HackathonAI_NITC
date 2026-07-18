"""Behavior and data-contract tests for the washing-machine category."""

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

from advisor.categories.washing_machine import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.washing_machine.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.washing_machine.normalizer import normalize_candidate
from advisor.categories.washing_machine.schemas import (
    WashingMachineCustomAnswer,
    WashingMachineNeedProfile,
)
from advisor.categories.washing_machine.setup_indexes import ensure_payload_indexes
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import (
    IntentLabel,
    ProfilePatch,
    RankingResult,
    TurnAnalysisResult,
)


INGESTION_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_giat/changeName.py"
)
normalize_metadata = INGESTION_MODULE["normalize_metadata"]
QDRANT_MODULE = runpy.run_path(
    Path(__file__).parents[2] / "ingestion/may_giat/qdrant.py"
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
                category=IntentLabel.WASHING_MACHINE,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return self.owner.patches.pop(0) if self.owner.patches else ProfilePatch()
        if self.schema is RankingResult:
            return (
                self.owner.rankings.pop(0)
                if self.owner.rankings
                else RankingResult(
                    selected_products=[
                        {
                            "product_id": "wm-1",
                            "reason": "Phù hợp gia đình 5 người.",
                            "trade_off": "Không có dữ liệu điều khiển thông minh.",
                        }
                    ]
                )
            )
        if self.schema is WashingMachineCustomAnswer:
            return self.owner.custom or WashingMachineCustomAnswer(
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
        custom: WashingMachineCustomAnswer | None = None,
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
            content="Mình đề xuất mẫu máy giặt thử nghiệm theo nhu cầu của bạn."
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
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-wm-1",
                    score=0.92,
                    payload={
                        "product_id": "wm-1",
                        "name": "Máy giặt Test 10 kg",
                        "text": "Máy giặt cửa trước 10 kg có Inverter.",
                        "image_path": "/public/may_giat.jpg",
                        "metadata": {
                            "brand": "Test",
                            "product_type": "front_load",
                            "Loại sản phẩm": "Cửa trước",
                            "drum_type": "horizontal",
                            "Lồng giặt": "Lồng ngang",
                            "original_price_vnd": 13_000_000,
                            "promotional_price_vnd": 11_500_000,
                            "wash_capacity_kg": 10.0,
                            "people_min": 3,
                            "people_max": 5,
                            "Số người sử dụng": "Từ 3 - 5 người",
                            "has_inverter": True,
                            "has_dryer": False,
                            "spin_speed_rpm": 1200,
                            "program_count": 12,
                            "width_cm": 60.0,
                            "height_cm": 85.0,
                            "depth_cm": 62.0,
                        },
                    },
                )
            ]
        )


def test_ingestion_normalizes_faulty_prices_units_people_and_capabilities() -> None:
    metadata = normalize_metadata(
        {
            "brand": "LG",
            "Loại sản phẩm": "Máy giặt sấy",
            "Lồng giặt": "Lồng ngang",
            "Loại Inverter": "Công nghệ Inverter Direct Drive",
            "Số người sử dụng": "Trên 7 người (Trên 10 kg)",
            "Khối lượng tải chính": "10.5 Kg giặt / 7 Kg sấy",
            "khoi_luong_giat_kg": 10.5,
            "khoi_luong_say_kg": 7,
            "Ngang": "59",
            "Cao": "850 mm",
            "Sâu": "0.63 m",
            "giá gốc": "18640000.0",
            "giá khuyến mãi": "13190000.0",
            "gia_goc_vnd": 186_400_000,
            "gia_khuyen_mai_vnd": 131_900_000,
        }
    )

    assert metadata["original_price_vnd"] == 18_640_000
    assert metadata["promotional_price_vnd"] == 13_190_000
    assert metadata["product_type"] == "washer_dryer"
    assert metadata["drum_type"] == "horizontal"
    assert metadata["people_min"] == 8
    assert "people_max" not in metadata
    assert metadata["has_inverter"] is True
    assert metadata["has_dryer"] is True
    assert metadata["width_cm"] == 59.0
    assert metadata["height_cm"] == 85.0
    assert metadata["depth_cm"] == 63.0


def test_ingestion_point_shape_uses_cloud_inference_and_stable_payload() -> None:
    product = {
        "id": "sku-wm",
        "name": "Máy giặt Test sku-wm",
        "text": "Máy giặt cửa trước 10 kg.",
        "image_path": "/public/may_giat.jpg",
        "metadata": {"category_scope": "washing_machine"},
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
    assert spec.name == config["category"] == "washing_machine"
    assert spec.display_name == "Máy giặt"
    assert config["collection"] == "maygiat"
    assert config["embedding_model"] == "intfloat/multilingual-e5-small"
    assert set(config["payload_fields"].values()) <= set(config["payload_indexes"])
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS


def test_profile_schema_and_missing_questions_are_washing_machine_specific() -> None:
    profile = WashingMachineNeedProfile(
        household_size=5,
        budget_max_vnd=12_000_000,
        usage_preferences=["energy_saving"],
        hard_constraints={
            "product_types": ["front_load"],
            "min_wash_capacity_kg": 9,
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
        WashingMachineNeedProfile(usage_preferences=["frozen_storage"])


def test_filter_contains_only_indexed_hard_constraints() -> None:
    query_filter = build_filter(
        {
            "household_size": 5,
            "budget_max_vnd": 12_000_000,
            "soft_preferences": ["steam", "wifi"],
            "hard_constraints": {
                "brands": ["LG"],
                "product_types": ["front_load"],
                "drum_types": ["horizontal"],
                "min_wash_capacity_kg": 9,
                "max_width_cm": 60,
                "inverter": True,
                "dryer": False,
            },
        }
    )
    serialized = query_filter.model_dump(mode="json", exclude_none=True)
    text = str(serialized)
    for path in (
        "metadata.brand",
        "metadata.product_type",
        "metadata.drum_type",
        "metadata.wash_capacity_kg",
        "metadata.has_inverter",
        "metadata.has_dryer",
    ):
        assert path in text
    assert "steam" not in text
    assert "wifi" not in text
    assert "clothing_care" in text


def test_filter_matches_normalized_payload_in_local_qdrant() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="maygiat-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="maygiat-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        "brand": "LG",
                        "product_type": "front_load",
                        "drum_type": "horizontal",
                        "original_price_vnd": 13_000_000,
                        "promotional_price_vnd": 11_500_000,
                        "wash_capacity_kg": 10.0,
                        "people_min": 3,
                        "people_max": 5,
                        "has_inverter": True,
                        "has_dryer": False,
                        "width_cm": 60.0,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        "brand": "LG",
                        "product_type": "clothing_care",
                        "original_price_vnd": 9_000_000,
                        "people_min": 3,
                        "people_max": 5,
                    }
                },
            ),
        ],
        wait=True,
    )
    result = client.query_points(
        collection_name="maygiat-test",
        query=[1.0, 0.0],
        query_filter=build_filter(
            {
                "household_size": 5,
                "budget_max_vnd": 12_000_000,
                "hard_constraints": {
                    "brands": ["LG"],
                    "product_types": ["front_load"],
                    "min_wash_capacity_kg": 9,
                    "inverter": True,
                },
            }
        ),
        limit=5,
    )
    assert [point.id for point in result.points] == [1]


def test_candidate_normalizer_whitelists_washing_machine_fields() -> None:
    candidate = normalize_candidate(FakeQdrant().query_points().points[0])
    assert candidate["product_id"] == "wm-1"
    assert candidate["effective_price_vnd"] == 11_500_000
    assert candidate["product_type"] == "front_load"
    assert candidate["wash_capacity_kg"] == 10.0
    assert candidate["has_inverter"] is True
    assert candidate["has_dryer"] is False
    assert candidate["dimensions_cm"] == {
        "width": 60.0,
        "height": 85.0,
        "depth": 62.0,
    }
    assert "metadata" not in candidate


def test_profile_patch_operations_use_washing_machine_allowlist() -> None:
    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "budget_max_vnd": 20_000_000,
            "usage_preferences": ["daily_laundry"],
            "hard_constraints": {"brands": ["Panasonic"]},
        },
        ProfilePatch(
            replace={"hard_constraints.brands": ["LG"]},
            remove={"usage_preferences": ["daily_laundry"]},
            add={"usage_preferences": ["energy_saving"]},
            clear=["budget_max_vnd"],
            set={"room_area_m2": 20},
        ),
        spec,
    )
    assert profile["hard_constraints"]["brands"] == ["LG"]
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
    removed = "metadata.wash_capacity_kg"
    del client.payload_schema[removed]
    assert ensure_payload_indexes(
        client, "maygiat", config["payload_indexes"], apply=False
    ) == {removed: "float"}
    assert client.created == []
    assert ensure_payload_indexes(
        client, "maygiat", config["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "float")]
    assert ensure_payload_indexes(
        client, "maygiat", config["payload_indexes"], apply=True
    ) == {}
    assert client.created == [(removed, "float")]


def test_prompts_and_no_match_text_are_category_specific() -> None:
    spec = get_category_spec()
    extraction = spec.build_need_extraction_prompt("Cần máy giặt", {})
    ranking = spec.build_ranking_prompt(
        {"need_profile": {}, "hard_constraints": {}, "candidates": []}
    )
    response = spec.build_response_prompt(
        {"need_profile": {}, "selected_products": []}
    )
    combined = " ".join((extraction, ranking, response, no_match_answer({})))
    assert "máy giặt" in combined.casefold()
    assert "khối lượng" in combined.casefold()
    assert "tủ lạnh" not in combined.casefold()
    assert "btu" not in combined.casefold()


def test_generic_washing_machine_query_interrupts_and_resumes_to_advice() -> None:
    llm = FakeLLM()
    qdrant = FakeQdrant()
    graph = build_graph(llm=llm, qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "washing-machine-flow"}}

    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn giúp tôi một chiếc máy giặt")]},
        config,
    )
    assert [item["question_id"] for item in interrupted["__interrupt__"][0].value["questions"]] == [
        "household_size",
        "budget",
        "usage_preferences",
    ]

    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "household_size", "option_id": "three_five"},
                    {"question_id": "budget", "option_id": "8m_12m"},
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
    assert completed["conversation"]["active_category"] == "washing_machine"
    assert completed["need_profile"]["household_size"] == 5
    assert completed["need_profile"]["budget_max_vnd"] == 12_000_000
    assert completed["ranking"]["selected_products"][0]["product_id"] == "wm-1"
    assert completed["response"]["answer"].startswith("Mình đề xuất")
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "maygiat"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_other_household_answer_is_interpreted_without_second_interrupt() -> None:
    llm = FakeLLM(
        custom=WashingMachineCustomAnswer(
            interpretation_status="custom_value",
            raw_answer="Nhà có 9 người",
            household_size=9,
            confidence=1,
        )
    )
    graph = build_graph(llm=llm, qdrant_client=FakeQdrant())
    config = {"configurable": {"thread_id": "washing-machine-custom"}}
    graph.invoke({"messages": [HumanMessage(content="Mua máy giặt")]}, config)
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {
                        "question_id": "household_size",
                        "option_id": "other",
                        "custom_answer": "Nhà có 9 người",
                    },
                    {"question_id": "budget", "option_id": "12m_20m"},
                    {
                        "question_id": "usage_preferences",
                        "option_id": "wash_and_dry",
                    },
                ]
            }
        ),
        config,
    )
    assert "__interrupt__" not in completed
    assert completed["need_profile"]["household_size"] == 9
    assert completed["need_profile"]["hard_constraints"]["dryer"] is True
    assert completed["need_profile"]["custom_answers"]["household_size"][
        "raw_answer"
    ] == "Nhà có 9 người"
