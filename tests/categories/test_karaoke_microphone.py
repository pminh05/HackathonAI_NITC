"""Behavior and data-contract tests for the micro-karaoke category."""

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

from advisor.categories.karaoke_microphone import (
    get_category_spec,
    get_missing_profile_fields,
    load_config,
    no_match_answer,
)
from advisor.categories.karaoke_microphone.filter_builder import (
    STANDARDIZED_METADATA_PATHS,
    build_filter,
)
from advisor.categories.karaoke_microphone.normalizer import normalize_candidate
from advisor.categories.karaoke_microphone.schemas import (
    KaraokeMicrophoneNeedProfile,
)
from advisor.categories.karaoke_microphone.setup_indexes import (
    ensure_payload_indexes,
)
from advisor.categories.refrigerator import load_config as load_refrigerator_config
from advisor.categories.registry import build_default_registry
from advisor.graph import build_graph
from advisor.nodes import apply_profile_patch
from advisor.schemas import IntentLabel, ProfilePatch, RankingResult, TurnAnalysisResult


ROOT = Path(__file__).parents[2]
INGESTION = runpy.run_path(ROOT / "ingestion/micro_karaoke/changeName.py")
QDRANT = runpy.run_path(ROOT / "ingestion/micro_karaoke/qdrant.py")
normalize_metadata = INGESTION["normalize_metadata"]


def karaoke_point() -> Any:
    return SimpleNamespace(
        id="point-mic-1",
        score=0.94,
        payload={
            "product_id": "mic-1",
            "name": "Micro karaoke JBL Test",
            "text": "Micro karaoke không dây UHF, dải âm 40–20000 Hz.",
            "image_path": "/public/micro_karaoke.jpg",
            "metadata": {
                "category_scope": "karaoke_microphone",
                "brand": "Jbl",
                "brand_key": "jbl",
                "model_code": "MIC-X",
                "original_price_vnd": 4_000_000,
                "promotional_price_vnd": 3_500_000,
                "microphone_type": "wireless",
                "wireless_band": "uhf",
                "rf_frequency_min_mhz": 640.0,
                "rf_frequency_max_mhz": 690.0,
                "audio_frequency_min_hz": 40.0,
                "audio_frequency_max_hz": 20_000.0,
                "distortion_pct": 0.5,
                "distortion_operator": "Nhỏ hơn",
                "manufacture_year": 2025,
                "origin": "Việt Nam",
                "data_quality_flags": [],
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
                category=IntentLabel.KARAOKE_MICROPHONE,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            if self.owner.patches:
                return self.owner.patches.pop(0)
            return ProfilePatch()
        if self.schema is RankingResult:
            if self.owner.rankings:
                return self.owner.rankings.pop(0)
            return RankingResult(
                selected_products=[
                    {
                        "product_id": "mic-1",
                        "reason": "Đúng loại không dây và nhu cầu karaoke gia đình.",
                        "trade_off": "Chưa có dữ liệu xác minh phạm vi thu sóng.",
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
        rankings: list[RankingResult] | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.patches = list(patches or [])
        self.rankings = list(rankings or [])

    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, _: str) -> AIMessage:
        return AIMessage(content="Mình đề xuất micro karaoke phù hợp nhu cầu của bạn.")


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
        return SimpleNamespace(points=[karaoke_point()])


def test_ingestion_normalizes_filter_safe_metadata() -> None:
    metadata = normalize_metadata(
        {
            "brand": "JBL",
            "model_code": "MIC-X",
            "Loại sản phẩm": "Không dây",
            "Băng tần": "UHF",
            "Tần số hoạt động": "640 - 690 MHz",
            "loai_san_pham_chuan": "Không dây",
            "bang_tan_chuan": "UHF",
            "loai_du_lieu_tan_so": "Gồm cả sóng và âm thanh",
            "tan_so_song_min_mhz": 690,
            "tan_so_song_max_mhz": 690,
            "tan_so_am_thanh_min_hz": 40,
            "tan_so_am_thanh_max_hz": 20_000,
            "do_meo_tieng_pct": 0.5,
            "toan_tu_do_meo": "Nhỏ hơn",
            "nam_san_xuat_chuan": 2025,
            "giá gốc": "4.000.000 đ",
            "giá khuyến mãi": "3.500.000 đ",
        }
    )
    assert metadata["category_scope"] == "karaoke_microphone"
    assert metadata["brand_key"] == "jbl"
    assert metadata["microphone_type"] == "wireless"
    assert metadata["wireless_band"] == "uhf"
    assert metadata["rf_frequency_min_mhz"] == 640.0
    assert metadata["rf_frequency_max_mhz"] == 690.0
    assert metadata["promotional_price_vnd"] == 3_500_000


def test_source_catalog_normalizes_all_37_unique_products() -> None:
    payload = json.loads(
        (ROOT / "ingestion/micro_karaoke/data/micro_karaoke.json").read_text(
            encoding="utf-8-sig"
        )
    )
    products = payload["data_clean"]
    normalized = [normalize_metadata(product) for product in products]
    assert len(products) == len({product["sku"] for product in products}) == 37
    assert all(item["category_scope"] == "karaoke_microphone" for item in normalized)
    assert all(item.get("brand") and item.get("brand_key") for item in normalized)
    assert sum("original_price_vnd" in item for item in normalized) == 5
    assert not any(
        "rf_frequency_min_mhz" in item and item.get("microphone_type") == "wired"
        for item in normalized
    )


def test_ingestion_omits_unsafe_frequency_fields() -> None:
    metadata = normalize_metadata(
        {
            "brand": "Lane",
            "Loại sản phẩm": "Có dây",
            "loai_san_pham_chuan": "Có dây",
            "tan_so_song_min_mhz": 100,
            "tan_so_song_max_mhz": 100,
            "tan_so_am_thanh_min_hz": 700,
            "tan_so_am_thanh_max_hz": 700,
            "kiem_tra_du_lieu": "Dải tần âm thanh bất thường",
        }
    )
    assert metadata["microphone_type"] == "wired"
    assert "wireless_band" not in metadata
    assert "rf_frequency_min_mhz" not in metadata
    assert "audio_frequency_min_hz" not in metadata
    assert metadata["data_quality_flags"] == ["Dải tần âm thanh bất thường"]


def test_ingestion_point_shape_uses_canonical_cloud_inference_payload() -> None:
    product = {
        "id": "mic-sku",
        "name": "Micro karaoke Test",
        "text": "Micro karaoke không dây UHF.",
        "image_path": "/public/micro_karaoke.jpg",
        "metadata": {
            "category_scope": "karaoke_microphone",
            "brand": "Jbl",
            "brand_key": "jbl",
        },
    }
    first = next(QDRANT["point_generator"]([product]))
    second = next(QDRANT["point_generator"]([product]))
    assert first.id == second.id
    assert first.vector.model == "intfloat/multilingual-e5-small"
    assert first.vector.text == "passage: Micro karaoke không dây UHF."
    assert first.payload["metadata"]["category_scope"] == "karaoke_microphone"
    assert QDRANT["INPUT_FILE"].name == "micro_karaoke_processed_vi.json"
    with pytest.raises(ValueError):
        next(
            QDRANT["point_generator"](
                [{**product, "metadata": {"brand": "Jbl", "brand_key": "jbl"}}]
            )
        )


def test_spec_profile_questions_and_registry_contract() -> None:
    spec = get_category_spec()
    spec.validate()
    config = load_config()
    assert spec.name == config["category"] == "karaoke_microphone"
    assert spec.display_name == "Micro karaoke"
    assert config["collection"] == "microkaraoke"
    assert set(config["payload_fields"].values()) <= STANDARDIZED_METADATA_PATHS
    assert build_default_registry().get_spec("karaoke_microphone").name == (
        "karaoke_microphone"
    )
    profile = KaraokeMicrophoneNeedProfile(
        connection_preference="wireless",
        usage_preferences=["home_family"],
        budget_segment="open",
    )
    assert get_missing_profile_fields(profile.model_dump()) == []
    assert get_missing_profile_fields({}) == [
        "usage_context",
        "connection_preference",
        "budget",
    ]
    budget_options = {
        option["option_id"]
        for option in config["question_catalog"]["budget"]["options"]
    }
    assert budget_options == {
        "under_2m",
        "2m_5m",
        "5m_10m",
        "over_10m",
        "open",
        "other",
    }
    with pytest.raises(ValidationError):
        KaraokeMicrophoneNeedProfile(usage_preferences=["weekly_storage"])


def test_filter_enforces_only_indexed_hard_fields() -> None:
    query_filter = build_filter(
        {
            "connection_preference": "wireless",
            "budget_max_vnd": 5_000_000,
            "soft_preferences": ["anti_feedback", "long_battery"],
            "hard_constraints": {
                "brands": ["JBL"],
                "wireless_bands": ["uhf"],
            },
        }
    )
    text = str(query_filter.model_dump(mode="json", exclude_none=True))
    for path in (
        "metadata.category_scope",
        "metadata.brand_key",
        "metadata.microphone_type",
        "metadata.wireless_band",
        "metadata.original_price_vnd",
        "metadata.promotional_price_vnd",
    ):
        assert path in text
    assert "jbl" in text
    assert "anti_feedback" not in text
    assert "long_battery" not in text


def test_filter_keeps_unknown_price_but_excludes_over_budget_and_wrong_type() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="micro-test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    base = {
        "category_scope": "karaoke_microphone",
        "brand_key": "jbl",
        "wireless_band": "uhf",
    }
    client.upsert(
        collection_name="micro-test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "microphone_type": "wireless",
                        "promotional_price_vnd": 3_500_000,
                    }
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={"metadata": {**base, "microphone_type": "wireless"}},
            ),
            models.PointStruct(
                id=3,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "microphone_type": "wired",
                        "original_price_vnd": 2_000_000,
                    }
                },
            ),
            models.PointStruct(
                id=4,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        **base,
                        "microphone_type": "wireless",
                        "original_price_vnd": 12_000_000,
                    }
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
                "connection_preference": "wireless",
                "budget_max_vnd": 5_000_000,
                "hard_constraints": {"brands": ["JBL"]},
            }
        ),
        limit=10,
    )
    assert {point.id for point in result.points} == {1, 2}


def test_normalizer_patch_indexes_prompts_and_no_match() -> None:
    candidate = normalize_candidate(karaoke_point())
    assert candidate["effective_price_vnd"] == 3_500_000
    assert candidate["microphone_type"] == "wireless"
    assert candidate["rf_frequency_mhz"] == {"min": 640.0, "max": 690.0}
    assert "metadata" not in candidate
    spec = get_category_spec()
    profile, changed = apply_profile_patch(
        {
            "connection_preference": "wired",
            "usage_preferences": ["home_family"],
            "hard_constraints": {"brands": []},
        },
        ProfilePatch(
            set={"connection_preference": "wireless", "household_size": 4},
            add={"hard_constraints.brands": ["jbl"]},
            replace={"usage_preferences": ["stage_event"]},
        ),
        spec,
    )
    assert profile["connection_preference"] == "wireless"
    assert profile["usage_preferences"] == ["stage_event"]
    assert profile["hard_constraints"]["brands"] == ["jbl"]
    assert "household_size" not in profile
    assert set(changed) == {
        "connection_preference",
        "usage_preferences",
        "hard_constraints.brands",
    }
    fake = FakeQdrant()
    removed = "metadata.wireless_band"
    del fake.payload_schema[removed]
    assert ensure_payload_indexes(
        fake, "microkaraoke", load_config()["payload_indexes"]
    ) == {removed: "keyword"}
    assert ensure_payload_indexes(
        fake, "microkaraoke", load_config()["payload_indexes"], apply=True
    ) == {}
    combined = " ".join(
        (
            spec.build_need_extraction_prompt("Cần micro karaoke", {}),
            spec.build_ranking_prompt(
                {"need_profile": {}, "hard_constraints": {}, "candidates": []}
            ),
            spec.build_response_prompt(
                {"need_profile": {}, "selected_products": []}
            ),
            no_match_answer({"budget_max_vnd": 3_000_000}),
        )
    ).casefold()
    assert "micro karaoke" in combined
    assert "tủ lạnh" not in combined
    assert "chưa có giá xác minh" in combined
    assert "không thể khẳng định" in combined


def test_graph_interrupts_and_resumes_to_microkaraoke_collection() -> None:
    qdrant = FakeQdrant()
    graph = build_graph(llm=FakeLLM(), qdrant_client=qdrant)
    config = {"configurable": {"thread_id": "karaoke-microphone-flow"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn micro karaoke")]}, config
    )
    assert [
        item["question_id"] for item in interrupted["__interrupt__"][0].value["questions"]
    ] == ["usage_context", "connection_preference", "budget"]
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "usage_context", "option_id": "home_family"},
                    {
                        "question_id": "connection_preference",
                        "option_id": "wireless",
                    },
                    {"question_id": "budget", "option_id": "open"},
                ]
            }
        ),
        config,
    )
    assert completed["conversation"]["active_category"] == "karaoke_microphone"
    assert completed["need_profile"]["connection_preference"] == "wireless"
    assert completed["need_profile"]["usage_preferences"] == ["home_family"]
    assert completed["need_profile"]["budget_segment"] == "open"
    assert completed["ranking"]["selected_products"][0]["product_id"] == "mic-1"
    assert qdrant.query_kwargs is not None
    assert qdrant.query_kwargs["collection_name"] == "microkaraoke"
    assert qdrant.query_kwargs["query"].model == "intfloat/multilingual-e5-small"


def test_switch_refrigerator_karaoke_and_back_restores_context() -> None:
    analyses = [
        TurnAnalysisResult(
            category=IntentLabel.REFRIGERATOR,
            category_transition="new",
            action="discover",
            has_profile_update=True,
        ),
        TurnAnalysisResult(
            category=IntentLabel.KARAOKE_MICROPHONE,
            category_transition="switch",
            switch_evidence="micro karaoke",
            action="switch_category",
        ),
        TurnAnalysisResult(
            category=IntentLabel.REFRIGERATOR,
            category_transition="switch",
            switch_evidence="tủ lạnh",
            action="switch_category",
        ),
    ]
    refrigerator_patch = ProfilePatch(
        set={"household_size": 4, "budget_max_vnd": 20_000_000},
        replace={"usage_preferences": ["weekly_storage"]},
    )
    ranking = RankingResult(
        selected_products=[
            {"product_id": "mic-1", "reason": "Phù hợp.", "trade_off": "A."}
        ]
    )
    qdrant = FakeQdrant()
    graph = build_graph(
        llm=FakeLLM(
            analyses=analyses,
            patches=[refrigerator_patch],
            rankings=[ranking, ranking],
        ),
        qdrant_client=qdrant,
    )
    config = {"configurable": {"thread_id": "fridge-karaoke-fridge"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Tư vấn tủ lạnh cho 4 người")]}, config
    )
    assert first["conversation"]["active_category"] == "refrigerator"
    switched = graph.invoke(
        {"messages": [HumanMessage(content="Chuyển qua micro karaoke")]}, config
    )
    assert switched["conversation"]["active_category"] == "karaoke_microphone"
    completed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "usage_context", "option_id": "home_family"},
                    {
                        "question_id": "connection_preference",
                        "option_id": "wireless",
                    },
                    {"question_id": "budget", "option_id": "open"},
                ]
            }
        ),
        config,
    )
    assert completed["need_profile"]["connection_preference"] == "wireless"
    restored = graph.invoke(
        {"messages": [HumanMessage(content="Quay lại tủ lạnh")]}, config
    )
    assert restored["conversation"]["active_category"] == "refrigerator"
    assert restored["need_profile"]["household_size"] == 4
    assert qdrant.query_count == 2
