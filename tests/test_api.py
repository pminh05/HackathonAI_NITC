"""HTTP/SSE and persistence tests for the FastAPI transport."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from advisor.api import create_app
from advisor.categories.air_conditioner import load_config as load_air_conditioner_config
from advisor.categories.refrigerator import load_config
from advisor.categories.washing_machine import load_config as load_washing_machine_config
from advisor.schemas import (
    ClarificationDecision,
    CustomAnswerInterpretation,
    IntentLabel,
    IntentResult,
    ProfilePatch,
    RankingResult,
    RefrigeratorNeedExtraction,
    TurnAnalysisResult,
    ApplicationSettings,
)


class FakeStructuredModel:
    def __init__(self, owner: FakeLLM, schema: type[Any]) -> None:
        self.owner = owner
        self.schema = schema

    def invoke(self, _: str) -> Any:
        if self.schema is TurnAnalysisResult:
            return TurnAnalysisResult(
                category=IntentLabel.REFRIGERATOR,
                category_transition="new",
                action="discover",
                has_profile_update=True,
            )
        if self.schema is ProfilePatch:
            return ProfilePatch()
        if self.schema is IntentResult:
            return IntentResult(label=IntentLabel.REFRIGERATOR)
        if self.schema is RefrigeratorNeedExtraction:
            return RefrigeratorNeedExtraction()
        if self.schema is ClarificationDecision:
            return ClarificationDecision(
                sufficient=False,
                question_ids=["household_size", "budget", "usage_preferences"],
            )
        if self.schema is RankingResult:
            return RankingResult(
                selected_products=[
                    {
                        "product_id": "sku-api",
                        "reason": "Phù hợp nhu cầu.",
                        "trade_off": "Ít tiện ích hơn.",
                    }
                ]
            )
        if self.schema is CustomAnswerInterpretation:
            return CustomAnswerInterpretation(
                interpretation_status="custom_value",
                raw_answer="7 người",
                household_size=7,
                confidence=1,
            )
        raise AssertionError(self.schema)


class FakeLLM:
    def with_structured_output(self, schema: type[Any], **_: Any) -> FakeStructuredModel:
        return FakeStructuredModel(self, schema)

    def invoke(self, _: str) -> AIMessage:
        return AIMessage(content="Đây là câu trả lời tư vấn từ API.")


class FakeQdrant:
    def __init__(self) -> None:
        self.payload_schemas = {
            config["collection"]: {
                field: SimpleNamespace(data_type=SimpleNamespace(value=schema))
                for field, schema in config["payload_indexes"].items()
            }
            for config in (
                load_config(),
                load_air_conditioner_config(),
                load_washing_machine_config(),
            )
        }

    def get_collection(self, collection: str) -> Any:
        return SimpleNamespace(payload_schema=self.payload_schemas[collection])

    def query_points(self, **_: Any) -> Any:
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-api",
                    score=0.9,
                    payload={
                        "product_id": "sku-api",
                        "name": "Tủ lạnh API",
                        "text": "Tủ lạnh 350 lít có Inverter.",
                        "image_path": "/catalog/public/tu_lanh.jpg",
                        "metadata": {
                            "brand": "Test",
                            "Giá gốc vnd": 18_000_000,
                            "Giá khuyến mãi vnd": 15_000_000,
                            "Dung tích sử dụng lít": 350,
                            "Số người sử dụng": "3 - 4 người",
                            "Có inverter": True,
                        },
                    },
                )
            ]
        )


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block or block.startswith(":"):
            continue
        event = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data += line.removeprefix("data: ")
        if data:
            import json

            events.append((event, json.loads(data)))
    return events


def api_settings(tmp_path: Any) -> ApplicationSettings:
    return ApplicationSettings(
        _env_file=None,
        checkpoint_db_path=tmp_path / "api-checkpoints.sqlite",
        sse_heartbeat_seconds=0.05,
    )


def test_chat_interrupt_status_resume_and_duplicate_resume(tmp_path: Any) -> None:
    application = create_app(
        api_settings(tmp_path), llm=FakeLLM(), qdrant_client=FakeQdrant()
    )
    with TestClient(application) as client:
        image_response = client.get("/product-images/tu_lanh.jpg")
        assert image_response.status_code == 200
        assert image_response.headers["content-type"] == "image/jpeg"

        response = client.post("/chat", json={"message": "Tư vấn tủ lạnh"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        assert [event for event, _ in events][-1] == "clarification_required"
        thread_id = events[0][1]["thread_id"]
        questions = events[-1][1]["questions"]
        assert len(questions) == 3

        status = client.get(f"/chat/{thread_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "waiting_for_clarification"

        pending_chat = client.post(
            "/chat",
            json={"message": "Tin nhắn mới", "thread_id": thread_id},
        )
        assert pending_chat.status_code == 409

        invalid = client.post(
            f"/chat/{thread_id}/resume",
            json={
                "answers": [
                    {"question_id": "household_size", "option_id": "bad"},
                    {"question_id": "budget", "option_id": "10m_20m"},
                    {"question_id": "usage_preferences", "option_id": "energy_saving"},
                ]
            },
        )
        assert invalid.status_code == 422

        resumed = client.post(
            f"/chat/{thread_id}/resume",
            json={
                "answers": [
                    {"question_id": "household_size", "option_id": "three_four"},
                    {"question_id": "budget", "option_id": "10m_20m"},
                    {"question_id": "usage_preferences", "option_id": "energy_saving"},
                ]
            },
        )
        resumed_events = parse_sse(resumed.text)
        assert resumed_events[-1][0] == "completed"
        assert resumed_events[-1][1]["answer"] == "Đây là câu trả lời tư vấn từ API."
        assert "".join(
            data["delta"] for event, data in resumed_events if event == "token"
        ) == "Đây là câu trả lời tư vấn từ API."
        assert resumed_events[-1][1]["selected_products"][0]["product_id"] == "sku-api"
        assert (
            resumed_events[-1][1]["selected_products"][0]["image_path"]
            == "/catalog/public/tu_lanh.jpg"
        )

        duplicate = client.post(
            f"/chat/{thread_id}/resume",
            json={
                "answers": [
                    {"question_id": "household_size", "option_id": "three_four"}
                ]
            },
        )
        assert duplicate.status_code == 409

        continued = client.post(
            "/chat",
            json={"message": "Tư vấn tiếp", "thread_id": thread_id},
        )
        continued_events = parse_sse(continued.text)
        assert continued_events[0][1]["mode"] == "continued"
        assert continued_events[-1][0] == "completed"


def test_pending_thread_survives_app_restart(tmp_path: Any) -> None:
    settings = api_settings(tmp_path)
    first_app = create_app(settings, llm=FakeLLM(), qdrant_client=FakeQdrant())
    with TestClient(first_app) as client:
        events = parse_sse(
            client.post("/chat", json={"message": "Tôi cần tủ lạnh"}).text
        )
        thread_id = events[0][1]["thread_id"]

    second_app = create_app(settings, llm=FakeLLM(), qdrant_client=FakeQdrant())
    with TestClient(second_app) as client:
        status = client.get(f"/chat/{thread_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "waiting_for_clarification"
        result = client.post(
            f"/chat/{thread_id}/resume",
            json={
                "answers": [
                    {"question_id": "household_size", "option_id": "three_four"},
                    {"question_id": "budget", "option_id": "10m_20m"},
                    {"question_id": "usage_preferences", "option_id": "weekly_storage"},
                ]
            },
        )
        assert parse_sse(result.text)[-1][0] == "completed"


def test_resume_rejects_partial_form(
    tmp_path: Any,
) -> None:
    application = create_app(
        api_settings(tmp_path), llm=FakeLLM(), qdrant_client=FakeQdrant()
    )
    with TestClient(application) as client:
        events = parse_sse(
            client.post("/chat", json={"message": "Tư vấn tủ lạnh"}).text
        )
        thread_id = events[0][1]["thread_id"]
        partial = client.post(
            f"/chat/{thread_id}/resume",
            json={
                "answers": [
                    {"question_id": "household_size", "option_id": "three_four"}
                ]
            },
        )
        assert partial.status_code == 422
        assert partial.json()["detail"]["code"] == "invalid_answers"
        assert partial.json()["detail"]["missing"] == ["budget", "usage_preferences"]


class FakeStreamingGraph:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def astream(self, *_: Any, **__: Any) -> Any:
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content='{"secret":"ranking-json"}'),
                {"langgraph_node": "rank_candidates"},
            ),
        }
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="Xin "),
                {"langgraph_node": "compose_response"},
            ),
        }
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="chào"),
                {"langgraph_node": "compose_response"},
            ),
        }
        self.values = {
            "control": {"stage": "completed"},
            "response": {"answer": "Xin chào"},
            "ranking": {"selected_products": []},
        }
        yield {
            "type": "updates",
            "data": {"compose_response": {"control": {"stage": "completed"}}},
        }

    async def aget_state(self, _: Any) -> Any:
        return SimpleNamespace(
            values=self.values,
            created_at="now" if self.values else None,
        )


def test_sse_only_forwards_plain_response_tokens() -> None:
    application = create_app(graph=FakeStreamingGraph())
    with TestClient(application) as client:
        response = client.post("/chat", json={"message": "hello"})
    events = parse_sse(response.text)
    deltas = [data["delta"] for event, data in events if event == "token"]
    assert deltas == ["Xin ", "chào"]
    assert "ranking-json" not in response.text
    assert events[-1] == (
        "completed",
        {"thread_id": events[0][1]["thread_id"], "answer": "Xin chào", "selected_products": []},
    )


def test_health_unknown_thread_and_cors() -> None:
    application = create_app(graph=FakeStreamingGraph())
    with TestClient(application) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        missing = client.get("/chat/00000000-0000-0000-0000-000000000001")
        assert missing.status_code == 404
        cors = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert cors.headers["access-control-allow-origin"] == "http://localhost:3000"
