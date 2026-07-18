"""FastAPI transport for streamed chat and LangGraph human-in-the-loop resume."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from advisor.categories.registry import CategoryRegistry, build_default_registry
from advisor.graph import build_graph
from advisor.persistence.checkpointer import open_async_sqlite_checkpointer
from advisor.retrieval.qdrant import (
    AdvisorConfigurationError,
    create_qdrant_client,
    find_missing_indexes,
)
from advisor.schemas import ApplicationSettings, ClarificationAnswer


logger = logging.getLogger(__name__)
PUBLIC_IMAGE_DIR = Path(__file__).resolve().parents[2] / "public"

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

PUBLIC_PROGRESS_STAGES = {
    "intent_detected",
    "need_extracted",
    "clarification_ready",
    "clarification_completed",
    "filter_built",
    "retrieval_completed",
    "ranking_completed",
}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    thread_id: UUID | None = None

    @model_validator(mode="after")
    def reject_blank_message(self) -> ChatRequest:
        self.message = self.message.strip()
        if not self.message:
            raise ValueError("message must not be blank")
        return self


class ResumeRequest(BaseModel):
    answers: list[ClarificationAnswer] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def reject_duplicate_questions(self) -> ResumeRequest:
        ids = [answer.question_id for answer in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("Each clarification question may be answered only once")
        return self


class ThreadStatusResponse(BaseModel):
    thread_id: UUID
    status: Literal["running", "waiting_for_clarification", "completed"]
    questions: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    selected_products: list[dict[str, Any]] = Field(default_factory=list)


class ThreadRunRegistry:
    """Single-process guard against concurrent mutations of one checkpoint thread."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._running: set[str] = set()

    async def try_acquire(self, thread_id: str) -> bool:
        async with self._guard:
            if thread_id in self._running:
                return False
            self._running.add(thread_id)
            return True

    async def release(self, thread_id: str) -> None:
        async with self._guard:
            self._running.discard(thread_id)

    async def is_running(self, thread_id: str) -> bool:
        async with self._guard:
            return thread_id in self._running


def _json_default(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value"):
        return value.value
    return str(value)


def encode_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=_json_default, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _message_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def _find_interrupt_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if "__interrupt__" in value:
            interrupts = value["__interrupt__"] or []
            for item in interrupts:
                payload = getattr(item, "value", item)
                if isinstance(payload, dict):
                    return payload
        for nested in value.values():
            found = _find_interrupt_payload(nested)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_interrupt_payload(nested)
            if found:
                return found
    return None


def _find_progress_stages(value: Any) -> list[str]:
    stages: list[str] = []
    if isinstance(value, dict):
        control = value.get("control")
        if isinstance(control, dict) and control.get("stage") in PUBLIC_PROGRESS_STAGES:
            stages.append(control["stage"])
        for nested in value.values():
            stages.extend(_find_progress_stages(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            stages.extend(_find_progress_stages(nested))
    return stages


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _snapshot_exists(snapshot: Any) -> bool:
    return bool(getattr(snapshot, "created_at", None)) or bool(
        getattr(snapshot, "values", None)
    )


async def _get_snapshot(app: FastAPI, thread_id: str) -> Any:
    return await app.state.graph.aget_state(_thread_config(thread_id))


def _is_waiting(values: dict[str, Any]) -> bool:
    clarification = values.get("clarification") or {}
    return clarification.get("status") == "pending" and bool(
        clarification.get("questions")
    )


def _public_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hide options removed from the current clarification contract."""
    return [
        {
            **question,
            "options": [
                option
                for option in question.get("options", [])
                if option.get("option_id") != "skip"
            ],
        }
        for question in questions
    ]


def _status_from_values(
    thread_id: UUID, values: dict[str, Any], *, running: bool
) -> ThreadStatusResponse:
    clarification = values.get("clarification") or {}
    if running:
        status = "running"
    elif _is_waiting(values):
        status = "waiting_for_clarification"
    elif (values.get("control") or {}).get("stage") == "completed":
        status = "completed"
    else:
        status = "running"
    return ThreadStatusResponse(
        thread_id=thread_id,
        status=status,
        questions=(
            _public_questions(clarification.get("questions", []))
            if _is_waiting(values)
            else []
        ),
        answer=(values.get("response") or {}).get("answer"),
        selected_products=(values.get("ranking") or {}).get("selected_products", []),
    )


def _validate_answers(values: dict[str, Any], submission: ResumeRequest) -> None:
    questions = _public_questions(
        (values.get("clarification") or {}).get("questions", [])
    )
    expected = {question["question_id"] for question in questions}
    received = {answer.question_id for answer in submission.answers}
    if received != expected:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_answers",
                "message": "Answers must match every pending question exactly once.",
                "missing": sorted(expected - received),
                "unexpected": sorted(received - expected),
            },
        )
    allowed = {
        question["question_id"]: {
            option["option_id"] for option in question.get("options", [])
        }
        for question in questions
    }
    invalid = [
        {"question_id": answer.question_id, "option_id": answer.option_id}
        for answer in submission.answers
        if answer.option_id not in allowed.get(answer.question_id, set())
    ]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_options",
                "message": "One or more option IDs do not belong to the pending form.",
                "invalid": invalid,
            },
        )


async def _stream_graph(
    *,
    request: Request,
    graph_input: dict[str, Any] | Command,
    thread_id: str,
    session_mode: Literal["started", "continued", "resumed"],
) -> AsyncIterator[str]:
    app = request.app
    registry: ThreadRunRegistry = app.state.thread_registry
    terminal_emitted = False
    progress_emitted: set[str] = set()
    iterator: Any | None = None
    next_task: asyncio.Task[Any] | None = None
    try:
        yield encode_sse(
            "session", {"thread_id": thread_id, "mode": session_mode}
        )
        iterator = app.state.graph.astream(
            graph_input,
            config=_thread_config(thread_id),
            stream_mode=["messages", "updates"],
            version="v2",
        ).__aiter__()
        while True:
            next_task = asyncio.create_task(anext(iterator))
            while not next_task.done():
                done, _ = await asyncio.wait(
                    {next_task}, timeout=app.state.settings.sse_heartbeat_seconds
                )
                if done:
                    break
                if await request.is_disconnected():
                    next_task.cancel()
                    return
                yield ": heartbeat\n\n"
            try:
                part = next_task.result()
            except StopAsyncIteration:
                break
            next_task = None
            part_type = part.get("type") if isinstance(part, dict) else None
            data = part.get("data") if isinstance(part, dict) else None
            if part_type == "messages" and isinstance(data, (list, tuple)) and len(data) == 2:
                chunk, metadata = data
                if (metadata or {}).get("langgraph_node") == "compose_response":
                    delta = _message_chunk_text(chunk)
                    if delta:
                        yield encode_sse("token", {"delta": delta})
            elif part_type == "updates":
                for stage in _find_progress_stages(data):
                    if stage not in progress_emitted:
                        progress_emitted.add(stage)
                        yield encode_sse("progress", {"stage": stage})
                interrupt_payload = _find_interrupt_payload(data)
                if interrupt_payload:
                    terminal_emitted = True
                    yield encode_sse(
                        "clarification_required",
                        {
                            "thread_id": thread_id,
                            "message": interrupt_payload.get("message", ""),
                            "questions": _public_questions(
                                interrupt_payload.get("questions", [])
                            ),
                        },
                    )

        if not terminal_emitted:
            snapshot = await _get_snapshot(app, thread_id)
            values = dict(snapshot.values)
            if _is_waiting(values):
                terminal_emitted = True
                yield encode_sse(
                    "clarification_required",
                    {
                        "thread_id": thread_id,
                        "message": values["clarification"].get("message", ""),
                        "questions": _public_questions(
                            values["clarification"]["questions"]
                        ),
                    },
                )
            elif (values.get("control") or {}).get("stage") == "completed":
                terminal_emitted = True
                yield encode_sse(
                    "completed",
                    {
                        "thread_id": thread_id,
                        "answer": (values.get("response") or {}).get("answer", ""),
                        "selected_products": (values.get("ranking") or {}).get(
                            "selected_products", []
                        ),
                    },
                )
        if not terminal_emitted:
            yield encode_sse(
                "error",
                {
                    "code": "incomplete_run",
                    "message": "The graph stopped without a terminal result.",
                    "retryable": True,
                },
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Chat stream failed for thread_id=%s", thread_id)
        configuration_error = isinstance(exc, AdvisorConfigurationError)
        yield encode_sse(
            "error",
            {
                "code": "configuration_error" if configuration_error else "service_error",
                "message": (
                    "The advisor service is not configured correctly."
                    if configuration_error
                    else "The advisor could not complete this request."
                ),
                "retryable": not configuration_error,
            },
        )
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
        if iterator is not None and hasattr(iterator, "aclose"):
            try:
                await iterator.aclose()
            except (RuntimeError, asyncio.CancelledError):
                pass
        await registry.release(thread_id)


def _streaming_response(iterator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        iterator,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def create_app(
    settings: ApplicationSettings | None = None,
    *,
    graph: Any | None = None,
    llm: Any | None = None,
    qdrant_client: Any | None = None,
    category_registry: CategoryRegistry | None = None,
    validate_services: bool = True,
) -> FastAPI:
    """Create the API app with injectable graph dependencies for tests."""
    app_settings = settings or ApplicationSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = app_settings
        application.state.thread_registry = ThreadRunRegistry()
        application.state.ready = False
        if graph is not None:
            application.state.graph = graph
            application.state.ready = True
            yield
            application.state.ready = False
            return

        if llm is None and not app_settings.google_api_key:
            raise AdvisorConfigurationError("GOOGLE_API_KEY is required")
        client = qdrant_client or create_qdrant_client(app_settings)
        owns_client = qdrant_client is None
        resolved_registry = category_registry or build_default_registry()
        resolved_registry.validate_all()
        try:
            if validate_services:
                for category, definition in resolved_registry.all().items():
                    if not definition.implemented:
                        continue
                    spec = resolved_registry.get_spec(category)
                    missing = await asyncio.to_thread(
                        find_missing_indexes,
                        client,
                        spec.config["collection"],
                        spec.config["payload_indexes"],
                    )
                    if missing:
                        hint = (
                            f" Run `{spec.setup_indexes_command}`."
                            if spec.setup_indexes_command
                            else ""
                        )
                        raise AdvisorConfigurationError(
                            f"Qdrant category {category!r} is missing payload "
                            f"indexes: {', '.join(sorted(missing))}.{hint}"
                        )
            async with open_async_sqlite_checkpointer(app_settings) as checkpointer:
                application.state.graph = build_graph(
                    settings=app_settings,
                    checkpointer=checkpointer,
                    llm=llm,
                    qdrant_client=client,
                    category_registry=resolved_registry,
                )
                application.state.ready = True
                yield
                application.state.ready = False
        finally:
            if owns_client and hasattr(client, "close"):
                client.close()

    application = FastAPI(
        title="Product Advisor API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    if PUBLIC_IMAGE_DIR.is_dir():
        application.mount(
            "/product-images",
            StaticFiles(directory=PUBLIC_IMAGE_DIR),
            name="product-images",
        )

    sse_response_docs = {
        200: {
            "description": "Server-Sent Events stream",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    }

    @application.get("/healthz")
    async def healthz(request: Request) -> dict[str, str]:
        return {"status": "ok" if request.app.state.ready else "starting"}

    @application.get("/chat/{thread_id}", response_model=ThreadStatusResponse)
    async def get_chat_status(thread_id: UUID, request: Request) -> ThreadStatusResponse:
        thread_key = str(thread_id)
        running = await request.app.state.thread_registry.is_running(thread_key)
        snapshot = await _get_snapshot(request.app, thread_key)
        if not running and not _snapshot_exists(snapshot):
            raise HTTPException(status_code=404, detail="Thread not found")
        values = dict(snapshot.values) if _snapshot_exists(snapshot) else {}
        return _status_from_values(thread_id, values, running=running)

    @application.post("/chat", responses=sse_response_docs)
    async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        thread_uuid = payload.thread_id or uuid4()
        thread_key = str(thread_uuid)
        registry: ThreadRunRegistry = request.app.state.thread_registry
        if not await registry.try_acquire(thread_key):
            raise HTTPException(status_code=409, detail="Thread is already running")
        try:
            mode: Literal["started", "continued"] = "started"
            if payload.thread_id is not None:
                snapshot = await _get_snapshot(request.app, thread_key)
                if not _snapshot_exists(snapshot):
                    raise HTTPException(status_code=404, detail="Thread not found")
                values = dict(snapshot.values)
                if _is_waiting(values):
                    raise HTTPException(
                        status_code=409,
                        detail="Thread is waiting for clarification answers",
                    )
                if (values.get("control") or {}).get("stage") != "completed":
                    raise HTTPException(status_code=409, detail="Thread is not ready")
                mode = "continued"
            return _streaming_response(
                _stream_graph(
                    request=request,
                    graph_input={"messages": [HumanMessage(content=payload.message)]},
                    thread_id=thread_key,
                    session_mode=mode,
                )
            )
        except Exception:
            await registry.release(thread_key)
            raise

    @application.post("/chat/{thread_id}/resume", responses=sse_response_docs)
    async def resume_chat(
        thread_id: UUID, payload: ResumeRequest, request: Request
    ) -> StreamingResponse:
        thread_key = str(thread_id)
        registry: ThreadRunRegistry = request.app.state.thread_registry
        if not await registry.try_acquire(thread_key):
            raise HTTPException(status_code=409, detail="Thread is already running")
        try:
            snapshot = await _get_snapshot(request.app, thread_key)
            if not _snapshot_exists(snapshot):
                raise HTTPException(status_code=404, detail="Thread not found")
            values = dict(snapshot.values)
            if not _is_waiting(values):
                raise HTTPException(
                    status_code=409,
                    detail="Thread is not waiting for clarification",
                )
            _validate_answers(values, payload)
            command = Command(
                resume={
                    "answers": [answer.model_dump(mode="json") for answer in payload.answers]
                }
            )
            return _streaming_response(
                _stream_graph(
                    request=request,
                    graph_input=command,
                    thread_id=thread_key,
                    session_mode="resumed",
                )
            )
        except Exception:
            await registry.release(thread_key)
            raise

    return application


app = create_app()
