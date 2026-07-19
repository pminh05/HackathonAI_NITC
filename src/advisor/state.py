"""LangGraph state for one product-advisor conversation."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AdvisorState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    identity: dict[str, Any]
    memory_context: dict[str, Any]
    conversation: dict[str, Any]
    category_contexts: dict[str, dict[str, Any]]
    routing: dict[str, Any]
    need_profile: dict[str, Any]
    clarification: dict[str, Any]
    retrieval: dict[str, Any]
    ranking: dict[str, Any]
    response: dict[str, Any]
    control: dict[str, Any]


NodeUpdate = dict[str, Any]
