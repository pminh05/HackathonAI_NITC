"""LangGraph builder for the refrigerator advisory MVP."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from advisor.nodes import (
    AdvisorRuntime,
    build_filter_node,
    collect_clarification_node,
    compose_response_node,
    detect_intent_node,
    extract_need_profile_node,
    generate_clarification_node,
    placeholder_response_node,
    prepare_turn_node,
    rank_candidates_node,
    retrieve_candidates_node,
)
from advisor.schemas import ApplicationSettings, IntentLabel
from advisor.state import AdvisorState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def _route_intent(state: AdvisorState) -> str:
    if state["routing"]["intent"] == IntentLabel.REFRIGERATOR.value:
        return "refrigerator"
    return "placeholder"


def _route_clarification(state: AdvisorState) -> str:
    if state["clarification"].get("questions"):
        return "interrupt"
    return "continue"


def build_graph(
    *,
    settings: ApplicationSettings | None = None,
    checkpointer: Any | None = None,
    llm: Any | None = None,
    qdrant_client: Any | None = None,
) -> CompiledStateGraph:
    """Compile the graph without making network requests.

    Dependencies are lazy or injectable. An in-memory checkpointer is supplied
    by default so the returned graph supports interrupt/resume immediately.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    runtime = AdvisorRuntime(
        settings=settings or ApplicationSettings(),
        llm=llm,
        qdrant_client=qdrant_client,
    )
    builder = StateGraph(AdvisorState)
    builder.add_node("prepare_turn", prepare_turn_node)
    builder.add_node(
        "detect_intent", partial(detect_intent_node, advisor_runtime=runtime)
    )
    builder.add_node(
        "extract_need", partial(extract_need_profile_node, advisor_runtime=runtime)
    )
    builder.add_node(
        "generate_clarification",
        partial(generate_clarification_node, advisor_runtime=runtime),
    )
    builder.add_node(
        "collect_clarification",
        partial(collect_clarification_node, advisor_runtime=runtime),
    )
    builder.add_node(
        "build_filter", partial(build_filter_node, advisor_runtime=runtime)
    )
    builder.add_node(
        "retrieve_candidates",
        partial(retrieve_candidates_node, advisor_runtime=runtime),
    )
    builder.add_node(
        "rank_candidates", partial(rank_candidates_node, advisor_runtime=runtime)
    )
    builder.add_node(
        "compose_response", partial(compose_response_node, advisor_runtime=runtime)
    )
    builder.add_node("placeholder_response", placeholder_response_node)

    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "detect_intent")
    builder.add_conditional_edges(
        "detect_intent",
        _route_intent,
        {"refrigerator": "extract_need", "placeholder": "placeholder_response"},
    )
    builder.add_edge("extract_need", "generate_clarification")
    builder.add_conditional_edges(
        "generate_clarification",
        _route_clarification,
        {"interrupt": "collect_clarification", "continue": "build_filter"},
    )
    builder.add_edge("collect_clarification", "build_filter")
    builder.add_edge("build_filter", "retrieve_candidates")
    builder.add_edge("retrieve_candidates", "rank_candidates")
    builder.add_edge("rank_candidates", "compose_response")
    builder.add_edge("compose_response", END)
    builder.add_edge("placeholder_response", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
