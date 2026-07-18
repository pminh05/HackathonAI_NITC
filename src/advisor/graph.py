"""LangGraph builder for the multi-category product advisor."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from advisor.categories.registry import CategoryRegistry, build_default_registry
from advisor.nodes import (
    AdvisorRuntime,
    analyze_turn_node,
    build_filter_node,
    collect_clarification_node,
    compose_response_node,
    conversation_response_node,
    extract_need_profile_node,
    generate_clarification_node,
    plan_execution_node,
    placeholder_response_node,
    prepare_turn_node,
    rank_candidates_node,
    retrieve_candidates_node,
)
from advisor.schemas import ApplicationSettings, ExecutionMode, TurnAction
from advisor.state import AdvisorState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def _route_turn(state: AdvisorState) -> str:
    category = state.get("routing", {}).get("category")
    if not category or not state.get("routing", {}).get("supported"):
        return "placeholder"
    analysis = state.get("conversation", {}).get("analysis", {})
    action = analysis.get("action", TurnAction.DISCOVER.value)
    if action == TurnAction.CONVERSATION.value:
        return "conversation"
    context = state.get("category_contexts", {}).get(category, {})
    recommendation = context.get("recommendation_context", {})
    has_recommendations = bool(recommendation.get("last_presented_ids"))
    if action in {
        TurnAction.PRODUCT_DETAIL.value,
        TurnAction.COMPARE.value,
        TurnAction.EXPLAIN.value,
        TurnAction.MORE_OPTIONS.value,
        TurnAction.SWITCH_CATEGORY.value,
    } and has_recommendations:
        return "plan"
    return "extract"


def _route_clarification(state: AdvisorState) -> str:
    if state["clarification"].get("questions"):
        return "interrupt"
    return "continue"


def _route_execution(state: AdvisorState) -> str:
    mode = state.get("conversation", {}).get("execution_mode")
    if mode == ExecutionMode.REUSE.value:
        return "reuse"
    if mode == ExecutionMode.RERANK.value:
        return "rerank"
    return "retrieve"


def build_graph(
    *,
    settings: ApplicationSettings | None = None,
    checkpointer: Any | None = None,
    llm: Any | None = None,
    qdrant_client: Any | None = None,
    category_registry: CategoryRegistry | None = None,
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
        category_registry=category_registry or build_default_registry(),
    )
    builder = StateGraph(AdvisorState)
    builder.add_node("prepare_turn", prepare_turn_node)
    builder.add_node(
        "analyze_turn", partial(analyze_turn_node, advisor_runtime=runtime)
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
    builder.add_node(
        "plan_execution", partial(plan_execution_node, advisor_runtime=runtime)
    )
    builder.add_node("conversation_response", conversation_response_node)
    builder.add_node("placeholder_response", placeholder_response_node)

    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "analyze_turn")
    builder.add_conditional_edges(
        "analyze_turn",
        _route_turn,
        {
            "extract": "extract_need",
            "plan": "plan_execution",
            "conversation": "conversation_response",
            "placeholder": "placeholder_response",
        },
    )
    builder.add_edge("extract_need", "generate_clarification")
    builder.add_conditional_edges(
        "generate_clarification",
        _route_clarification,
        {"interrupt": "collect_clarification", "continue": "plan_execution"},
    )
    builder.add_edge("collect_clarification", "generate_clarification")
    builder.add_conditional_edges(
        "plan_execution",
        _route_execution,
        {
            "reuse": "compose_response",
            "rerank": "rank_candidates",
            "retrieve": "build_filter",
        },
    )
    builder.add_edge("build_filter", "retrieve_candidates")
    builder.add_edge("retrieve_candidates", "rank_candidates")
    builder.add_edge("rank_candidates", "compose_response")
    builder.add_edge("compose_response", END)
    builder.add_edge("conversation_response", END)
    builder.add_edge("placeholder_response", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
