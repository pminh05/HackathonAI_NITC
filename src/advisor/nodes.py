"""Conversation-aware LangGraph nodes for the product advisor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from advisor.categories.base import CategorySpec, deserialize_filter, serialize_filter
from advisor.categories.registry import CategoryRegistry, build_default_registry
from advisor.retrieval.qdrant import (
    AdvisorConfigurationError,
    create_qdrant_client,
    find_missing_indexes,
    query_products,
)
from advisor.schemas import (
    ApplicationSettings,
    CategoryTransition,
    ClarificationQuestion,
    ClarificationSubmission,
    ExecutionMode,
    IntentLabel,
    ProfilePatch,
    RankingResult,
    TurnAction,
    TurnAnalysisResult,
)
from advisor.state import AdvisorState, NodeUpdate

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI


CATEGORY_SLUGS = {
    IntentLabel.REFRIGERATOR: "refrigerator",
    IntentLabel.AIR_CONDITIONER: "air_conditioner",
    IntentLabel.WASHING_MACHINE: "washing_machine",
    IntentLabel.DRYER: "dryer",
    IntentLabel.DISHWASHER: "dishwasher",
    IntentLabel.COOLER_FREEZER: "cooler_freezer",
    IntentLabel.WATER_HEATER: "water_heater",
    IntentLabel.KARAOKE_MICROPHONE: "karaoke_microphone",
    IntentLabel.PHONE_RECORDING_MICROPHONE: "phone_recording_microphone",
    IntentLabel.SMARTWATCH: "smartwatch",
    IntentLabel.DESKTOP: "desktop",
    IntentLabel.MONITOR: "monitor",
    IntentLabel.PRINTER: "printer",
    IntentLabel.TABLET: "tablet",
}
SLUG_LABELS = {slug: label for label, slug in CATEGORY_SLUGS.items()}


TURN_ANALYSIS_PROMPT = """Bạn điều phối một cuộc hội thoại tư vấn sản phẩm nhiều lượt.

Tin nhắn hiện tại:
{message}

Ngành hàng đang hoạt động: {active_category}
Các sản phẩm đã được nhắc, ID và vị trí của chúng:
{recommendation_aliases}

Các câu đang chờ làm rõ:
{pending_questions}

Lịch sử gần nhất:
{conversation_history}

Hãy xác định category, category_transition, action, scope và các product_id được
tham chiếu. Quy tắc:
- Khi đã có ngành hàng đang hoạt động, mặc định category_transition=inherit.
- Follow-up rút gọn như “mẫu nào tiết kiệm điện nhất?”, “mẫu đầu”, “còn mẫu
  khác?” phải kế thừa ngành hiện tại, không được gắn thành một chủ đề mới.
- Chỉ dùng switch khi người dùng chủ động và rõ ràng chuyển sang ngành khác;
  switch_evidence phải là đoạn trích nguyên văn có trong tin nhắn.
- Chỉ trả referenced_product_ids có trong danh sách sản phẩm được cung cấp.
- product_detail, compare và explain đều chỉ nói về context đã có; more_options
  là xin lựa chọn mới; refine_needs là thêm/đính chính/xóa nhu cầu.
- has_profile_update=true khi tin nhắn thay đổi nhu cầu hoặc trả lời câu hỏi làm rõ.
- conversation chỉ dùng cho chào hỏi/cảm ơn; khi đó có thể trả direct_reply ngắn,
  tự nhiên. Không đưa lời tư vấn sản phẩm vào direct_reply.
"""


def create_gemini_chat_model(settings: ApplicationSettings) -> ChatGoogleGenerativeAI:
    """Create the configured Gemini model with the lowest thinking level."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before creating Gemini.") from exc

    if not settings.google_api_key:
        raise AdvisorConfigurationError("GOOGLE_API_KEY is required")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        thinking_level="minimal",
        include_thoughts=False,
        temperature=0,
        max_retries=2,
    )


@dataclass
class AdvisorRuntime:
    """Lazily initialized dependencies captured by compiled graph nodes."""

    settings: ApplicationSettings = field(default_factory=ApplicationSettings)
    llm: Any | None = None
    qdrant_client: Any | None = None
    category_registry: CategoryRegistry = field(default_factory=build_default_registry)
    _qdrant_checked_categories: set[str] = field(default_factory=set)

    def get_llm(self) -> Any:
        if self.llm is None:
            self.llm = create_gemini_chat_model(self.settings)
        return self.llm

    def get_qdrant(self) -> Any:
        if self.qdrant_client is None:
            self.qdrant_client = create_qdrant_client(self.settings)
        return self.qdrant_client

    def structured(self, schema: type[Any]) -> Any:
        return self.get_llm().with_structured_output(schema, method="json_schema")

    def get_category(self, name: str) -> CategorySpec:
        return self.category_registry.get_spec(name)

    def assert_qdrant_ready(self, category: str) -> None:
        if category in self._qdrant_checked_categories:
            return
        spec = self.get_category(category)
        config = spec.config
        missing = find_missing_indexes(
            self.get_qdrant(), config["collection"], config["payload_indexes"]
        )
        if missing:
            fields = ", ".join(sorted(missing))
            setup_hint = (
                f" Run `{spec.setup_indexes_command}`."
                if spec.setup_indexes_command
                else ""
            )
            raise AdvisorConfigurationError(
                f"Qdrant category {category!r} is missing required payload "
                f"indexes: {fields}.{setup_hint}"
            )
        self._qdrant_checked_categories.add(category)


def _latest_user_text(state: AdvisorState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    raise ValueError("Advisor graph requires at least one HumanMessage")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
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
    return str(content)


def _conversation_history(state: AdvisorState, limit: int = 12) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in state.get("messages", [])[-limit:]:
        message_type = getattr(message, "type", "")
        if message_type not in {"human", "ai"}:
            continue
        history.append(
            {
                "role": "user" if message_type == "human" else "assistant",
                "content": _message_text(message),
            }
        )
    return history


def _merge_unique(existing: list[Any], incoming: list[Any]) -> list[Any]:
    merged = list(existing)
    for item in incoming:
        if item not in merged:
            merged.append(item)
    return merged


def merge_need_profile(
    current: dict[str, Any], updates: dict[str, Any], *, category: str | None = None
) -> dict[str, Any]:
    """Backward-compatible merge used by legacy checkpoints and form options."""
    merged = dict(current)
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, list):
            merged[key] = _merge_unique(merged.get(key, []), value or [])
        elif isinstance(value, dict):
            nested = dict(merged.get(key) or {})
            for nested_key, nested_value in (value or {}).items():
                if nested_value in (None, [], {}):
                    continue
                if isinstance(nested_value, list):
                    nested[nested_key] = _merge_unique(
                        nested.get(nested_key, []), nested_value
                    )
                else:
                    nested[nested_key] = nested_value
            merged[key] = nested
        else:
            merged[key] = value
    merged.setdefault("category", category or current.get("category") or "refrigerator")
    return merged


def _fresh_category_context() -> dict[str, Any]:
    return {
        "profile": {},
        "profile_revision": 0,
        "clarification": {
            "status": "not_checked",
            "round": 0,
            "questions": [],
            "resolved_fields": [],
        },
        "recommendation_context": {
            "candidate_pool": [],
            "product_snapshots": {},
            "ranking_by_id": {},
            "last_presented_ids": [],
            "presented_ids": [],
            "presentations": [],
        },
    }


def _legacy_recommendation_context(state: AdvisorState) -> dict[str, Any]:
    retrieval = state.get("retrieval") or {}
    ranking = state.get("ranking") or {}
    candidates = list(retrieval.get("candidates") or [])
    selected = list(ranking.get("selected_products") or [])
    selected_ids = [item["product_id"] for item in selected if item.get("product_id")]
    candidates_by_id = {item["product_id"]: item for item in candidates}
    return {
        "candidate_pool": candidates,
        "product_snapshots": {
            product_id: candidates_by_id[product_id]
            for product_id in selected_ids
            if product_id in candidates_by_id
        },
        "ranking_by_id": {
            item["product_id"]: item for item in selected if item.get("product_id")
        },
        "last_presented_ids": selected_ids,
        "presented_ids": selected_ids,
        "presentations": [selected_ids] if selected_ids else [],
        "discovery_query": retrieval.get("query_text"),
    }


def _get_category_context(state: AdvisorState, category: str) -> dict[str, Any]:
    contexts = state.get("category_contexts") or {}
    if category in contexts:
        context = dict(contexts[category])
        context.setdefault("profile", {})
        context.setdefault("profile_revision", 0)
        context.setdefault("clarification", _fresh_category_context()["clarification"])
        context.setdefault(
            "recommendation_context",
            _fresh_category_context()["recommendation_context"],
        )
        return context
    if category == "refrigerator" and any(
        state.get(key) for key in ("need_profile", "retrieval", "ranking", "clarification")
    ):
        return {
            "profile": dict(state.get("need_profile") or {}),
            "profile_revision": 1 if state.get("need_profile") else 0,
            "clarification": dict(
                state.get("clarification")
                or _fresh_category_context()["clarification"]
            ),
            "recommendation_context": _legacy_recommendation_context(state),
        }
    return _fresh_category_context()


def _put_category_context(
    state: AdvisorState, category: str, context: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    contexts = dict(state.get("category_contexts") or {})
    contexts[category] = context
    return contexts


def _active_category(state: AdvisorState) -> str | None:
    conversation = state.get("conversation") or {}
    active = conversation.get("active_category")
    if active:
        return str(active)
    routing = state.get("routing") or {}
    category = routing.get("category")
    return str(category) if category else None


def _recommendation_aliases(context: dict[str, Any]) -> list[dict[str, Any]]:
    recommendation = context.get("recommendation_context") or {}
    snapshots = recommendation.get("product_snapshots") or {}
    aliases: list[dict[str, Any]] = []
    presentations = list(recommendation.get("presentations") or [])[-3:]
    if not presentations and recommendation.get("last_presented_ids"):
        presentations = [recommendation["last_presented_ids"]]
    seen: set[str] = set()
    for batch_recency, batch in enumerate(reversed(presentations)):
        for position, product_id in enumerate(batch, start=1):
            if product_id in seen:
                continue
            seen.add(product_id)
            product = snapshots.get(product_id) or {}
            aliases.append(
                {
                    "batch_recency": batch_recency,
                    "position": position,
                    "product_id": product_id,
                    "name": product.get("name"),
                    "brand": product.get("brand"),
                }
            )
    return aliases


def prepare_turn_node(state: AdvisorState) -> NodeUpdate:
    query = _latest_user_text(state)
    previous = state.get("conversation") or {}
    active = _active_category(state)
    turn_id = int(previous.get("turn_id") or 0) + 1
    update: NodeUpdate = {
        "conversation": {
            **previous,
            "active_category": active,
            "turn_id": turn_id,
            "analysis": {},
            "execution_mode": None,
        },
        "routing": {
            **(state.get("routing") or {}),
            "category": active,
        },
        "retrieval": {},
        "ranking": {},
        "response": {},
        "control": {
            "stage": "prepared",
            "current_user_input": query,
            "input_kind": "message",
            "profile_changed_paths": [],
        },
    }
    if active:
        context = _get_category_context(state, active)
        update.update(
            {
                "category_contexts": _put_category_context(state, active, context),
                "need_profile": context.get("profile", {}),
                "clarification": context.get("clarification", {}),
            }
        )
    else:
        update["clarification"] = {"status": "not_checked", "round": 0}
    return update


def analyze_turn_node(state: AdvisorState, advisor_runtime: AdvisorRuntime) -> NodeUpdate:
    query = state["control"]["current_user_input"]
    current_active = _active_category(state)
    current_context = (
        _get_category_context(state, current_active)
        if current_active
        else _fresh_category_context()
    )
    prompt = TURN_ANALYSIS_PROMPT.format(
        message=query,
        active_category=current_active or "chưa có",
        recommendation_aliases=json.dumps(
            _recommendation_aliases(current_context), ensure_ascii=False
        ),
        pending_questions=json.dumps(
            current_context.get("clarification", {}).get("questions", []),
            ensure_ascii=False,
        ),
        conversation_history=json.dumps(
            _conversation_history(state), ensure_ascii=False
        ),
    )
    analysis = advisor_runtime.structured(TurnAnalysisResult).invoke(prompt)
    requested_category = CATEGORY_SLUGS.get(analysis.category)
    evidence = (analysis.switch_evidence or "").strip()
    valid_switch = bool(
        current_active
        and requested_category
        and requested_category != current_active
        and analysis.category_transition is CategoryTransition.SWITCH
        and evidence
        and evidence.casefold() in query.casefold()
    )

    if current_active is None:
        active = requested_category
        transition = CategoryTransition.NEW
    elif valid_switch:
        active = requested_category
        transition = CategoryTransition.SWITCH
    else:
        active = current_active
        transition = CategoryTransition.INHERIT

    contexts = dict(state.get("category_contexts") or {})
    context = _get_category_context(state, active) if active else _fresh_category_context()
    if active and analysis.action is TurnAction.RESTART_CATEGORY:
        context = _fresh_category_context()
    if active:
        contexts[active] = context

    recommendation = context.get("recommendation_context") or {}
    allowed_ids = set(recommendation.get("product_snapshots") or {})
    allowed_ids.update(
        item.get("product_id")
        for item in recommendation.get("candidate_pool") or []
        if item.get("product_id")
    )
    referenced_ids = [
        product_id
        for product_id in analysis.referenced_product_ids
        if product_id in allowed_ids
    ]
    normalized_analysis = {
        **analysis.model_dump(mode="json"),
        "category_transition": transition.value,
        "referenced_product_ids": referenced_ids,
    }
    conversation = {
        **(state.get("conversation") or {}),
        "active_category": active,
        "analysis": normalized_analysis,
    }
    return {
        "conversation": conversation,
        "category_contexts": contexts,
        "routing": {
            "intent": analysis.category.value,
            "category": active,
            "transition": transition.value,
            "supported": bool(
                active
                and active in advisor_runtime.category_registry.all()
                and advisor_runtime.category_registry.get(active).implemented
            ),
        },
        "need_profile": context.get("profile", {}),
        "clarification": context.get("clarification", {}),
        "control": {**state["control"], "stage": "intent_detected"},
    }


def _path_get(profile: dict[str, Any], path: str) -> Any:
    value: Any = profile
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _path_set(profile: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = profile
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            nested = {}
            target[part] = nested
        target = nested
    target[parts[-1]] = value


def _path_clear(
    profile: dict[str, Any], path: str, list_patch_paths: frozenset[str]
) -> None:
    parts = path.split(".")
    target = profile
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            return
        target = nested
    if path in list_patch_paths:
        target[parts[-1]] = []
    else:
        target.pop(parts[-1], None)


def _validated_patch_paths(paths: Any, valid_paths: frozenset[str]) -> list[str]:
    return [str(path) for path in paths if str(path) in valid_paths]


def apply_profile_patch(
    current: dict[str, Any], patch: ProfilePatch, category_spec: CategorySpec | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Apply clear → set/replace → remove → add without additive corrections."""
    if category_spec is None:
        # Backward compatibility for direct callers during the refrigerator MVP.
        category_spec = build_default_registry().get_spec("refrigerator")
    valid_paths = category_spec.valid_patch_paths
    list_paths = category_spec.list_patch_paths
    profile = json.loads(json.dumps(current, ensure_ascii=False, default=str))
    changed_paths: list[str] = []

    for path in _validated_patch_paths(patch.clear, valid_paths):
        before = _path_get(profile, path)
        _path_clear(profile, path, list_paths)
        if before != _path_get(profile, path):
            changed_paths.append(path)

    for operation in (patch.set, patch.replace):
        for path in _validated_patch_paths(operation, valid_paths):
            value = operation[path]
            if path in list_paths and not isinstance(value, list):
                continue
            before = _path_get(profile, path)
            _path_set(profile, path, value)
            if before != value:
                changed_paths.append(path)

    for path in _validated_patch_paths(patch.remove, valid_paths):
        if path not in list_paths:
            continue
        before = list(_path_get(profile, path) or [])
        removed = patch.remove[path]
        value = [item for item in before if item not in removed]
        _path_set(profile, path, value)
        if before != value:
            changed_paths.append(path)

    for path in _validated_patch_paths(patch.add, valid_paths):
        if path not in list_paths:
            continue
        before = list(_path_get(profile, path) or [])
        value = _merge_unique(before, patch.add[path])
        _path_set(profile, path, value)
        if before != value:
            changed_paths.append(path)

    if patch.evidence:
        evidence = dict(profile.get("evidence") or {})
        evidence.update(patch.evidence)
        profile["evidence"] = evidence
    extras = {
        key: value
        for key, value in profile.items()
        if key in {"category", "custom_answers", "latest_query"}
    }
    try:
        validated = category_spec.profile_model.model_validate(profile).model_dump(
            exclude_none=True
        )
    except ValueError:
        # A malformed generic patch must not corrupt a previously valid profile or
        # trigger a retrying LLM call on the latency-sensitive path.
        return merge_need_profile({}, current, category=category_spec.name), []
    validated.update(extras)
    profile = merge_need_profile({}, validated, category=category_spec.name)
    return profile, list(dict.fromkeys(changed_paths))


def extract_need_profile_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    category = _active_category(state)
    if not category:
        return {"control": {**state["control"], "stage": "need_extracted"}}
    spec = advisor_runtime.get_category(category)
    context = _get_category_context(state, category)
    current = context.get("profile") or {}
    analysis = (state.get("conversation") or {}).get("analysis") or {}
    extraction = advisor_runtime.structured(ProfilePatch).invoke(
        spec.build_need_extraction_prompt(
            state["control"]["current_user_input"],
            current,
            turn_action=analysis.get("action", TurnAction.DISCOVER.value),
            pending_questions=(context.get("clarification") or {}).get(
                "questions", []
            ),
            conversation_history=_conversation_history(state),
        )
    )
    if isinstance(extraction, ProfilePatch):
        patch = extraction
        profile, changed_paths = apply_profile_patch(current, patch, spec)
    else:
        # Allows a pending legacy run/fake to complete during rolling deployment.
        updates = extraction.model_dump(exclude_none=True)
        profile = merge_need_profile(current, updates, category=category)
        changed_paths = [
            key for key, value in profile.items() if current.get(key) != value
        ]
    resolved = set(
        (context.get("clarification") or {}).get("resolved_fields", [])
    )
    for question_id, paths in spec.question_profile_paths.items():
        if paths.intersection(changed_paths):
            resolved.discard(question_id)
    profile["latest_query"] = state["control"]["current_user_input"]
    clarification = {
        **(context.get("clarification") or {}),
        "resolved_fields": sorted(resolved),
    }
    changed = bool(changed_paths)
    context = {
        **context,
        "profile": profile,
        "profile_revision": int(context.get("profile_revision") or 0)
        + (1 if changed else 0),
        "clarification": clarification,
    }
    return {
        "category_contexts": _put_category_context(state, category, context),
        "need_profile": profile,
        "clarification": clarification,
        "control": {
            **state["control"],
            "stage": "need_extracted",
            "profile_changed_paths": changed_paths,
        },
    }


def _public_catalog(
    question_catalog: dict[str, Any], question_ids: list[str]
) -> dict[str, Any]:
    return {
        question_id: {
            "question_type": question_catalog[question_id]["question_type"],
            "question": question_catalog[question_id]["question"],
            "options": [
                {"option_id": option["option_id"], "label": option["label"]}
                for option in question_catalog[question_id]["options"]
            ],
        }
        for question_id in question_ids
    }


def generate_clarification_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    category = _active_category(state)
    if not category:
        return {"control": {**state["control"], "stage": "needs_sufficient"}}
    spec = advisor_runtime.get_category(category)
    context = _get_category_context(state, category)
    profile = context.get("profile") or {}
    previous = context.get("clarification") or {}
    resolved = set(previous.get("resolved_fields") or [])
    missing = [
        field
        for field in spec.get_missing_profile_fields(profile, spec.config)
        if field not in resolved
    ]
    if not missing:
        clarification = {
            **previous,
            "status": "not_required",
            "missing_information": [],
            "questions": [],
            "resolved_fields": sorted(resolved),
        }
        context = {**context, "clarification": clarification}
        return {
            "category_contexts": _put_category_context(state, category, context),
            "clarification": clarification,
            "control": {**state["control"], "stage": "needs_sufficient"},
        }

    selected = missing[:3]
    catalog = spec.config["question_catalog"]
    questions: list[dict[str, Any]] = []
    for question_id in selected:
        options = [
            {"option_id": option["option_id"], "label": option["label"]}
            for option in catalog[question_id]["options"]
        ]
        questions.append(
            ClarificationQuestion(
                question_id=question_id,
                question_type=catalog[question_id]["question_type"],
                question=catalog[question_id]["question"],
                options=options,
            ).model_dump(mode="json")
        )
    round_number = int(previous.get("round") or 0) + 1
    question_text = " ".join(item["question"] for item in questions)
    message = (
        "Để gợi ý sát hơn, mình cần thêm một chút thông tin. "
        f"{question_text} Hãy chọn lần lượt các đáp án phù hợp nhất."
    )
    clarification = {
        **previous,
        "status": "pending",
        "missing_information": missing,
        "questions": questions,
        "round": round_number,
        "message": message,
        "resolved_fields": sorted(resolved),
    }
    context = {**context, "clarification": clarification}
    turn_id = (state.get("conversation") or {}).get("turn_id", 0)
    return {
        "messages": [
            AIMessage(
                content=message,
                id=f"clarification-{turn_id}-{round_number}",
            )
        ],
        "category_contexts": _put_category_context(state, category, context),
        "clarification": clarification,
        "control": {**state["control"], "stage": "clarification_ready"},
    }


def _find_option(
    config: dict[str, Any], question_id: str, option_id: str
) -> dict[str, Any]:
    try:
        question = config["question_catalog"][question_id]
    except KeyError as exc:
        raise ValueError(f"Unknown clarification question: {question_id}") from exc
    for option in question["options"]:
        if option["option_id"] == option_id:
            return option
    raise ValueError(f"Unknown option {option_id!r} for question {question_id!r}")


def _interpret_custom_answer(
    advisor_runtime: AdvisorRuntime,
    spec: CategorySpec,
    profile: dict[str, Any],
    question_id: str,
    custom_answer: str,
) -> BaseModel:
    catalog_question = spec.config["question_catalog"][
        question_id
    ]
    return advisor_runtime.structured(spec.custom_answer_model).invoke(
        spec.build_custom_answer_prompt(
            {
                "question": catalog_question["question"],
                "question_id": question_id,
                "available_options": [
                    {"option_id": item["option_id"], "label": item["label"]}
                    for item in catalog_question["options"]
                    if item["option_id"] != "other"
                ],
                "custom_answer": custom_answer,
                "current_need_profile": profile,
            }
        )
    )


def collect_clarification_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    category = _active_category(state)
    if not category:
        raise ValueError("Clarification requires an active category")
    spec = advisor_runtime.get_category(category)
    context = _get_category_context(state, category)
    clarification = context.get("clarification") or state["clarification"]
    payload = {
        "type": "clarification_required",
        "category": category,
        "message": clarification.get("message", ""),
        "questions": clarification["questions"],
    }
    raw_submission = interrupt(payload)
    submission = ClarificationSubmission.model_validate(raw_submission)
    expected = {item["question_id"] for item in clarification["questions"]}
    received = {item.question_id for item in submission.answers}
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(
            "Clarification answers must match the full form; "
            f"missing={missing}, unexpected={unexpected}"
        )

    profile = dict(context.get("profile") or state.get("need_profile") or {})
    custom_evidence = dict(profile.get("custom_answers") or {})
    resolved = set(clarification.get("resolved_fields") or [])
    changed_paths: list[str] = []
    answer_labels: list[str] = []
    for answer in submission.answers:
        option = _find_option(
            spec.config,
            answer.question_id,
            answer.option_id,
        )
        answer_labels.append(f"{answer.question_id}: {option.get('label', answer.option_id)}")
        if answer.option_id != "other":
            updates = option.get("profile_updates") or {}
            before = json.dumps(profile, ensure_ascii=False, sort_keys=True, default=str)
            profile = merge_need_profile(profile, updates, category=category)
            if before != json.dumps(
                profile, ensure_ascii=False, sort_keys=True, default=str
            ):
                changed_paths.extend(spec.question_profile_paths.get(answer.question_id, set()))
            continue

        custom_answer = (answer.custom_answer or "").strip()
        interpretation = _interpret_custom_answer(
            advisor_runtime, spec, profile, answer.question_id, custom_answer
        )
        interpreted_updates = interpretation.model_dump(
            exclude={"interpretation_status", "raw_answer", "confidence"},
            exclude_none=True,
        )
        before = json.dumps(profile, ensure_ascii=False, sort_keys=True, default=str)
        profile = merge_need_profile(profile, interpreted_updates, category=category)
        if before != json.dumps(profile, ensure_ascii=False, sort_keys=True, default=str):
            changed_paths.extend(spec.question_profile_paths.get(answer.question_id, set()))
        custom_evidence[answer.question_id] = {
            "raw_answer": custom_answer,
            "status": interpretation.interpretation_status,
            "confidence": interpretation.confidence,
        }
    if custom_evidence:
        profile["custom_answers"] = custom_evidence

    previous_answers = list(clarification.get("answers") or [])
    previous_answers.extend(submission.model_dump(mode="json")["answers"])
    resolved.update(received)
    clarification = {
        **clarification,
        "status": "answer_received",
        "answers": previous_answers,
        "resolved_fields": sorted(resolved),
    }
    context = {
        **context,
        "profile": profile,
        "profile_revision": int(context.get("profile_revision") or 0)
        + (1 if changed_paths else 0),
        "clarification": clarification,
    }
    return {
        "messages": [HumanMessage(content="; ".join(answer_labels))],
        "category_contexts": _put_category_context(state, category, context),
        "need_profile": profile,
        "clarification": clarification,
        "control": {
            **state["control"],
            "stage": "clarification_completed",
            "input_kind": "structured_clarification",
            "profile_changed_paths": list(dict.fromkeys(changed_paths)),
        },
    }


def _profile_signature(profile: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in profile.items()
        if key not in {"latest_query", "evidence", "custom_answers"}
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def plan_execution_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    category = _active_category(state)
    if not category:
        return {"control": {**state["control"], "stage": "execution_planned"}}
    context = _get_category_context(state, category)
    spec = advisor_runtime.get_category(category)
    recommendation = dict(context.get("recommendation_context") or {})
    analysis = (state.get("conversation") or {}).get("analysis") or {}
    action = analysis.get("action", TurnAction.DISCOVER.value)
    changed_paths = set(state.get("control", {}).get("profile_changed_paths") or [])
    candidate_pool = list(recommendation.get("candidate_pool") or [])
    last_ids = list(recommendation.get("last_presented_ids") or [])
    mode = ExecutionMode.RETRIEVE
    retrieval: dict[str, Any] = {}
    ranking: dict[str, Any] = {}

    if action == TurnAction.MORE_OPTIONS.value and candidate_pool:
        presented = set(recommendation.get("presented_ids") or [])
        remaining = [
            candidate
            for candidate in candidate_pool
            if candidate.get("product_id") not in presented
        ]
        if remaining:
            mode = ExecutionMode.RERANK
            retrieval = {"candidates": remaining, "candidate_count": len(remaining)}
        else:
            mode = ExecutionMode.RETRIEVE
            retrieval = {"exclude_product_ids": sorted(presented)}
    elif action in {
        TurnAction.PRODUCT_DETAIL.value,
        TurnAction.COMPARE.value,
        TurnAction.EXPLAIN.value,
    } and last_ids:
        mode = ExecutionMode.REUSE
    elif not candidate_pool:
        mode = ExecutionMode.RETRIEVE
    elif changed_paths & spec.hard_retrieval_paths:
        mode = ExecutionMode.RETRIEVE
    elif changed_paths:
        if analysis.get("scope") == "current_recommendations":
            mode = ExecutionMode.RERANK
            retrieval = {
                "candidates": candidate_pool,
                "candidate_count": len(candidate_pool),
            }
        else:
            mode = ExecutionMode.RETRIEVE
    else:
        mode = ExecutionMode.REUSE

    if mode is ExecutionMode.REUSE:
        snapshots = dict(recommendation.get("product_snapshots") or {})
        snapshots.update(
            {
                item["product_id"]: item
                for item in candidate_pool
                if item.get("product_id")
            }
        )
        referenced_ids = analysis.get("referenced_product_ids") or last_ids
        candidates = [snapshots[item] for item in referenced_ids if item in snapshots]
        stored_ranking = recommendation.get("ranking_by_id") or {}
        selected = [
            stored_ranking.get(
                item,
                {
                    "product_id": item,
                    "reason": "Sản phẩm đang được nhắc trong hội thoại.",
                    "trade_off": "Xem lại theo ưu tiên hiện tại của khách hàng.",
                },
            )
            for item in referenced_ids
            if item in snapshots
        ]
        retrieval = {"candidates": candidates, "candidate_count": len(candidates)}
        ranking = {"selected_products": selected}

    conversation = {
        **(state.get("conversation") or {}),
        "execution_mode": mode.value,
    }
    return {
        "conversation": conversation,
        "retrieval": retrieval,
        "ranking": ranking,
        "control": {**state["control"], "stage": "execution_planned"},
    }


def build_filter_node(state: AdvisorState, advisor_runtime: AdvisorRuntime) -> NodeUpdate:
    category = _active_category(state)
    if not category:
        raise ValueError("Filter construction requires an active category")
    spec = advisor_runtime.get_category(category)
    config = spec.config
    query_filter = spec.build_filter(state["need_profile"], config["payload_fields"])
    retrieval = {
        **(state.get("retrieval") or {}),
        "collection": config["collection"],
        "embedding_model": config["embedding_model"],
        "filter": serialize_filter(query_filter),
    }
    return {
        "retrieval": retrieval,
        "control": {**state["control"], "stage": "filter_built"},
    }


def _base_search_query(state: AdvisorState) -> str:
    category = _active_category(state)
    context = _get_category_context(state, category) if category else {}
    recommendation = context.get("recommendation_context") or {}
    analysis = (state.get("conversation") or {}).get("analysis") or {}
    if analysis.get("action") == TurnAction.MORE_OPTIONS.value:
        base_query = recommendation.get("discovery_query") or state["control"][
            "current_user_input"
        ]
    else:
        base_query = state["control"]["current_user_input"]
    return str(base_query)


def retrieve_candidates_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    category = _active_category(state)
    if not category:
        raise ValueError("Retrieval requires an active category")
    spec = advisor_runtime.get_category(category)
    advisor_runtime.assert_qdrant_ready(category)
    config = spec.config
    query_text = spec.build_search_text(
        state["need_profile"], _base_search_query(state)
    )
    exclude_ids = set((state.get("retrieval") or {}).get("exclude_product_ids") or [])
    limit = int(config["retrieval_limit"]) * (2 if exclude_ids else 1)
    points = query_products(
        advisor_runtime.get_qdrant(),
        collection=config["collection"],
        embedding_model=config["embedding_model"],
        query_text=query_text,
        query_filter=deserialize_filter(state["retrieval"].get("filter")),
        limit=limit,
        timeout=advisor_runtime.settings.qdrant_timeout_seconds,
    )
    candidates = [
        candidate
        for candidate in (spec.normalize_candidate(point) for point in points)
        if candidate["product_id"] not in exclude_ids
    ][: int(config["retrieval_limit"])]
    return {
        "retrieval": {
            **state["retrieval"],
            "query_text": query_text,
            "candidates": candidates,
            "candidate_count": len(candidates),
        },
        "control": {**state["control"], "stage": "retrieval_completed"},
    }


def rank_candidates_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    candidates = state["retrieval"].get("candidates", [])
    if not candidates:
        return {
            "ranking": {"selected_products": []},
            "control": {**state["control"], "stage": "ranking_completed"},
        }
    category = _active_category(state)
    if not category:
        raise ValueError("Ranking requires an active category")
    spec = advisor_runtime.get_category(category)
    prompt = spec.build_ranking_prompt(
        {
            "need_profile": state["need_profile"],
            "hard_constraints": state["need_profile"].get("hard_constraints", {}),
            "candidates": candidates,
        }
    )
    allowed_ids = {candidate["product_id"] for candidate in candidates}
    result: RankingResult | None = None
    for attempt in range(2):
        result = advisor_runtime.structured(RankingResult).invoke(prompt)
        selected_ids = {item.product_id for item in result.selected_products}
        if selected_ids and selected_ids <= allowed_ids:
            break
        if attempt == 0:
            prompt += (
                "\nKết quả trước chứa product_id không hợp lệ. Hãy làm lại và chỉ dùng "
                f"một trong các ID sau: {sorted(allowed_ids)}"
            )
    assert result is not None
    selected_ids = {item.product_id for item in result.selected_products}
    if not selected_ids or not selected_ids <= allowed_ids:
        raise ValueError("Gemini selected product IDs outside the retrieved candidates")
    return {
        "ranking": {
            "selected_products": [
                item.model_dump(mode="json") for item in result.selected_products
            ]
        },
        "control": {**state["control"], "stage": "ranking_completed"},
    }


def _finalize_response(state: AdvisorState, answer: str) -> NodeUpdate:
    category = _active_category(state)
    if not category:
        return {
            "messages": [AIMessage(content=answer)],
            "response": {"answer": answer},
            "control": {**state["control"], "stage": "completed"},
        }
    context = _get_category_context(state, category)
    recommendation = dict(context.get("recommendation_context") or {})
    candidates = list((state.get("retrieval") or {}).get("candidates") or [])
    candidates_by_id = {
        item["product_id"]: item for item in candidates if item.get("product_id")
    }
    selected = list((state.get("ranking") or {}).get("selected_products") or [])
    selected_public: list[dict[str, Any]] = []
    for item in selected:
        product = candidates_by_id.get(item.get("product_id"), {})
        selected_public.append({**product, **item})

    mode = (state.get("conversation") or {}).get("execution_mode")
    signature = _profile_signature(state.get("need_profile") or {})
    if mode == ExecutionMode.RETRIEVE.value:
        if recommendation.get("retrieval_signature") != signature:
            recommendation["presented_ids"] = []
        recommendation["candidate_pool"] = candidates
        recommendation["retrieval_signature"] = signature
        recommendation["discovery_query"] = state["control"]["current_user_input"]

    snapshots = dict(recommendation.get("product_snapshots") or {})
    ranking_by_id = dict(recommendation.get("ranking_by_id") or {})
    for item in selected_public:
        product_id = item.get("product_id")
        if not product_id:
            continue
        snapshots[product_id] = candidates_by_id.get(product_id, item)
        ranking_by_id[product_id] = {
            key: item.get(key) for key in ("product_id", "reason", "trade_off")
        }
    recommendation["product_snapshots"] = dict(list(snapshots.items())[-12:])
    recommendation["ranking_by_id"] = ranking_by_id

    analysis = (state.get("conversation") or {}).get("analysis") or {}
    action = analysis.get("action")
    selected_ids = [item["product_id"] for item in selected_public if item.get("product_id")]
    if action in {
        TurnAction.DISCOVER.value,
        TurnAction.REFINE_NEEDS.value,
        TurnAction.MORE_OPTIONS.value,
        TurnAction.SWITCH_CATEGORY.value,
        TurnAction.RESTART_CATEGORY.value,
    } and selected_ids:
        recommendation["last_presented_ids"] = selected_ids
        recommendation["presented_ids"] = _merge_unique(
            recommendation.get("presented_ids") or [], selected_ids
        )
        presentations = list(recommendation.get("presentations") or [])
        presentations.append(selected_ids)
        recommendation["presentations"] = presentations[-3:]
    recommendation["last_answer"] = answer
    context = {
        **context,
        "profile": state.get("need_profile") or context.get("profile", {}),
        "recommendation_context": recommendation,
    }
    return {
        "messages": [AIMessage(content=answer)],
        "category_contexts": _put_category_context(state, category, context),
        "retrieval": state.get("retrieval") or {},
        "ranking": {"selected_products": selected_public},
        "response": {"answer": answer},
        "control": {**state["control"], "stage": "completed"},
    }


def compose_response_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    candidates = state["retrieval"].get("candidates", [])
    selected = state.get("ranking", {}).get("selected_products", [])
    category = _active_category(state)
    if not category:
        raise ValueError("Response composition requires an active category")
    spec = advisor_runtime.get_category(category)
    if not candidates:
        return _finalize_response(state, spec.no_match_answer(state["need_profile"]))
    candidates_by_id = {item["product_id"]: item for item in candidates}
    grounded_selection = [
        {**item, "product_data": candidates_by_id[item["product_id"]]}
        for item in selected
        if item.get("product_id") in candidates_by_id
    ]
    analysis = (state.get("conversation") or {}).get("analysis") or {}
    prompt = spec.build_response_prompt(
        {
            "user_query": state["control"]["current_user_input"],
            "turn_action": analysis.get("action", TurnAction.DISCOVER.value),
            "conversation_history": _conversation_history(state),
            "need_profile": state["need_profile"],
            "selected_products": grounded_selection,
        }
    )
    llm_message = advisor_runtime.get_llm().invoke(prompt)
    answer = _message_text(llm_message).strip()
    if not answer:
        raise ValueError("Gemini returned an empty advisory response")
    return _finalize_response(state, answer)


def conversation_response_node(state: AdvisorState) -> NodeUpdate:
    analysis = (state.get("conversation") or {}).get("analysis") or {}
    answer = (analysis.get("direct_reply") or "Mình đang ở đây. Bạn muốn tìm hiểu thêm điều gì?").strip()
    return _finalize_response(state, answer)


def placeholder_response_node(state: AdvisorState) -> NodeUpdate:
    active = _active_category(state)
    label = SLUG_LABELS.get(active) if active else None
    if label:
        answer = (
            "Bản hiện tại hỗ trợ tư vấn tủ lạnh, máy lạnh, máy giặt, máy sấy "
            "quần áo, máy rửa chén, tủ mát, tủ đông và máy nước nóng; "
            f"ngành hàng {label.value} chưa được bật."
        )
    else:
        answer = "Mình chưa nhận ra ngành hàng bạn cần. Bạn đang muốn tìm loại sản phẩm nào?"
    return _finalize_response(state, answer)
