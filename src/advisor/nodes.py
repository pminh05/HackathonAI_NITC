"""LangGraph nodes for the one-round product-advisor MVP."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langgraph.types import interrupt

from advisor.categories.refrigerator import get_missing_profile_fields, load_config
from advisor.categories.refrigerator.filter_builder import (
    build_filter,
    deserialize_filter,
    serialize_filter,
)
from advisor.categories.refrigerator.prompts import (
    build_advisory_prompt,
    build_clarification_prompt,
    build_custom_answer_prompt,
    build_need_extraction_prompt,
    build_response_prompt,
)
from advisor.categories.refrigerator.setup_indexes import find_missing_indexes
from advisor.retrieval.qdrant import (
    AdvisorConfigurationError,
    create_qdrant_client,
    normalize_candidate,
    query_products,
)
from advisor.schemas import (
    ApplicationSettings,
    ClarificationDecision,
    ClarificationQuestion,
    ClarificationSubmission,
    CustomAnswerInterpretation,
    IntentLabel,
    IntentResult,
    RankingResult,
    RefrigeratorNeedExtraction,
)
from advisor.state import AdvisorState, NodeUpdate

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI


ROUTING_PROMPT = """Phân loại tin nhắn vào đúng 1 nhãn:

Tủ Lạnh
Máy lạnh
Máy giặt
Máy sấy quần áo
Máy rửa chén
Tủ mát, tủ đông
Máy nước nóng
Micro karaoke
Micro thu âm điện thoại
Đồng hồ thông minh
Máy tính để bàn
Màn hình máy tính
Máy in
Máy tính bảng
Khác

Dùng "Khác" nếu tin nhắn không liên quan đến sản phẩm thuộc 14 ngành hàng trên.
Không giải thích.

Tin nhắn: {message}
"""


def create_gemini_chat_model(settings: ApplicationSettings) -> ChatGoogleGenerativeAI:
    """Create Gemini 2.5 Flash with thinking disabled for every graph call."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before creating Gemini.") from exc

    if not settings.google_api_key:
        raise AdvisorConfigurationError("GOOGLE_API_KEY is required")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        thinking_budget=0,
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
    refrigerator_config: dict[str, Any] = field(default_factory=load_config)
    _qdrant_checked: bool = False

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

    def assert_qdrant_ready(self) -> None:
        if self._qdrant_checked:
            return
        config = self.refrigerator_config
        missing = find_missing_indexes(
            self.get_qdrant(), config["collection"], config["payload_indexes"]
        )
        if missing:
            fields = ", ".join(sorted(missing))
            raise AdvisorConfigurationError(
                "Qdrant collection is missing required payload indexes: "
                f"{fields}. Run `python -m "
                "advisor.categories.refrigerator.setup_indexes --apply`."
            )
        self._qdrant_checked = True


def _latest_user_text(state: AdvisorState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    raise ValueError("Advisor graph requires at least one HumanMessage")


def _merge_unique(existing: list[Any], incoming: list[Any]) -> list[Any]:
    merged = list(existing)
    for item in incoming:
        if item not in merged:
            merged.append(item)
    return merged


def merge_need_profile(
    current: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    """Merge category profile updates without losing previously confirmed data."""
    merged = dict(current)
    for key, value in updates.items():
        if value is None:
            continue
        if key in {"usage_preferences", "soft_preferences", "implicit_needs"}:
            merged[key] = _merge_unique(merged.get(key, []), value or [])
        elif key in {"hard_constraints", "evidence"}:
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
    merged.setdefault("category", "refrigerator")
    merged.setdefault("usage_preferences", [])
    merged.setdefault("soft_preferences", [])
    merged.setdefault("implicit_needs", [])
    merged.setdefault("hard_constraints", {})
    return merged


def prepare_turn_node(state: AdvisorState) -> NodeUpdate:
    query = _latest_user_text(state)
    return {
        "routing": {},
        "clarification": {"status": "not_checked", "round": 0},
        "retrieval": {},
        "ranking": {},
        "response": {},
        "control": {"stage": "prepared", "current_user_input": query},
    }


def detect_intent_node(state: AdvisorState, advisor_runtime: AdvisorRuntime) -> NodeUpdate:
    query = state["control"]["current_user_input"]
    result = advisor_runtime.structured(IntentResult).invoke(
        ROUTING_PROMPT.format(message=query)
    )
    return {
        "routing": {
            "intent": result.label.value,
            "category": "refrigerator"
            if result.label is IntentLabel.REFRIGERATOR
            else None,
        },
        "control": {**state["control"], "stage": "intent_detected"},
    }


def extract_need_profile_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    query = state["control"]["current_user_input"]
    current = state.get("need_profile", {})
    extraction = advisor_runtime.structured(RefrigeratorNeedExtraction).invoke(
        build_need_extraction_prompt(query, current)
    )
    updates = extraction.model_dump(exclude_none=True)
    profile = merge_need_profile(current, updates)
    profile["latest_query"] = query
    return {
        "need_profile": profile,
        "control": {**state["control"], "stage": "need_extracted"},
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
    config = advisor_runtime.refrigerator_config
    profile = state["need_profile"]
    missing = get_missing_profile_fields(profile, config)
    if not missing:
        return {
            "clarification": {
                "status": "not_required",
                "missing_information": [],
                "questions": [],
                "round": 0,
            },
            "control": {**state["control"], "stage": "needs_sufficient"},
        }

    catalog = config["question_catalog"]
    public_catalog = _public_catalog(catalog, missing)
    try:
        decision = advisor_runtime.structured(ClarificationDecision).invoke(
            build_clarification_prompt(
                {
                    "user_query": state["control"]["current_user_input"],
                    "current_need_profile": profile,
                    "missing_information": missing,
                    "question_catalog": public_catalog,
                }
            )
        )
        selected = [item for item in decision.question_ids if item in missing]
    except Exception:
        selected = []

    # Rules are authoritative: append any missing core question the LLM omitted.
    for question_id in missing:
        if question_id not in selected and len(selected) < 3:
            selected.append(question_id)

    questions = [
        ClarificationQuestion(
            question_id=question_id,
            question_type=catalog[question_id]["question_type"],
            question=catalog[question_id]["question"],
            options=[
                {"option_id": option["option_id"], "label": option["label"]}
                for option in catalog[question_id]["options"]
            ],
        ).model_dump(mode="json")
        for question_id in selected
    ]
    return {
        "clarification": {
            "status": "pending",
            "missing_information": missing,
            "questions": questions,
            "round": 1,
        },
        "control": {**state["control"], "stage": "clarification_ready"},
    }


def _find_option(config: dict[str, Any], question_id: str, option_id: str) -> dict[str, Any]:
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
    profile: dict[str, Any],
    question_id: str,
    custom_answer: str,
) -> CustomAnswerInterpretation:
    catalog_question = advisor_runtime.refrigerator_config["question_catalog"][question_id]
    return advisor_runtime.structured(CustomAnswerInterpretation).invoke(
        build_custom_answer_prompt(
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
    clarification = state["clarification"]
    payload = {
        "type": "clarification_required",
        "category": "refrigerator",
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
            f"Clarification answers must match the form; missing={missing}, "
            f"unexpected={unexpected}"
        )

    profile = dict(state["need_profile"])
    custom_evidence = dict(profile.get("custom_answers") or {})
    for answer in submission.answers:
        option = _find_option(
            advisor_runtime.refrigerator_config, answer.question_id, answer.option_id
        )
        if answer.option_id != "other":
            profile = merge_need_profile(profile, option.get("profile_updates") or {})
            continue

        custom_answer = (answer.custom_answer or "").strip()
        interpretation = _interpret_custom_answer(
            advisor_runtime, profile, answer.question_id, custom_answer
        )
        interpreted_updates = interpretation.model_dump(
            exclude={"interpretation_status", "raw_answer", "confidence"},
            exclude_none=True,
        )
        profile = merge_need_profile(profile, interpreted_updates)
        custom_evidence[answer.question_id] = {
            "raw_answer": custom_answer,
            "status": interpretation.interpretation_status,
            "confidence": interpretation.confidence,
        }
    if custom_evidence:
        profile["custom_answers"] = custom_evidence

    return {
        "need_profile": profile,
        "clarification": {
            **clarification,
            "status": "completed",
            "answers": submission.model_dump(mode="json")["answers"],
        },
        "control": {**state["control"], "stage": "clarification_completed"},
    }


def build_filter_node(state: AdvisorState, advisor_runtime: AdvisorRuntime) -> NodeUpdate:
    config = advisor_runtime.refrigerator_config
    query_filter = build_filter(state["need_profile"], config["payload_fields"])
    retrieval = {
        "collection": config["collection"],
        "embedding_model": config["embedding_model"],
        "filter": serialize_filter(query_filter),
    }
    return {
        "retrieval": retrieval,
        "control": {**state["control"], "stage": "filter_built"},
    }


def _build_search_text(state: AdvisorState) -> str:
    profile = state["need_profile"]
    parts = [state["control"]["current_user_input"]]
    if profile.get("household_size"):
        parts.append(f"phù hợp gia đình {profile['household_size']} người")
    if profile.get("budget_max_vnd"):
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    parts.extend(profile.get("usage_preferences", []))
    parts.extend(profile.get("soft_preferences", []))
    parts.extend(profile.get("implicit_needs", []))
    hard = profile.get("hard_constraints") or {}
    if hard:
        parts.append(json.dumps(hard, ensure_ascii=False, default=str))
    return ". ".join(str(part) for part in parts if part)


def retrieve_candidates_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    advisor_runtime.assert_qdrant_ready()
    config = advisor_runtime.refrigerator_config
    query_text = _build_search_text(state)
    points = query_products(
        advisor_runtime.get_qdrant(),
        collection=config["collection"],
        embedding_model=config["embedding_model"],
        query_text=query_text,
        query_filter=deserialize_filter(state["retrieval"].get("filter")),
        limit=int(config["retrieval_limit"]),
        timeout=advisor_runtime.settings.qdrant_timeout_seconds,
    )
    candidates = [normalize_candidate(point) for point in points]
    return {
        "retrieval": {
            **state["retrieval"],
            "query_text": query_text,
            "candidates": candidates,
            "candidate_count": len(candidates),
        },
        "control": {**state["control"], "stage": "retrieval_completed"},
    }


def _no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    if profile.get("budget_max_vnd"):
        constraints.append(f"ngân sách tối đa {profile['budget_max_vnd']:,} đồng")
    if profile.get("household_size"):
        constraints.append(f"nhu cầu cho {profile['household_size']} người")
    constraint_text = ", ".join(constraints) or "các điều kiện hiện tại"
    return (
        f"Mình chưa tìm thấy tủ lạnh đáp ứng đầy đủ {constraint_text}. "
        "Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý bỏ yêu cầu nào của bạn."
    )


def rank_candidates_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    candidates = state["retrieval"].get("candidates", [])
    if not candidates:
        return {
            "ranking": {"selected_products": []},
            "control": {**state["control"], "stage": "ranking_completed"},
        }

    prompt = build_advisory_prompt(
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


def compose_response_node(
    state: AdvisorState, advisor_runtime: AdvisorRuntime
) -> NodeUpdate:
    candidates = state["retrieval"].get("candidates", [])
    selected = state.get("ranking", {}).get("selected_products", [])
    if not candidates:
        answer = _no_match_answer(state["need_profile"])
    else:
        candidates_by_id = {item["product_id"]: item for item in candidates}
        grounded_selection = [
            {
                **item,
                "product_data": candidates_by_id[item["product_id"]],
            }
            for item in selected
            if item["product_id"] in candidates_by_id
        ]
        prompt = build_response_prompt(
            {
                "need_profile": state["need_profile"],
                "selected_products": grounded_selection,
            }
        )
        answer = _message_text(advisor_runtime.get_llm().invoke(prompt)).strip()
        if not answer:
            raise ValueError("Gemini returned an empty advisory response")
    return {
        "response": {"answer": answer},
        "control": {**state["control"], "stage": "completed"},
    }


def placeholder_response_node(state: AdvisorState) -> NodeUpdate:
    intent = state["routing"]["intent"]
    if intent == IntentLabel.OTHER.value:
        answer = "Mình hiện chỉ hỗ trợ luồng tư vấn tủ lạnh trong bản MVP."
    else:
        answer = (
            "Bản MVP hiện mới hỗ trợ tư vấn tủ lạnh; "
            f"ngành hàng {intent} sẽ được bổ sung sau."
        )
    return {
        "response": {"answer": answer},
        "control": {**state["control"], "stage": "completed"},
    }
