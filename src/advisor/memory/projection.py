"""Project untrusted natural-language memories into category-owned profile fields."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from typing import Any, Iterable

from advisor.categories.base import CategorySpec
from advisor.guardrails import GuardrailEngine
from advisor.schemas import ProfilePatch


CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "refrigerator": ("refrigerator", "fridge", "tủ lạnh", "tu lanh"),
    "washing_machine": ("washing machine", "washer", "máy giặt", "may giat"),
    "air_conditioner": ("air conditioner", "máy lạnh", "điều hòa"),
    "dryer": ("dryer", "máy sấy", "máy sấy quần áo"),
    "dishwasher": ("dishwasher", "máy rửa chén", "máy rửa bát"),
    "cooler_freezer": ("freezer", "tủ đông", "tủ mát"),
    "water_heater": ("water heater", "máy nước nóng", "bình nóng lạnh"),
    "karaoke_microphone": ("karaoke microphone", "micro karaoke"),
    "phone_recording_microphone": (
        "phone recording microphone",
        "micro thu âm điện thoại",
    ),
    "smartwatch": ("smartwatch", "đồng hồ thông minh"),
    "desktop": ("desktop", "máy tính để bàn"),
    "monitor": ("computer monitor", "màn hình máy tính"),
    "printer": ("printer", "máy in"),
    "tablet": ("tablet", "máy tính bảng"),
}

_HOUSEHOLD_PATTERNS = (
    re.compile(
        r"(?:gia\s*đình|nhà|hộ\s*gia\s*đình|household|family)"
        r"[^\d]{0,32}(\d{1,2})\s*(?:người|thành\s*viên|people|persons?|members?)",
        re.IGNORECASE,
    ),
    re.compile(r"(\d{1,2})[- ](?:person|member)\s+household", re.IGNORECASE),
    re.compile(
        r"(\d{1,2})\s*(?:người|thành\s*viên|people|persons?|members?)"
        r"\s*(?:trong|ở|in)\s*(?:gia\s*đình|nhà|household|family)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:household|family)\s*(?:size\s*(?:is|:)?|of|has|consists?\s+of)?"
        r"[^\d]{0,12}(\d{1,2})(?!\d)",
        re.IGNORECASE,
    ),
)
_ENERGY_PATTERN = re.compile(
    r"(?:tiết\s*kiệm\s*điện|tiết\s*kiệm\s*năng\s*lượng|energy[- ]?saving|"
    r"energy[- ]?efficient|save(?:s|d)?\s+(?:electricity|energy))",
    re.IGNORECASE,
)
_SHORT_PATTERN = re.compile(
    r"(?:trả\s*lời|câu\s*trả\s*lời)?\s*(?:thật\s*)?(?:ngắn\s*gọn|súc\s*tích)|"
    r"(?:concise|brief|short)\s+(?:answers?|responses?)|prefers?\s+(?:concise|brief)",
    re.IGNORECASE,
)
_LONG_PATTERN = re.compile(
    r"(?:trả\s*lời|giải\s*thích)\s+(?:chi\s*tiết|đầy\s*đủ)|"
    r"(?:detailed|thorough)\s+(?:answers?|responses?)",
    re.IGNORECASE,
)
_NAME_PATTERNS = (
    re.compile(
        r"(?:hãy\s+)?gọi\s+(?:tôi|mình|người\s*dùng)\s+là\s+"
        r"[\"']?([A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]{0,29}(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]{0,29})?)[\"']?",
    ),
    re.compile(
        r"(?:tên\s+gọi\s+mong\s+muốn|tên\s+ưa\s+thích|nickname|"
        r"preferred\s+(?:name|nickname))"
        r"[^\wÀ-ỹĐđ]{0,12}(?:là\s+)?"
        r"[\"']?([A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]{0,29}(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]{0,29})?)[\"']?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:muốn|thích)\s+được\s+gọi\s+là\s+"
        r"[\"']?([A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]{0,29}(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]{0,29})?)[\"']?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:prefers?\s+to\s+be\s+called|call\s+(?:me|the\s+user))\s+"
        r"[\"']?([A-Z][A-Za-z'-]{0,29}(?:\s+[A-Z][A-Za-z'-]{0,29})?)[\"']?",
        re.IGNORECASE,
    ),
)
_TONE_PATTERNS = {
    "friendly": re.compile(r"(?:thân\s*thiện|friendly|warm tone)", re.IGNORECASE),
    "professional": re.compile(
        r"(?:chuyên\s*nghiệp|professional|formal tone)", re.IGNORECASE
    ),
}
_MILLION_BUDGET_PATTERN = re.compile(
    r"(?:dưới|tối\s*đa|không\s*quá|ngân\s*sách|budget|up\s*to|under)?"
    r"[^\d]{0,18}(\d+(?:[.,]\d+)?)\s*(?:triệu|tr|million)",
    re.IGNORECASE,
)
_VND_BUDGET_PATTERN = re.compile(
    r"(?:ngân\s*sách|budget|tối\s*đa|dưới|under)[^\d]{0,18}"
    r"([\d.,]{6,15})\s*(?:đ|đồng|vnd)",
    re.IGNORECASE,
)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _safe_name(value: str) -> str | None:
    candidate = value.strip(" \t\r\n.,;:!?\"'")
    words = candidate.split()
    if len(words) > 1 and words[-1].casefold() in {"and", "và", "but", "nhưng"}:
        candidate = " ".join(words[:-1])
    if not 1 <= len(candidate) <= 50:
        return None
    if any(char in candidate for char in "{}[]<>\\/=@"):
        return None
    if not all(char.isalpha() or char in " '-" for char in candidate):
        return None
    return candidate


def extract_response_preferences(text: str) -> dict[str, str]:
    """Extract non-decision response preferences without invoking an LLM."""

    preferences: dict[str, str] = {}
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match and (name := _safe_name(match.group(1))):
            preferences["preferred_name"] = name
            break
    if _SHORT_PATTERN.search(text):
        preferences["answer_length"] = "short"
    elif _LONG_PATTERN.search(text):
        preferences["answer_length"] = "detailed"
    for tone, pattern in _TONE_PATTERNS.items():
        if pattern.search(text):
            preferences["tone"] = tone
            break
    return preferences


def response_preferences_from_memories(
    memories: Iterable[dict[str, Any]], guardrail: GuardrailEngine
) -> dict[str, str]:
    preferences: dict[str, str] = {}
    for item in _sort_memories(memories):
        text = str(item.get("memory") or "")
        decision = guardrail.inspect(text, surface="recalled_memory")
        guardrail.record(decision, text)
        if guardrail.should_block(decision):
            continue
        extracted = extract_response_preferences(text)
        for key, value in extracted.items():
            preferences.setdefault(key, value)
    return preferences


def _sort_memories(memories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[str, float]:
        timestamp = str(item.get("updated_at") or item.get("created_at") or "")
        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        return timestamp, score

    return sorted((dict(item) for item in memories), key=key, reverse=True)


def _question_for_path(spec: CategorySpec, path: str) -> str | None:
    for question_id, paths in spec.question_profile_paths.items():
        if path in paths:
            return question_id
    return None


def _path_get(profile: dict[str, Any], path: str) -> Any:
    value: Any = profile
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _path_set(profile: dict[str, Any], path: str, value: Any) -> None:
    target = profile
    parts = path.split(".")
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            nested = {}
            target[part] = nested
        target = nested
    target[parts[-1]] = value


def _path_has_value(profile: dict[str, Any], path: str) -> bool:
    value = _path_get(profile, path)
    return value not in (None, "", [], {})


def _patch_paths(patch: ProfilePatch) -> set[str]:
    return {
        *patch.set,
        *patch.replace,
        *patch.add,
        *patch.remove,
        *patch.clear,
    }


def _validated_candidate_patch(
    *,
    spec: CategorySpec,
    question_id: str,
    current_profile: dict[str, Any],
    patch: ProfilePatch,
) -> ProfilePatch | None:
    paths = _patch_paths(patch)
    allowed = spec.question_profile_paths.get(question_id, frozenset())
    if not paths or not paths <= allowed or not paths <= spec.valid_patch_paths:
        return None
    profile = deepcopy(current_profile)
    before = json.dumps(profile, ensure_ascii=False, sort_keys=True, default=str)
    for path in patch.clear:
        _path_set(profile, path, [] if path in spec.list_patch_paths else None)
    for operation in (patch.set, patch.replace):
        for path, value in operation.items():
            if path in spec.list_patch_paths and not isinstance(value, list):
                return None
            _path_set(profile, path, deepcopy(value))
    for path, removed in patch.remove.items():
        if path not in spec.list_patch_paths:
            return None
        _path_set(
            profile,
            path,
            [item for item in list(_path_get(profile, path) or []) if item not in removed],
        )
    for path, added in patch.add.items():
        if path not in spec.list_patch_paths:
            return None
        values = list(_path_get(profile, path) or [])
        for item in added:
            if item not in values:
                values.append(item)
        _path_set(profile, path, values)
    try:
        spec.profile_model.model_validate(profile)
    except ValueError:
        return None
    after = json.dumps(profile, ensure_ascii=False, sort_keys=True, default=str)
    return patch if before != after else None


def _source(item: dict[str, Any]) -> dict[str, Any]:
    try:
        score = float(item.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "memory_id": str(item.get("id") or ""),
        "score": score,
        "categories": [str(value) for value in item.get("categories") or []],
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def _candidate(
    *,
    spec: CategorySpec,
    question_id: str,
    display_value: str,
    patch: ProfilePatch,
    source: dict[str, Any],
) -> dict[str, Any]:
    patch_data = patch.model_dump(mode="json", exclude_defaults=True)
    digest = hashlib.sha256(
        json.dumps(
            [spec.name, question_id, patch_data, source.get("memory_id")],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    catalog = spec.config["question_catalog"][question_id]
    return {
        "candidate_id": f"mem-{digest}",
        "question_id": question_id,
        "question": catalog["question"],
        "display_value": display_value,
        "proposed_patch": patch_data,
        "options": [
            {"option_id": option["option_id"], "label": option["label"]}
            for option in catalog["options"]
        ],
        "sources": [_source(source)],
    }


def _household_size(text: str) -> int | None:
    for pattern in _HOUSEHOLD_PATTERNS:
        match = pattern.search(text)
        if match:
            size = int(match.group(1))
            return size if 1 <= size <= 30 else None
    return None


def _category_matches(item: dict[str, Any], spec: CategorySpec) -> bool:
    metadata = item.get("metadata") or {}
    raw_category = str(
        metadata.get("active_category") or metadata.get("category") or ""
    ).strip()
    aliases = (spec.display_name, *CATEGORY_ALIASES.get(spec.name, ()))
    folded_text = _fold(str(item.get("memory") or ""))
    text_matches = any(_fold(alias) in folded_text for alias in aliases)
    if raw_category:
        metadata_matches = _fold(raw_category) in {
            _fold(spec.name),
            _fold(spec.display_name),
            *(_fold(alias) for alias in CATEGORY_ALIASES.get(spec.name, ())),
        }
        # Budget memories are required to name their category. Requiring both
        # signals prevents an inaccurate turn-level metadata tag from crossing
        # a hard constraint into another product category.
        return metadata_matches and text_matches
    return text_matches


def _budget_vnd(text: str) -> int | None:
    match = _MILLION_BUDGET_PATTERN.search(text)
    if match:
        amount = float(match.group(1).replace(",", "."))
        result = int(amount * 1_000_000)
        return result if result >= 0 else None
    match = _VND_BUDGET_PATTERN.search(text)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    return int(digits) if digits else None


def project_memories(
    *,
    memories: Iterable[dict[str, Any]],
    spec: CategorySpec,
    current_profile: dict[str, Any],
    current_changed_paths: Iterable[str] = (),
    confirmed_question_ids: Iterable[str] = (),
    guardrail: GuardrailEngine,
    limit: int = 3,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return validated decision candidates and IDs quarantined by guardrails."""

    changed = set(current_changed_paths)
    confirmed = set(confirmed_question_ids)
    candidates: list[dict[str, Any]] = []
    blocked_ids: list[str] = []
    claimed_questions: set[str] = set()

    for item in _sort_memories(memories):
        text = str(item.get("memory") or "")
        decision = guardrail.inspect(text, surface="recalled_memory")
        guardrail.record(decision, text)
        if guardrail.should_block(decision):
            blocked_ids.append(str(item.get("id") or ""))
            continue

        proposals: list[tuple[str | None, str, ProfilePatch]] = []
        size = _household_size(text)
        proposals.append(
            (
                _question_for_path(spec, "household_size") if size else None,
                f"Gia đình {size} người" if size else "",
                ProfilePatch(set={"household_size": size}) if size else ProfilePatch(),
            )
        )
        if _ENERGY_PATTERN.search(text):
            proposals.append(
                (
                    _question_for_path(spec, "usage_preferences"),
                    "Ưu tiên tiết kiệm điện",
                    ProfilePatch(add={"usage_preferences": ["energy_saving"]}),
                )
            )
        if _category_matches(item, spec) and (budget := _budget_vnd(text)) is not None:
            proposals.append(
                (
                    _question_for_path(spec, "budget_max_vnd"),
                    f"Ngân sách tối đa {budget:,.0f} đồng".replace(",", "."),
                    ProfilePatch(set={"budget_max_vnd": budget}),
                )
            )

        for question_id, display_value, patch in proposals:
            if not question_id or question_id in claimed_questions or question_id in confirmed:
                continue
            question_paths = spec.question_profile_paths.get(question_id, frozenset())
            submitted_paths = _patch_paths(patch)
            paths_shared_with_other_questions = {
                path
                for other_question, paths in spec.question_profile_paths.items()
                if other_question != question_id
                for path in paths
            }
            profile_precedence_paths = submitted_paths | (
                set(question_paths) - paths_shared_with_other_questions
            )
            if changed & question_paths or any(
                _path_has_value(current_profile, path)
                for path in profile_precedence_paths
            ):
                continue
            validated = _validated_candidate_patch(
                spec=spec,
                question_id=question_id,
                current_profile=current_profile,
                patch=patch,
            )
            if validated is None:
                continue
            candidates.append(
                _candidate(
                    spec=spec,
                    question_id=question_id,
                    display_value=display_value,
                    patch=validated,
                    source=item,
                )
            )
            claimed_questions.add(question_id)
            if len(candidates) >= limit:
                priority = {"household_size": 0, "usage_preferences": 1, "budget": 2}
                return sorted(
                    candidates,
                    key=lambda item: priority.get(str(item.get("question_id")), 99),
                ), blocked_ids
    priority = {"household_size": 0, "usage_preferences": 1, "budget": 2}
    return sorted(
        candidates,
        key=lambda item: priority.get(str(item.get("question_id")), 99),
    ), blocked_ids


__all__ = [
    "extract_response_preferences",
    "project_memories",
    "response_preferences_from_memories",
]
