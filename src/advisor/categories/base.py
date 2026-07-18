"""Stable contract between the shared advisor graph and product categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel
from qdrant_client import models


NeedPromptBuilder = Callable[..., str]
ContextPromptBuilder = Callable[[dict[str, Any]], str]
FilterBuilder = Callable[[dict[str, Any], dict[str, str] | None], models.Filter | None]
SearchTextBuilder = Callable[[dict[str, Any], str], str]
NoMatchAnswerBuilder = Callable[[dict[str, Any]], str]
CandidateNormalizer = Callable[[Any], dict[str, Any]]
MissingFieldsResolver = Callable[[dict[str, Any], dict[str, Any] | None], list[str]]


@dataclass(frozen=True)
class CategorySpec:
    """All category-owned behavior consumed by the generic graph.

    A new category implements this contract in its own package.  Shared graph,
    API, state, and retrieval modules must not import a concrete category.
    """

    name: str
    display_name: str
    config: dict[str, Any]
    profile_model: type[BaseModel]
    custom_answer_model: type[BaseModel]
    build_need_extraction_prompt: NeedPromptBuilder
    build_custom_answer_prompt: ContextPromptBuilder
    build_ranking_prompt: ContextPromptBuilder
    build_response_prompt: ContextPromptBuilder
    build_filter: FilterBuilder
    build_search_text: SearchTextBuilder
    no_match_answer: NoMatchAnswerBuilder
    normalize_candidate: CandidateNormalizer
    get_missing_profile_fields: MissingFieldsResolver
    valid_patch_paths: frozenset[str]
    list_patch_paths: frozenset[str]
    hard_retrieval_paths: frozenset[str]
    question_profile_paths: dict[str, frozenset[str]] = field(default_factory=dict)
    setup_indexes_command: str = ""

    def validate(self) -> None:
        """Fail early when a category package violates the shared contract."""
        required_config = {
            "category",
            "collection",
            "embedding_model",
            "retrieval_limit",
            "payload_fields",
            "payload_indexes",
            "required_profile_fields",
            "question_catalog",
        }
        missing = sorted(required_config - self.config.keys())
        if missing:
            raise ValueError(f"Category {self.name!r} is missing config keys: {missing}")
        if self.config["category"] != self.name:
            raise ValueError(
                f"Category spec name {self.name!r} does not match config category "
                f"{self.config['category']!r}"
            )
        unknown_questions = sorted(
            set(self.config["required_profile_fields"])
            - set(self.config["question_catalog"])
        )
        if unknown_questions:
            raise ValueError(
                f"Category {self.name!r} has required fields without questions: "
                f"{unknown_questions}"
            )
        if not self.list_patch_paths <= self.valid_patch_paths:
            raise ValueError("list_patch_paths must be a subset of valid_patch_paths")
        if not self.hard_retrieval_paths <= self.valid_patch_paths:
            raise ValueError("hard_retrieval_paths must be a subset of valid_patch_paths")
        if not set(self.question_profile_paths) <= set(
            self.config["question_catalog"]
        ):
            raise ValueError("question_profile_paths contains unknown question IDs")
        question_paths = set().union(*self.question_profile_paths.values()) if (
            self.question_profile_paths
        ) else set()
        if not question_paths <= self.valid_patch_paths:
            raise ValueError("question_profile_paths contains invalid profile paths")
        unindexed = sorted(
            set(self.config["payload_fields"].values())
            - set(self.config["payload_indexes"])
        )
        if unindexed:
            raise ValueError(
                f"Category {self.name!r} exposes unindexed filter fields: {unindexed}"
            )
        custom_fields = set(self.custom_answer_model.model_fields)
        required_custom_fields = {
            "interpretation_status",
            "raw_answer",
            "confidence",
        }
        if not required_custom_fields <= custom_fields:
            raise ValueError(
                "custom_answer_model must define interpretation_status, raw_answer, "
                "and confidence"
            )
        for question_id, question in self.config["question_catalog"].items():
            options = question.get("options") or []
            option_ids = [item.get("option_id") for item in options]
            if len(option_ids) != len(set(option_ids)):
                raise ValueError(f"Question {question_id!r} has duplicate option IDs")
            if "other" not in option_ids:
                raise ValueError(f"Question {question_id!r} must provide an 'other' option")
            for option in options:
                update_paths = _leaf_paths(option.get("profile_updates") or {})
                invalid_paths = sorted(update_paths - self.valid_patch_paths)
                if invalid_paths:
                    raise ValueError(
                        f"Question {question_id!r} has invalid profile updates: "
                        f"{invalid_paths}"
                    )


def _leaf_paths(value: dict[str, Any], prefix: str = "") -> set[str]:
    """Return dotted leaf paths from a nested profile update."""
    paths: set[str] = set()
    for key, nested in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(nested, dict) and nested:
            paths.update(_leaf_paths(nested, path))
        else:
            paths.add(path)
    return paths


def serialize_filter(query_filter: models.Filter | None) -> dict[str, Any] | None:
    """Serialize a Qdrant filter for checkpoint-safe graph state."""
    if query_filter is None:
        return None
    return query_filter.model_dump(mode="json", exclude_none=True)


def deserialize_filter(data: dict[str, Any] | None) -> models.Filter | None:
    """Restore a Qdrant filter from graph state."""
    return models.Filter.model_validate(data) if data else None
