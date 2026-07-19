"""Conversation-aware guidance when catalog retrieval returns no products."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from advisor.nodes import AdvisorRuntime, TURN_ANALYSIS_PROMPT, compose_response_node


class RecordingLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.answer)


def no_match_state() -> dict[str, Any]:
    previous_answer = (
        "Mình chưa tìm thấy micro karaoke đáp ứng loại micro không dây "
        "trong dữ liệu hiện có."
    )
    profile = {
        "category": "karaoke_microphone",
        "connection_preference": "wireless",
        "budget_max_vnd": 3_000_000,
        "hard_constraints": {"microphone_types": ["wireless"]},
    }
    return {
        "messages": [
            HumanMessage(content="Tìm micro karaoke không dây"),
            AIMessage(content=previous_answer),
            HumanMessage(content="micro khác"),
        ],
        "conversation": {
            "active_category": "karaoke_microphone",
            "execution_mode": "retrieve",
            "analysis": {
                "action": "more_options",
                "scope": "current_recommendations",
            },
        },
        "category_contexts": {
            "karaoke_microphone": {
                "profile": profile,
                "recommendation_context": {
                    "candidate_pool": [],
                    "product_snapshots": {},
                    "ranking_by_id": {},
                    "last_presented_ids": [],
                    "presented_ids": [],
                    "presentations": [],
                    "discovery_query": "Tìm micro karaoke không dây",
                    "last_answer": previous_answer,
                },
            }
        },
        "need_profile": profile,
        "retrieval": {"candidates": [], "candidate_count": 0},
        "ranking": {"selected_products": []},
        "control": {
            "current_user_input": "micro khác",
            "stage": "ranking_completed",
        },
    }


def test_no_match_uses_system_rules_and_conversation_context() -> None:
    answer = (
        "Mình hiểu bạn muốn một lựa chọn khác. Hiện chưa có mẫu không dây khớp "
        "đầy đủ; bạn muốn tăng ngân sách hay xem micro có dây?"
    )
    llm = RecordingLLM(answer)

    result = compose_response_node(no_match_state(), AdvisorRuntime(llm=llm))

    assert result["response"]["answer"] == answer
    assert result["ranking"]["selected_products"] == []
    recommendation = result["category_contexts"]["karaoke_microphone"][
        "recommendation_context"
    ]
    assert recommendation["discovery_query"] == "Tìm micro karaoke không dây"
    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert "không chỉ dừng ở câu" in messages[0].content
    assert "không bịa tên sản phẩm" in messages[0].content.casefold()
    prompt = messages[1].content
    assert "Tin nhắn hiện tại: micro khác" in prompt
    assert "Loại hành động của lượt này: more_options" in prompt
    assert "Câu trả lời trước" in prompt
    assert "micro karaoke đáp ứng loại micro không dây" in prompt
    assert "2-3 hướng xử lý cụ thể" in prompt


def test_empty_no_match_model_answer_still_returns_actionable_guidance() -> None:
    result = compose_response_node(
        no_match_state(), AdvisorRuntime(llm=RecordingLLM(""))
    )

    answer = result["response"]["answer"].casefold()
    assert "chưa tìm thấy micro karaoke" in answer
    assert "phương án gần nhất" in answer
    assert "tiêu chí có thể linh hoạt" in answer


def test_turn_analysis_explicitly_inherits_short_more_options_follow_up() -> None:
    prompt = TURN_ANALYSIS_PROMPT.casefold()

    assert "micro khác" in prompt
    assert "vẫn là more_options" in prompt
    assert "kế thừa ngành hiện tại" in prompt
