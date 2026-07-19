"""Structured follow-up question generation for completed chat turns."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def reject_blank_content(self) -> ConversationMessage:
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("conversation messages must not be blank")
        return self


class SuggestionRequest(BaseModel):
    conversation: list[ConversationMessage] = Field(min_length=2, max_length=40)

    @model_validator(mode="after")
    def limit_total_context(self) -> SuggestionRequest:
        total_characters = sum(len(message.content) for message in self.conversation)
        if total_characters > 40_000:
            raise ValueError("conversation must not exceed 40,000 characters")
        if self.conversation[-1].role != "assistant":
            raise ValueError("conversation must end with an assistant message")
        return self


class SuggestionResult(BaseModel):
    questions: list[str] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def normalize_questions(self) -> SuggestionResult:
        normalized: list[str] = []
        seen: set[str] = set()
        for question in self.questions:
            value = " ".join(question.split()).strip()
            if not 3 <= len(value) <= 200:
                raise ValueError("each suggestion must contain 3 to 200 characters")
            key = value.casefold().rstrip("?!. ")
            if key in seen:
                raise ValueError("suggestions must be unique")
            seen.add(key)
            normalized.append(value)
        self.questions = normalized
        return self


SUGGESTION_SYSTEM_PROMPT = """Bạn tạo câu hỏi gợi ý cho người dùng của trợ lý mua sắm.

Đọc hội thoại và viết đúng 4 câu mà NGƯỜI DÙNG có thể gửi tiếp cho trợ lý.
Các câu phải:
- bằng tiếng Việt tự nhiên, ngắn gọn, cụ thể và liên quan trực tiếp đến ngữ cảnh;
- hữu ích để tiến gần hơn đến quyết định mua hàng;
- đa dạng về ý định, ưu tiên làm rõ, so sánh, đào sâu hoặc hành động tiếp theo;
- không lặp lại câu hỏi hay thông tin đã được giải đáp trong hội thoại;
- không bịa sản phẩm, giá, chính sách hoặc dữ kiện chưa xuất hiện;
- không nói về việc bạn đang phân tích hội thoại.

Chỉ trả dữ liệu theo schema được yêu cầu."""


def generate_suggestions(llm: Any, request: SuggestionRequest) -> SuggestionResult:
    """Generate four structured suggestions using the configured chat model."""
    transcript = "\n\n".join(
        f"{('Người dùng' if message.role == 'user' else 'Trợ lý')}: {message.content}"
        for message in request.conversation
    )
    structured = llm.with_structured_output(SuggestionResult, method="json_schema")
    result = structured.invoke(
        [
            SystemMessage(content=SUGGESTION_SYSTEM_PROMPT),
            HumanMessage(content=f"Hội thoại hiện tại:\n\n{transcript}"),
        ]
    )
    if isinstance(result, SuggestionResult):
        return result
    return SuggestionResult.model_validate(result)
