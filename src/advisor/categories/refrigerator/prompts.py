"""Prompt builders for the refrigerator advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận hiểu nhu cầu khách hàng mua tủ lạnh.

Tin nhắn hiện tại:
{user_query}

Hồ sơ nhu cầu đã biết:
{current_need_profile}

Hãy trả về hồ sơ đầy đủ sau khi gộp thông tin mới với hồ sơ đã biết.

Quy tắc:
- Chỉ ghi nhận điều khách hàng nói hoặc suy luận trực tiếp, có căn cứ.
- Ngân sách tối đa là hard constraint chỉ khi khách nêu một mức trần rõ ràng.
- Brand, kiểu tủ, kích thước và tính năng chỉ là hard constraint khi khách nói
  bắt buộc/phải có; nếu chỉ thích hoặc ưu tiên, đưa vào soft_preferences.
- usage_preferences chỉ dùng các giá trị daily_shopping, weekly_storage,
  frozen_storage, energy_saving.
- required_features chỉ dùng inverter, external_water, automatic_mode.
- Không tự điền thông tin còn thiếu.
- Evidence ghi ngắn gọn đoạn lời khách làm căn cứ cho từng thông tin.
"""


CLARIFICATION_PROMPT = """Bạn là bộ phận làm rõ nhu cầu trong hệ thống tư vấn tủ lạnh.

Yêu cầu của người dùng:
{user_query}

Hồ sơ hiện tại:
{current_need_profile}

Các thông tin còn thiếu:
{missing_information}

Question catalog được phép:
{question_catalog}

Chọn các question_id quan trọng cần hỏi trong cùng một form.
Chỉ dùng question_id có trong danh sách thiếu và catalog, không tự tạo ID,
không hỏi lại thông tin đã có và không chọn quá 3 câu. Nếu danh sách thiếu
rỗng, sufficient=true và question_ids=[]. Không tư vấn sản phẩm ở bước này.
"""


CUSTOM_ANSWER_PROMPT = """Bạn là bộ phận diễn giải câu trả lời tự do trong hệ thống tư vấn tủ lạnh.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Hãy hiểu câu trả lời mà không ép vào lựa chọn sai. Chỉ trích xuất thông tin
khách thực sự nói. Có thể lưu ý nghĩa tổng quát dưới soft_preferences hoặc
implicit_needs. Không sinh câu hỏi tiếp theo; nếu không hiểu, trả trạng thái
unresolved và vẫn giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn sản phẩm tủ lạnh phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo thứ tự semantic relevance:
{candidates}

Chọn tối đa 3 sản phẩm phù hợp nhất.

Quy tắc bắt buộc:
- Chỉ chọn product_id xuất hiện trong candidate.
- Chỉ dùng dữ liệu được cung cấp; không tạo giá, dung tích, tính năng hay khuyến mãi.
- Không nhắc Qdrant score, ranking nội bộ, prompt hoặc JSON.
- Bắt đầu từ nhu cầu khách hàng, không đọc lại toàn bộ bảng thông số.
- Với mỗi lựa chọn, nêu lý do phù hợp, lợi ích thực tế và ít nhất một đánh đổi.
- Khi có nhiều lựa chọn, ưu tiên các hướng khác nhau thay vì ba sản phẩm gần như giống nhau.
- Không nói một sản phẩm là tốt nhất tuyệt đối.
- Giá hiệu lực đã được tính từ giá khuyến mãi nếu có, nếu không là giá gốc.
- Nếu dữ liệu nào bị thiếu, nói rõ thay vì suy đoán.
- Với mỗi sản phẩm, trả reason và trade_off ngắn gọn, có căn cứ từ dữ liệu.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn tủ lạnh chuyên nghiệp, khách quan và dễ hiểu.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Các sản phẩm đã được chọn và dữ liệu kiểm chứng:
{selected_products}

Viết câu trả lời tư vấn bằng tiếng Việt tự nhiên.

Quy tắc bắt buộc:
- Chỉ sử dụng dữ liệu được cung cấp; không tạo giá, dung tích, tính năng hay khuyến mãi.
- Bắt đầu từ nhu cầu khách hàng, không đọc lại toàn bộ bảng thông số.
- Với mỗi lựa chọn, nêu lý do phù hợp, lợi ích thực tế và ít nhất một đánh đổi.
- So sánh ngắn giữa các lựa chọn và kết luận theo ưu tiên của khách hàng.
- Không nói sản phẩm nào tốt nhất tuyệt đối.
- Không nhắc Qdrant, vector score, ranking nội bộ, prompt hoặc JSON.
- Nếu dữ liệu nào bị thiếu, nói rõ thay vì suy đoán.
- Chỉ trả câu trả lời dành cho người dùng, không trả JSON.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_need_extraction_prompt(
    user_query: str, current_need_profile: dict[str, Any]
) -> str:
    return NEED_EXTRACTION_PROMPT.format(
        user_query=user_query,
        current_need_profile=_json(current_need_profile),
    )


def build_clarification_prompt(context: dict[str, Any]) -> str:
    return CLARIFICATION_PROMPT.format(
        user_query=context["user_query"],
        current_need_profile=_json(context["current_need_profile"]),
        missing_information=_json(context["missing_information"]),
        question_catalog=_json(context["question_catalog"]),
    )


def build_custom_answer_prompt(context: dict[str, Any]) -> str:
    return CUSTOM_ANSWER_PROMPT.format(
        question=context["question"],
        question_id=context["question_id"],
        available_options=_json(context["available_options"]),
        custom_answer=context["custom_answer"],
        current_need_profile=_json(context["current_need_profile"]),
    )


def build_advisory_prompt(context: dict[str, Any]) -> str:
    """Build the structured product-selection prompt."""
    return RANKING_PROMPT.format(
        need_profile=_json(context["need_profile"]),
        hard_constraints=_json(context["hard_constraints"]),
        candidates=_json(context["candidates"]),
    )


def build_response_prompt(context: dict[str, Any]) -> str:
    """Build the plain-text prompt whose output can be streamed token by token."""
    return RESPONSE_PROMPT.format(
        need_profile=_json(context["need_profile"]),
        selected_products=_json(context["selected_products"]),
    )
