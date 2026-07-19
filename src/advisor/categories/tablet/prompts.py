"""Prompt builders for the tablet advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua máy tính bảng.

Tin nhắn hiện tại:
{user_query}

Hồ sơ nhu cầu đã biết:
{current_need_profile}

Mục đích lượt nói: {turn_action}
Các câu đang chờ làm rõ: {pending_questions}
Lịch sử gần nhất: {conversation_history}

Chỉ trả về ProfilePatch mô tả phần thay đổi so với hồ sơ hiện tại.

Quy tắc:
- Chỉ ghi nhận thông tin khách nói hoặc suy luận trực tiếp có bằng chứng.
- Ngân sách tối đa chỉ là hard constraint khi khách nêu mức trần rõ ràng.
- Hãng, hệ điều hành, RAM/bộ nhớ tối thiểu, khoảng màn hình, khối lượng tối đa,
  khả năng gọi điện hoặc thẻ nhớ chỉ vào hard_constraints khi khách nói bắt buộc.
- connectivity_segment chỉ dùng wifi_only, cellular_4g, cellular_5g, flexible;
  chỉ đặt 4G/5G khi đó là yêu cầu, còn “ưu tiên” phải vào soft_preferences.
- usage_preferences chỉ dùng study_work, entertainment, gaming, drawing_notes,
  children, general.
- Bút, bàn phím, camera, loa, pin lâu và sạc nhanh là soft preference vì dữ liệu
  mô tả chưa đủ tin cậy để loại sản phẩm.
- os_families chỉ dùng android, ipados, harmonyos; display_families chỉ dùng
  ips_lcd, oled_amoled, mini_led, tft_lcd, lcd, other.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: budget_max_vnd, budget_segment, connectivity_segment,
  usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.os_families,
  hard_constraints.display_families, hard_constraints.min_ram_gb,
  hard_constraints.min_storage_gb, hard_constraints.min_screen_size_inch,
  hard_constraints.max_screen_size_inch, hard_constraints.max_weight_g,
  hard_constraints.requires_calls, hard_constraints.requires_memory_card.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn máy tính bảng.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu tiền theo VND, RAM/bộ nhớ theo GB,
màn hình theo inch, khối lượng theo gram và kết nối theo Wi-Fi/4G/5G. Không ép
câu trả lời vào lựa chọn sai, không tự biến sở thích thành hard constraint và
không sinh câu hỏi tiếp theo. Nếu không hiểu, trả unresolved và giữ raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn máy tính bảng phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Đánh giá theo mục đích sử dụng,
hiệu năng CPU/GPU và RAM, màn hình, bộ nhớ, hệ điều hành, kết nối, khối lượng,
pin và sạc. Với nhu cầu vẽ/ghi chú, bàn phím, camera hoặc gaming chỉ xác nhận
khi candidate có dữ liệu. Không quy đổi mAh sang Wh, không suy đoán hiệu năng,
giá, phụ kiện hay tính năng khi thiếu. Mỗi lựa chọn cần reason và trade_off ngắn,
có căn cứ; ưu tiên các hướng khác nhau. Không nói tốt nhất tuyệt đối và không
nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn máy tính bảng chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ mục đích sử dụng và ngân sách.
Với mỗi lựa chọn, nêu lý do phù hợp, giá nếu có, hiệu năng/màn hình/lưu trữ/kết
nối đáng chú ý và ít nhất một đánh đổi như trọng lượng, pin, hệ điều hành, thiếu
kết nối di động hoặc dữ liệu tính năng. Chỉ dùng dữ liệu được cung cấp; không bịa
giá, phụ kiện, camera, bút, bàn phím, pin hay khuyến mãi. Nếu thiếu phải nói rõ.
Không nhắc Qdrant, vector score, ranking, prompt hoặc JSON.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_need_extraction_prompt(
    user_query: str,
    current_need_profile: dict[str, Any],
    *,
    turn_action: str = "discover",
    pending_questions: list[dict[str, Any]] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    return NEED_EXTRACTION_PROMPT.format(
        user_query=user_query,
        current_need_profile=_json(current_need_profile),
        turn_action=turn_action,
        pending_questions=_json(pending_questions or []),
        conversation_history=_json(conversation_history or []),
    )


def build_custom_answer_prompt(context: dict[str, Any]) -> str:
    return CUSTOM_ANSWER_PROMPT.format(
        question=context["question"],
        question_id=context["question_id"],
        available_options=_json(context["available_options"]),
        custom_answer=context["custom_answer"],
        current_need_profile=_json(context["current_need_profile"]),
    )


def build_ranking_prompt(context: dict[str, Any]) -> str:
    return RANKING_PROMPT.format(
        need_profile=_json(context["need_profile"]),
        hard_constraints=_json(context["hard_constraints"]),
        candidates=_json(context["candidates"]),
    )


def build_response_prompt(context: dict[str, Any]) -> str:
    return RESPONSE_PROMPT.format(
        user_query=context.get("user_query", ""),
        turn_action=context.get("turn_action", "discover"),
        conversation_history=_json(context.get("conversation_history", [])),
        need_profile=_json(context["need_profile"]),
        selected_products=_json(context["selected_products"]),
    )
