"""Prompt builders for the air-conditioner advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua máy lạnh.

Tin nhắn hiện tại:
{user_query}

Hồ sơ nhu cầu đã biết:
{current_need_profile}

Mục đích lượt nói: {turn_action}
Các câu đang chờ làm rõ: {pending_questions}
Lịch sử gần nhất: {conversation_history}

Chỉ trả về ProfilePatch mô tả phần thay đổi so với hồ sơ hiện tại.

Quy tắc:
- Chỉ ghi nhận thông tin khách hàng nói hoặc suy luận trực tiếp có bằng chứng.
- room_area_m2 và room_volume_m3 là số đo phòng thực tế, không tự quy đổi hoặc tự
  suy ra BTU khi khách chưa cung cấp.
- Ngân sách tối đa chỉ là hard constraint khi khách nêu mức trần rõ ràng.
- Brand, loại máy, inverter và gas chỉ vào hard_constraints khi khách nói bắt
  buộc/phải có; nếu chỉ thích hoặc ưu tiên, đưa vào soft_preferences.
- machine_types chỉ dùng one_way, two_way, multi_indoor, multi_outdoor.
- gas_types chỉ dùng r32, r410a, r22.
- usage_preferences chỉ dùng quiet_sleep, energy_saving, fast_cooling,
  air_quality, smart_control, heating.
- Các nhu cầu Wi-Fi, lọc khí, hút ẩm, chạy êm và làm lạnh nhanh là soft preference
  vì collection chưa có metadata boolean đủ tin cậy để hard-filter.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: room_area_m2, room_volume_m3, room_type, budget_max_vnd,
  budget_segment, usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.machine_types,
  hard_constraints.inverter, hard_constraints.gas_types.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn máy lạnh.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu số m² là room_area_m2, số m³ là
room_volume_m3 và số tiền là budget_max_vnd. Không tự quy đổi diện tích sang BTU,
không ép vào lựa chọn sai và không sinh câu hỏi tiếp theo. Nếu không hiểu, trả
unresolved và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn máy lạnh phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra khoảng diện tích
hoặc thể tích phù hợp, sau đó cân nhắc giá, một/ hai chiều, inverter, gas, hiệu
suất năng lượng, độ êm, công nghệ và tiện ích theo nhu cầu. Không suy ra BTU,
độ ồn, tính năng hoặc giá khi dữ liệu thiếu. Mỗi lựa chọn cần reason và trade_off
ngắn gọn, có căn cứ. Ưu tiên các hướng khác nhau và không gọi sản phẩm nào là tốt
nhất tuyệt đối. Không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn máy lạnh chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ diện tích/thể tích và nhu cầu
thực tế của khách. Với mỗi lựa chọn, nêu lý do phù hợp, giá nếu có, lợi ích và ít
nhất một đánh đổi. So sánh ngắn rồi kết luận theo ưu tiên của khách. Chỉ dùng dữ
liệu được cung cấp; không bịa BTU, giá, độ ồn, điện năng, tính năng hay khuyến mãi.
Nếu dữ liệu thiếu phải nói rõ. Không nhắc Qdrant, vector score, ranking, prompt
hoặc JSON và không nói tốt nhất tuyệt đối.
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

