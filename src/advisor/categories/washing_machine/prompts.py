"""Prompt builders for the washing-machine advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua máy giặt.

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
- Brand, loại máy, kiểu lồng, tải giặt, kích thước, inverter và sấy chỉ vào
  hard_constraints khi khách nói bắt buộc/phải có/ít nhất/không quá.
- product_types chỉ dùng top_load, front_load, washer_dryer, mini, wash_tower.
- drum_types chỉ dùng vertical, horizontal.
- usage_preferences chỉ dùng daily_laundry, bulky_items, hygiene_care,
  energy_saving, quick_wash, wash_and_dry.
- Hơi nước, giặt nhanh, khóa trẻ em, điều khiển ứng dụng, tự vệ sinh lồng và
  kháng khuẩn là soft preference vì metadata chưa đủ tin cậy để hard-filter.
- Câu “máy 10 kg” không tự biến thành tối thiểu 10 kg; chỉ đặt
  min_wash_capacity_kg khi khách nói “ít nhất/từ 10 kg”.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: household_size, budget_max_vnd, budget_segment,
  usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.product_types,
  hard_constraints.drum_types, hard_constraints.min_wash_capacity_kg,
  hard_constraints.max_width_cm, hard_constraints.max_height_cm,
  hard_constraints.max_depth_cm, hard_constraints.inverter,
  hard_constraints.dryer.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn máy giặt.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu số người là household_size, số tiền
là budget_max_vnd và tải giặt có đơn vị kg. Không ép câu trả lời vào lựa chọn sai,
không tự biến sở thích thành hard constraint và không sinh câu hỏi tiếp theo.
Nếu không hiểu, trả unresolved và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn máy giặt phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra số người, tải
giặt và giá hiệu lực, sau đó cân nhắc loại cửa/lồng, khả năng sấy, inverter,
tốc độ vắt, chương trình, công nghệ, tiện ích và kích thước theo nhu cầu. Với
đồ cồng kềnh ưu tiên tải phù hợp; với nhu cầu vệ sinh chỉ đánh giá hơi nước hoặc
kháng khuẩn khi candidate có dữ liệu. Không suy đoán giá, điện năng, tính năng
hoặc khả năng sấy khi dữ liệu thiếu. Mỗi lựa chọn cần reason và trade_off ngắn,
có căn cứ; ưu tiên các hướng khác nhau. Không gọi sản phẩm nào tốt nhất tuyệt
đối và không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn máy giặt chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ số người, khối lượng đồ và nhu
cầu thực tế. Với mỗi lựa chọn, nêu lý do phù hợp, giá nếu có, lợi ích và ít nhất
một đánh đổi như tải giặt, kiểu lồng, kích thước, giá hoặc khả năng sấy. So sánh
ngắn rồi kết luận theo ưu tiên của khách. Chỉ dùng dữ liệu được cung cấp; không
bịa giá, điện năng, chương trình, công nghệ, tính năng hay khuyến mãi. Nếu dữ
liệu thiếu phải nói rõ. Không nhắc Qdrant, vector score, ranking, prompt hoặc
JSON và không nói tốt nhất tuyệt đối.
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
