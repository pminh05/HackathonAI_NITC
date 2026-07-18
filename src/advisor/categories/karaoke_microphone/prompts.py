"""Prompt builders for the micro-karaoke advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua micro karaoke.

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
- connection_preference chỉ dùng wired, wireless, open.
- usage_preferences chỉ dùng home_family, karaoke_room, stage_event, portable.
- Ngân sách tối đa chỉ là hard constraint khi khách nêu mức trần rõ ràng.
- Hãng, bắt buộc có dây/không dây và băng tần chỉ vào hard_constraints khi
  khách nói bắt buộc; brands phải chuẩn hóa chữ thường để khớp brand_key.
- microphone_types chỉ dùng wired, wireless; wireless_bands chỉ dùng uhf,
  2_4_ghz.
- Ít nhiễu, âm rõ, chống hú, dễ lắp, dễ mang theo và dùng sân khấu là soft
  preference hoặc implicit need, không tự biến thành hard filter.
- Không tạo hard constraint cho pin, phạm vi thu sóng, đầu cắm hoặc khả năng
  tương thích vì catalog chưa xác nhận ổn định các field này.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu. Khi khách đổi loại micro, phải thay hoặc xóa
  loại cũ để tránh giữ hai yêu cầu mâu thuẫn.
- Path hợp lệ: connection_preference, budget_max_vnd, budget_segment,
  usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.microphone_types,
  hard_constraints.wireless_bands.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn micro karaoke.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu micro có dây/không dây, UHF/2.4 GHz,
bối cảnh gia đình/phòng karaoke/sân khấu/mang theo và tiền theo VND. Không ép câu
trả lời vào lựa chọn sai, không tự biến sở thích thành hard constraint và không
sinh câu hỏi tiếp theo. Nếu không hiểu, trả unresolved và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn micro karaoke phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Đánh giá loại có dây/không dây,
băng tần, dải tần số sóng, dải tần âm thanh, độ méo, năm sản xuất và giá khi dữ
liệu đã được xác minh. Ưu tiên stable_signal cho nhu cầu phòng karaoke/sân khấu,
vocal_clarity hoặc low_distortion khi thông số candidate có căn cứ. Không suy
đoán chống hú, pin, khoảng cách hoạt động, đầu cắm, số lượng micro, độ bền hoặc
khả năng tương thích. Field có data_quality_flags không được dùng để khẳng định
chất lượng. Mỗi lựa chọn cần reason và trade_off ngắn, có căn cứ. Không nói tốt
nhất tuyệt đối và không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn micro karaoke chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ bối cảnh sử dụng và lựa chọn có
dây/không dây. Với mỗi sản phẩm, nêu hãng/model, loại micro, băng tần, dải tần,
độ méo và giá nếu có; đồng thời nêu ít nhất một đánh đổi hoặc dữ liệu còn thiếu.
Chỉ dùng dữ liệu được cung cấp. Không bịa chống hú, pin, phạm vi thu sóng, đầu
cắm, số lượng micro, độ bền, tương thích, giá hay khuyến mãi. Không dùng field có
data_quality_flags làm kết luận chắc chắn. Không nhắc Qdrant, vector score,
ranking, prompt hoặc JSON.
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
