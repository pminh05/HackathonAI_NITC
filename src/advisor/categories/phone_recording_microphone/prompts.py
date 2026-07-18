"""Prompt builders for the recording-microphone advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua micro thu âm.

Tin nhắn hiện tại:
{user_query}

Hồ sơ nhu cầu đã biết:
{current_need_profile}

Mục đích lượt nói: {turn_action}
Các câu đang chờ làm rõ: {pending_questions}
Lịch sử gần nhất: {conversation_history}

Chỉ trả về ProfilePatch mô tả phần thay đổi so với hồ sơ hiện tại.

Quy tắc:
- Chỉ ghi nhận điều khách nói hoặc suy luận trực tiếp có bằng chứng.
- recording_setup chỉ dùng iphone_lightning, iphone_usb_c, android_usb_c,
  camera_3_5mm, computer_usb, open. Không suy cổng từ model điện thoại.
- usage_preferences chỉ dùng solo_content, two_person_interview,
  outdoor_mobile, podcast_livestream.
- Ngân sách tối đa chỉ là hard constraint khi khách nêu mức trần rõ ràng.
- Hãng, loại micro, thiết bị, cổng, số bộ phát, thời lượng, phạm vi, hướng thu,
  băng tần và tính năng chỉ vào hard_constraints khi khách nói bắt buộc, ít
  nhất hoặc phải có. Brands phải chuẩn hóa chữ thường để khớp brand_key.
- product_types chỉ dùng wireless_recording, podcast_livestream;
  required_compatibility_tags chỉ dùng ios, ipados, android, camera, macos,
  windows, playstation; connector_types chỉ dùng lightning, usb_c, 3_5mm,
  xlr, micro_usb, aux_in.
- pickup_patterns chỉ dùng omnidirectional, supercardioid; wireless_bands chỉ
  dùng 2_4_ghz. Không tự quy đổi hay suy băng tần.
- Lọc ồn, âm rõ, dễ cài, gọn nhẹ, ít nhiễu, dùng ngoài trời và monitoring là
  soft preference nếu khách chỉ nói ưu tiên.
- Câu “pin 8 giờ” không tự biến thành tối thiểu 8 giờ; chỉ đặt
  min_runtime_hours khi khách nói “ít nhất/từ 8 giờ”. Quy tắc tương tự cho phạm vi.
- Không tự điền thông tin còn thiếu; evidence trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: recording_setup, budget_max_vnd, budget_segment,
  usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.product_types,
  hard_constraints.required_compatibility_tags, hard_constraints.connector_types,
  hard_constraints.min_transmitter_count, hard_constraints.min_runtime_hours,
  hard_constraints.min_transmission_range_m, hard_constraints.pickup_patterns,
  hard_constraints.wireless_bands, hard_constraints.required_features.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn micro thu âm.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu iPhone/iPad/Android/camera/máy tính,
Lightning/USB-C/3.5 mm/XLR, số người cần thu đồng thời và số tiền VND. Không suy
cổng từ model thiết bị, không ép vào lựa chọn sai, không tự biến sở thích thành
hard constraint và không sinh câu hỏi tiếp theo. Nếu không hiểu, trả unresolved
và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn micro thu âm phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra thiết bị và
cổng tương thích, số bộ phát và giá hiệu lực; sau đó cân nhắc bối cảnh solo,
phỏng vấn, ngoài trời hay podcast, cùng thời lượng, phạm vi, hướng thu, băng tần,
dải âm, SPL, khối lượng và tính năng khi candidate có dữ liệu xác minh. Field có
data_quality_flags không được dùng làm khẳng định chắc chắn. Không suy đoán khả
năng chống ồn, độ trễ, chất lượng âm, pin, phạm vi, cổng hoặc tương thích. Mỗi
lựa chọn cần reason và trade_off ngắn, có căn cứ; ưu tiên các hướng khác nhau.
Không nói tốt nhất tuyệt đối và không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn micro thu âm chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ thiết bị, cổng kết nối và bối
cảnh thu âm. Với mỗi lựa chọn, nêu hãng/model, giá nếu có, khả năng tương thích,
số bộ phát nếu có dữ liệu, lý do phù hợp và ít nhất một đánh đổi hoặc dữ liệu
còn thiếu. So sánh ngắn rồi kết luận theo ưu tiên của khách. Chỉ dùng dữ liệu
được cung cấp; không bịa chống ồn, chất lượng âm, độ trễ, pin, phạm vi, cổng,
tương thích, giá hay khuyến mãi. Không dùng field có data_quality_flags làm kết
luận chắc chắn. Không nhắc Qdrant, vector score, ranking, prompt hoặc JSON và
không nói sản phẩm nào tốt nhất tuyệt đối.
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
