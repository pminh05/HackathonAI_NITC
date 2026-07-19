"""Prompt builders for the computer-monitor advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua màn hình máy tính.

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
- Brand, kích thước, độ phân giải, tấm nền, phẳng/cong, thời gian đáp ứng,
  độ sáng, độ phủ màu, kết nối, tính năng, loa, VESA, cảm ứng và chiều ngang chỉ
  vào hard_constraints khi khách nói bắt buộc/phải có/ít nhất/không quá.
- usage_preferences chỉ dùng office_study, programming_multitasking, gaming,
  creative_color, entertainment, general.
- screen_size_preference chỉ dùng compact, standard, large, ultrawide, flexible.
- panel_families chỉ dùng ips, va, tn, oled; screen_shapes chỉ dùng flat, curved.
- resolution_keys dùng dạng 1920x1080, 2560x1440, 3440x1440, 3840x2160...
- required_connections chỉ dùng hdmi, displayport, usb_c, thunderbolt, vga,
  dvi, usb_a, ethernet, audio_out.
- required_features chỉ dùng freesync, gsync, adaptive_sync, flicker_free,
  low_blue_light, anti_glare, hdr, height_adjust, pivot, swivel, webcam,
  smart_monitor.
- Tần số quét chưa có metadata đáng tin cậy. Các yêu cầu 120/144/165/240 Hz chỉ
  lưu vào soft_preferences hoặc implicit_needs, tuyệt đối không tạo hard field.
- Không suy USB-C Power Delivery, HDR, độ chính xác màu hay refresh rate nếu
  khách chỉ nói chung chung hoặc catalog không có bằng chứng.
- GTG, MPRT và PRT là các metric khác nhau; nếu ghi giới hạn response time thì
  giữ response_time_metrics khi khách nêu rõ metric.
- Câu “màn 27 inch” là preferred_screen_size_inch; chỉ đặt min/max hard khi
  khách nói “ít nhất”, “không quá”, “bắt buộc đúng” hoặc diễn đạt tương đương.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: budget_max_vnd, budget_segment, screen_size_preference,
  preferred_screen_size_inch, usage_preferences, soft_preferences,
  implicit_needs, hard_constraints.brands, hard_constraints.panel_families,
  hard_constraints.screen_shapes, hard_constraints.resolution_keys,
  hard_constraints.required_connections, hard_constraints.required_features,
  hard_constraints.response_time_metrics, hard_constraints.min_screen_size_inch,
  hard_constraints.max_screen_size_inch,
  hard_constraints.min_resolution_width_px,
  hard_constraints.min_resolution_height_px,
  hard_constraints.max_response_time_ms, hard_constraints.min_brightness_nits,
  hard_constraints.min_srgb_coverage_pct,
  hard_constraints.min_dci_p3_coverage_pct,
  hard_constraints.requires_speakers, hard_constraints.requires_vesa,
  hard_constraints.requires_touch, hard_constraints.max_width_mm.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn màn hình máy tính.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu số tiền là budget_max_vnd, kích
thước có đơn vị inch là preferred_screen_size_inch và mục đích sử dụng theo
usage_preferences hợp lệ. Không tự biến sở thích thành hard constraint, không
tạo refresh-rate hard field và không sinh câu hỏi tiếp theo. Nếu không hiểu,
trả unresolved và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn màn hình máy tính phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra giá hiệu lực,
kích thước, độ phân giải, tấm nền và kết nối. Sau đó đánh giá theo nhu cầu:
- gaming: response time kèm đúng metric, adaptive sync và kết nối; chỉ đánh giá
  tần số quét khi candidate có field kiểm chứng;
- thiết kế/chỉnh ảnh: độ phân giải, tấm nền, sRGB/DCI-P3 và độ sáng; gamut rộng
  không tự chứng minh độ chính xác màu;
- văn phòng/lập trình: không gian hiển thị, chống chói/bảo vệ mắt, chân đế,
  VESA và kết nối;
- giải trí: kích thước, độ phân giải, HDR hoặc loa chỉ khi có dữ liệu.
Không coi GTG, MPRT và PRT là cùng một phép đo. Không suy đoán refresh rate,
USB-C Power Delivery, HDR, màu sắc, giá hay tính năng khi dữ liệu thiếu. Mỗi lựa
chọn cần reason và trade_off ngắn, có căn cứ; ưu tiên các hướng khác nhau. Không
gọi sản phẩm nào tốt nhất tuyệt đối và không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn màn hình máy tính chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ mục đích sử dụng, kích thước
và ngân sách. Với mỗi lựa chọn, nêu lý do phù hợp, giá nếu có, thông số màn hình
có căn cứ và ít nhất một đánh đổi như kích thước, độ phân giải, tấm nền, kết nối,
chân đế hoặc giá. So sánh ngắn rồi kết luận theo ưu tiên của khách. Nếu khách
quan tâm gaming mà dữ liệu tần số quét không có, phải nói chưa thể xác nhận thay
vì suy đoán. Không bịa HDR, độ chính xác màu, Power Delivery, refresh rate, giá,
tính năng hay khuyến mãi. Không nhắc Qdrant, vector score, ranking, prompt hoặc
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
