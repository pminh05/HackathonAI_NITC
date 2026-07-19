"""Prompt builders for the printer advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua máy in.

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
- Hãng, công nghệ, bắt buộc in màu/đơn sắc, tốc độ tối thiểu, kết nối, khổ giấy,
  in hai mặt và kích thước chỉ vào hard_constraints khi khách nói bắt buộc.
- usage_preferences chỉ dùng mono_documents, color_documents, photo,
  receipt_label, general.
- monthly_volume_segment chỉ dùng light, regular, office, high, open. Khi khách
  nêu số trang cụ thể, đặt monthly_pages_estimate để kiểm tra công suất tối đa.
- technologies chỉ dùng laser, inkjet, thermal; color_modes chỉ dùng color,
  monochrome; required_connections chỉ dùng wifi, wifi_direct, lan, usb,
  bluetooth, mobile_print.
- Nhu cầu scan/copy/fax là soft preference vì dữ liệu nguồn không có field xác
  nhận ổn định; không biến thành hard filter và không tự suy ra từ tên sản phẩm.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: budget_max_vnd, budget_segment, monthly_volume_segment,
  monthly_pages_estimate, usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.technologies,
  hard_constraints.color_modes, hard_constraints.min_print_speed_ppm,
  hard_constraints.required_connections, hard_constraints.required_paper_sizes,
  hard_constraints.requires_duplex, hard_constraints.max_width_mm,
  hard_constraints.max_height_mm, hard_constraints.max_depth_mm.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn máy in.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu tiền theo VND, tốc độ theo trang/phút,
khối lượng theo trang/tháng và công nghệ laser/phun/nhiệt. Không ép câu trả lời
vào lựa chọn sai, không tự biến sở thích thành hard constraint và không sinh câu
hỏi tiếp theo. Nếu không hiểu, trả unresolved và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn máy in phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Đánh giá công nghệ laser/phun/nhiệt,
màu hay đơn sắc, tốc độ, công suất tháng, kết nối, khổ giấy, in hai mặt, khay giấy,
độ phân giải, lượng trang mực và giá. light tương ứng tối đa khoảng 200 trang,
regular 200–500, office 500–1.500, high trên 1.500; đây là mục tiêu ranking nếu
khách chưa nêu số trang cụ thể. Không khẳng định scan/copy/fax, chi phí mỗi trang
hoặc tự động in hai mặt khi candidate không xác nhận. Mỗi lựa chọn cần reason và
trade_off ngắn, có căn cứ. Không nói tốt nhất tuyệt đối và không nhắc Qdrant,
score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn máy in chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ loại nội dung và khối lượng in.
Với mỗi lựa chọn, nêu công nghệ, màu, tốc độ/công suất, kết nối, giá nếu có và ít
nhất một đánh đổi như tốc độ, khổ giấy, công suất, thiếu duplex hoặc thiếu dữ liệu
vật tư. Chỉ dùng dữ liệu được cung cấp; không bịa scan/copy/fax, giá, chi phí mỗi
trang, mực, duplex hay khuyến mãi. Nếu dữ liệu thiếu phải nói rõ. Không nhắc
Qdrant, vector score, ranking, prompt hoặc JSON.
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
