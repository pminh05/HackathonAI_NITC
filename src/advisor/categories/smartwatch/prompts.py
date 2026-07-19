"""Prompt builders for the smartwatch advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua đồng hồ thông minh.

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
- usage_preferences chỉ dùng health_monitoring, fitness_sports,
  outdoor_navigation, calls_notifications, children_safety, everyday_style.
- phone_platform chỉ dùng ios, android, flexible. iPhone nghĩa là ios.
- Ngân sách tối đa chỉ là hard constraint khi khách nêu mức trần rõ ràng.
- Hãng, loại màn hình/dây, kích thước, khối lượng, vòng cổ tay, thời lượng pin,
  ATM, kiểu cuộc gọi và capability chỉ vào hard_constraints khi khách nói rõ
  “cần”, “bắt buộc”, “ít nhất” hoặc “tối đa”.
- call_requirement chỉ dùng on_wrist hoặc standalone. Nghe gọi độc lập khác
  nghe gọi trên đồng hồ qua điện thoại.
- requires_cellular, requires_gps, requires_notifications, requires_swimming
  chỉ là true/false khi khách nêu yêu cầu rõ; “ưu tiên” phải vào soft_preferences.
- required_health_features chỉ dùng heart_rate, spo2, sleep, stress, ecg,
  blood_pressure, step_count, menstrual_cycle, vo2_max, body_composition.
- “Dùng khi bơi” có thể đặt requires_swimming=true; “chống nước” chung chung là
  sở thích nếu khách không nêu ATM hay tình huống bắt buộc.
- Không suy ra thời lượng sử dụng từ mAh, không quy đổi IP sang ATM và không coi
  chỉ số sức khỏe là chẩn đoán y tế.
- display_families chỉ dùng amoled_oled, mip, tft_lcd, ips_lcd, lcd, other.
- strap_material_families chỉ dùng silicone, rubber_tpu, leather, fabric_nylon,
  metal, titanium, composite, other.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: budget_max_vnd, budget_segment, phone_platform,
  usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.display_families,
  hard_constraints.strap_material_families,
  hard_constraints.min_screen_size_inch,
  hard_constraints.max_screen_size_inch, hard_constraints.max_case_width_mm,
  hard_constraints.max_weight_g, hard_constraints.wrist_circumference_cm,
  hard_constraints.min_typical_battery_hours,
  hard_constraints.min_water_resistance_atm,
  hard_constraints.call_requirement, hard_constraints.requires_cellular,
  hard_constraints.requires_gps, hard_constraints.requires_notifications,
  hard_constraints.requires_swimming,
  hard_constraints.required_health_features.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn đồng hồ thông minh.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu tiền theo VND, kích thước mặt theo
mm, màn hình theo inch, khối lượng theo gram, vòng cổ tay theo cm, pin theo giờ
hoặc ngày và chống nước theo ATM. Phân biệt iOS/Android, cuộc gọi qua điện thoại
với cuộc gọi độc lập. Không ép câu trả lời vào lựa chọn sai, không tự biến sở
thích thành hard constraint và không sinh câu hỏi tiếp theo. Nếu không hiểu,
trả unresolved và giữ raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn đồng hồ thông minh phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra hệ điện thoại,
giá hiệu lực và các capability bắt buộc; sau đó đánh giá mục đích sử dụng, kiểu
cuộc gọi, SIM/eSIM, GPS, thông báo, sức khỏe và luyện tập, pin theo đúng chế độ,
màn hình, kích thước đeo, khối lượng và chống nước. Với đồng hồ trẻ em, kiểm tra
riêng cuộc gọi độc lập, GPS và dữ liệu SOS. Không suy ra thời lượng từ mAh, không
quy đổi IP sang ATM, không nâng mô tả sức khỏe thành chẩn đoán y tế và không xác
nhận tính năng khi candidate thiếu dữ liệu. Mỗi lựa chọn cần reason và trade_off
ngắn, có căn cứ; ưu tiên các hướng khác nhau. Không nói tốt nhất tuyệt đối và
không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn đồng hồ thông minh chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ mục đích sử dụng, hệ điện thoại
và ngân sách. Với mỗi lựa chọn, nêu lý do phù hợp, giá nếu có, các capability đã
được kiểm chứng và ít nhất một đánh đổi như pin theo chế độ, kích thước/khối
lượng, kiểu cuộc gọi, tương thích, chống nước hoặc dữ liệu còn thiếu. So sánh
ngắn và kết luận theo ưu tiên của khách. Chỉ dùng dữ liệu được cung cấp; không
bịa giá, GPS, SIM, cảm biến, khả năng bơi, thời lượng pin hay khuyến mãi. Nếu
thiếu phải nói rõ. Không nhắc Qdrant, vector score, ranking, prompt hoặc JSON.
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
