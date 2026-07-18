"""Prompt builders for the water-heater advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn cập nhật nhu cầu khách hàng mua máy nước nóng.

Tin nhắn hiện tại:
{user_query}

Hồ sơ nhu cầu đã biết:
{current_need_profile}

Mục đích lượt nói: {turn_action}
Các câu đang chờ làm rõ: {pending_questions}
Lịch sử gần nhất: {conversation_history}

Chỉ trả về ProfilePatch mô tả phần thay đổi so với hồ sơ hiện tại.

Quy tắc:
- Chỉ ghi nhận điều khách nói hoặc suy luận trực tiếp có evidence.
- Ngân sách chỉ là hard constraint khi khách nêu mức trần rõ ràng.
- product_types chỉ dùng direct, indirect, solar, direct_multipoint.
- water_supply chỉ dùng stable, low_pressure, multi_outlet, open.
- Nước yếu chỉ tạo water_supply=low_pressure và soft preference; không tự suy ra
  bắt buộc có bơm. Chỉ thêm booster_pump vào required_features khi khách nói rõ
  phải có bơm trợ lực.
- Hãng, loại máy, dung tích tối thiểu/tối đa, công suất tối đa, thời gian làm
  nóng tối đa, kích thước, phụ kiện, IP và an toàn chỉ vào hard_constraints khi
  khách nói bắt buộc, ít nhất hoặc không quá.
- required_features chỉ dùng booster_pump, included_shower.
- required_safety_features chỉ dùng elcb, rcd, overheat_cutoff, flow_sensor,
  pressure_relief_valve, waterproof, anti_scald, thermal_stabilizer.
- Không suy ra khả năng chịu nước yếu từ công suất, bơm, hay áp lực còn thiếu.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: household_size, budget_max_vnd, budget_segment, water_supply,
  usage_preferences, soft_preferences, implicit_needs, hard_constraints.brands,
  hard_constraints.product_types, hard_constraints.min_capacity_lit,
  hard_constraints.max_capacity_lit, hard_constraints.max_power_w,
  hard_constraints.max_heating_time_minutes, hard_constraints.max_width_cm,
  hard_constraints.max_height_cm, hard_constraints.max_depth_cm,
  hard_constraints.required_features, hard_constraints.required_safety_features,
  hard_constraints.ip_ratings.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn máy nước nóng.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu máy trực tiếp, gián tiếp, năng lượng
mặt trời hoặc trực tiếp đa điểm theo product_types; tiền theo VND; dung tích theo
lít; công suất theo W. Nước yếu không tự động trở thành yêu cầu bắt buộc có bơm.
Không ép câu trả lời vào lựa chọn sai, không biến sở thích thành hard constraint
và không sinh câu hỏi tiếp theo. Nếu không hiểu, trả unresolved và giữ nguyên
raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn máy nước nóng phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra loại làm nóng,
giá hiệu lực và ràng buộc lắp đặt. Sau đó cân nhắc dung tích, công suất, thời gian
làm nóng, điều kiện nguồn nước, bơm trợ lực, phụ kiện, IP và dữ liệu an toàn.
Nhu cầu nước yếu chỉ được xem là đáp ứng khi candidate có bơm hoặc mô tả xác nhận
khả năng hoạt động với áp lực thấp; không suy đoán từ công suất. Số người chỉ là
tham khảo ranking vì metadata có thể thiếu. Không suy đoán giá, an toàn, phụ kiện,
khả năng chịu áp lực hay hiệu quả tiết kiệm khi dữ liệu thiếu. Mỗi lựa chọn cần
reason và trade_off ngắn, có căn cứ; ưu tiên các hướng khác nhau. Không gọi sản
phẩm nào tốt nhất tuyệt đối và không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn máy nước nóng chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ kiểu làm nóng, điều kiện nguồn
nước và nhu cầu thực tế. Với mỗi lựa chọn, nêu lý do phù hợp, giá nếu có, dung
tích/công suất/thời gian làm nóng liên quan, bơm hoặc vòi sen, dữ liệu an toàn/IP
và ít nhất một đánh đổi như thời gian chờ, kích thước, công suất điện, giá hoặc
dữ liệu còn thiếu. Chỉ dùng dữ liệu được cung cấp; không bịa giá, an toàn, áp lực,
phụ kiện, tính năng hay khuyến mãi. Nếu dữ liệu thiếu phải nói rõ. Nhắc người dùng
kiểm tra nguồn điện, đường nước và điều kiện lắp đặt thực tế với kỹ thuật viên.
Không nhắc Qdrant, vector score, ranking, prompt hoặc JSON và không nói tốt nhất
tuyệt đối.
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
