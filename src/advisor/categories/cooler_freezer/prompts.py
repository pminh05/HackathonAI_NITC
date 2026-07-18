"""Prompt builders for the cooler/freezer advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn cập nhật nhu cầu khách hàng mua tủ mát hoặc tủ đông.

Tin nhắn hiện tại:
{user_query}

Hồ sơ nhu cầu đã biết:
{current_need_profile}

Mục đích lượt nói: {turn_action}
Các câu đang chờ làm rõ: {pending_questions}
Lịch sử gần nhất: {conversation_history}

Chỉ trả về ProfilePatch mô tả phần thay đổi so với hồ sơ hiện tại.

Quy tắc:
- product_family chỉ dùng cooler, freezer hoặc open. Tủ mát là cooler, tủ đông
  là freezer; từ "mini" cập nhật hard_constraints.size_variants=[mini].
- Ngân sách tối đa chỉ là hard constraint khi khách nêu mức trần rõ ràng.
- Brand, mini/standard, dung tích, kích thước, nhiệt độ, inverter, gas và tính
  năng chỉ vào hard_constraints khi khách nói bắt buộc/phải có/ít nhất/không quá.
- required_features chỉ dùng glass_door, convertible_mode, fast_freeze, lock,
  wheels, external_temperature_control, led_light, drain.
- usage_preferences chỉ dùng display_drinks, fresh_food_cooling,
  bulk_frozen_storage, commercial_storage, convertible_use, energy_saving.
- Cửa kính, độ êm, tiết kiệm điện, sức chứa lớn và chuyển đổi là soft preference
  nếu khách chỉ nói thích hoặc ưu tiên.
- Với yêu cầu "phải đạt -24°C", đặt required_temperature_c=-24; không đổi dấu.
- Không tự điền dữ liệu còn thiếu. Evidence trích ngắn gọn lời khách làm căn cứ.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: product_family, budget_max_vnd, budget_segment,
  usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.size_variants,
  hard_constraints.min_capacity_lit, hard_constraints.max_capacity_lit,
  hard_constraints.required_temperature_c, hard_constraints.max_width_cm,
  hard_constraints.max_height_cm, hard_constraints.max_depth_cm,
  hard_constraints.inverter, hard_constraints.gas_types,
  hard_constraints.required_features.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn tủ mát, tủ đông.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu rõ tủ mát/cooler, tủ đông/freezer,
giá VND, dung tích lít, kích thước cm và nhiệt độ âm °C. Không ép câu trả lời
vào lựa chọn sai, không biến sở thích thành hard constraint và không sinh câu
hỏi tiếp theo. Nếu không hiểu, trả unresolved và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn tủ mát hoặc tủ đông phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra đúng family,
giá hiệu lực, dung tích và kích thước. Với tủ mát, cân nhắc cửa kính, đèn LED,
làm lạnh đều, độ ồn và nhu cầu trưng bày. Với tủ đông, cân nhắc nhiệt độ thấp
nhất, số ngăn, chuyển đổi mát/đông, làm đông nhanh, lỗ thoát nước và bánh xe.
Sau đó đánh giá inverter, điện năng, gas, công nghệ và tiện ích theo nhu cầu.
Không suy đoán giá, nhiệt độ, điện năng hoặc tính năng khi dữ liệu thiếu. Mỗi
lựa chọn cần reason và trade_off ngắn, có căn cứ; ưu tiên các hướng khác nhau.
Không gọi sản phẩm nào tốt nhất tuyệt đối và không nhắc Qdrant, score, prompt
hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn tủ mát, tủ đông chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Nêu rõ mỗi lựa chọn là tủ mát hay tủ đông.
Với mỗi sản phẩm, giải thích sự phù hợp, giá nếu có, dung tích, lợi ích thực tế
và ít nhất một đánh đổi có căn cứ. Tủ mát tập trung vào trưng bày/làm mát; tủ
đông tập trung vào trữ đông/nhiệt độ/số ngăn. So sánh ngắn rồi kết luận theo
ưu tiên của khách. Chỉ dùng dữ liệu được cung cấp; không bịa giá, nhiệt độ,
điện năng, công nghệ, tính năng hay khuyến mãi. Nếu thiếu dữ liệu phải nói rõ.
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
