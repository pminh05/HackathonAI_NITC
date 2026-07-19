"""Prompt builders for the dishwasher advisory flow."""

from __future__ import annotations

import json
from typing import Any


NEED_EXTRACTION_PROMPT = """Bạn là bộ phận cập nhật nhu cầu khách hàng mua máy rửa chén.

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
- Hãng, loại lắp đặt, sức chứa tối thiểu, kích thước tối đa, độ ồn tối đa và
  lượng nước tối đa chỉ vào hard_constraints khi khách nói bắt buộc, ít nhất
  hoặc không quá. Một mức sức chứa mục tiêu thông thường nằm ở capacity_segment.
- product_types chỉ dùng freestanding, built_in, semi_integrated, mini.
- capacity_segment chỉ dùng compact, standard, large, open.
- usage_preferences chỉ dùng quick_cycle, pots_and_pans, glass_care,
  hygiene_care, quiet_night, water_saving, drying_performance, smart_control,
  flexible_racks.
- Rửa nhanh, khử khuẩn, tự hé cửa, chống rò rỉ, khóa trẻ em, điều khiển ứng
  dụng, rửa bán tải và khay linh hoạt là soft preference vì metadata hiện là
  mô tả tự do, không được biến thành hard filter.
- Công suất W không phải điện năng tiêu thụ; không suy ra tiết kiệm điện từ đó.
- Không tự điền thông tin còn thiếu; evidence phải trích ngắn gọn lời khách.
- Dùng set cho scalar; replace cho toàn bộ list; add/remove cho từng phần; clear
  khi khách chủ động bỏ yêu cầu.
- Path hợp lệ: budget_max_vnd, budget_segment, capacity_segment,
  usage_preferences, soft_preferences, implicit_needs,
  hard_constraints.brands, hard_constraints.product_types,
  hard_constraints.min_place_settings, hard_constraints.min_vietnamese_meals,
  hard_constraints.max_width_cm, hard_constraints.max_height_cm,
  hard_constraints.max_depth_cm, hard_constraints.max_noise_db,
  hard_constraints.max_water_l_per_cycle.
"""


CUSTOM_ANSWER_PROMPT = """Bạn diễn giải câu trả lời tự do trong tư vấn máy rửa chén.

Câu hỏi: {question}
Loại thông tin: {question_id}
Các lựa chọn ban đầu: {available_options}
Câu trả lời tự do: {custom_answer}
Hồ sơ hiện tại: {current_need_profile}

Chỉ trích xuất điều khách thực sự nói. Hiểu loại máy độc lập, âm tủ, bán âm
hoặc mini theo product_types; tiền theo VND; sức chứa theo bộ châu Âu hoặc bữa
ăn Việt. Không ép câu trả lời vào lựa chọn sai, không tự biến sở thích thành
hard constraint và không sinh câu hỏi tiếp theo. Nếu không hiểu, trả unresolved
và giữ nguyên raw_answer.
"""


RANKING_PROMPT = """Bạn là bộ phận chọn máy rửa chén phù hợp.

Hồ sơ nhu cầu khách hàng:
{need_profile}

Hard constraints đã áp dụng:
{hard_constraints}

Danh sách candidate theo semantic relevance:
{candidates}

Chọn tối đa 3 product_id có trong candidate. Trước tiên kiểm tra loại lắp đặt,
sức chứa và giá hiệu lực, sau đó cân nhắc độ ồn, lượng nước, chương trình rửa,
công nghệ rửa/sấy, khay và tiện ích theo nhu cầu. compact tương ứng mục tiêu
6–8 bộ, standard 11–13 bộ, large từ 14 bộ; đây là mục tiêu ranking, không phải
hard filter nếu khách chưa nói “ít nhất”. Với nhu cầu vệ sinh, sấy, chống rò rỉ
hoặc điều khiển thông minh chỉ đánh giá khi candidate có dữ liệu xác nhận. Không
suy đoán giá, điện năng, tính năng hay hiệu quả sấy khi dữ liệu thiếu. Mỗi lựa
chọn cần reason và trade_off ngắn, có căn cứ; ưu tiên các hướng khác nhau. Không
gọi sản phẩm nào tốt nhất tuyệt đối và không nhắc Qdrant, score, prompt hoặc JSON.
"""


RESPONSE_PROMPT = """Bạn là nhân viên tư vấn máy rửa chén chuyên nghiệp và khách quan.

Tin nhắn hiện tại: {user_query}
Mục đích lượt nói: {turn_action}
Lịch sử gần nhất: {conversation_history}
Hồ sơ nhu cầu: {need_profile}
Các sản phẩm đã chọn và dữ liệu kiểm chứng: {selected_products}

Viết câu trả lời tiếng Việt tự nhiên. Bắt đầu từ kiểu lắp đặt, lượng chén đĩa và
nhu cầu thực tế. Với mỗi lựa chọn, nêu lý do phù hợp, giá nếu có, lợi ích và ít
nhất một đánh đổi như sức chứa, kích thước, độ ồn, lượng nước, giá hoặc dữ liệu
tính năng còn thiếu. So sánh ngắn rồi kết luận theo ưu tiên của khách. Chỉ dùng
dữ liệu được cung cấp; không bịa giá, điện năng, chương trình, công nghệ, tính
năng hay khuyến mãi. Nếu dữ liệu thiếu phải nói rõ. Không nhắc Qdrant, vector
score, ranking, prompt hoặc JSON và không nói tốt nhất tuyệt đối.
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
