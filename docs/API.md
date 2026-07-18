# Product Advisor API

Tài liệu này mô tả HTTP API tư vấn tủ lạnh, máy lạnh, máy giặt, máy sấy quần áo,
máy rửa chén, tủ mát, tủ đông, máy nước nóng, máy tính bảng và máy in. API dùng
FastAPI, lưu trạng thái hội thoại theo `thread_id` và trả kết quả chat bằng
Server-Sent Events (SSE).

## 1. Chạy dịch vụ

Yêu cầu các biến `GOOGLE_API_KEY`, `QDRANT_URL` và `QDRANT_API_KEY` đã được cấu hình trong `.env`.

```bash
cd /home/tuanta/VAIC/HackathonAI_NITC
PYTHONPATH=src .venv/bin/uvicorn advisor.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Các URL phục vụ phát triển:

- Base URL: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

MVP phải chạy với một worker. Khóa chống chạy đồng thời trên cùng `thread_id` hiện được giữ trong bộ nhớ của process; checkpoint hội thoại được lưu trong SQLite.

## 2. Luồng hội thoại

```text
POST /chat
    │
    ├── Đã đủ nhu cầu ──> token* ──> completed
    │
    └── Còn thiếu thông tin ──> clarification_required
                                      │
                                      ▼
                         POST /chat/{thread_id}/resume
                              với toàn bộ quick options
                                      │
                                      └── token* ──> completed
```

Quy tắc quan trọng:

- Client phải lưu `thread_id` nhận từ event `session`.
- Khi nhận `clarification_required`, client hiển thị các câu hỏi multiple choice.
- Lệnh resume phải trả lời đầy đủ mọi câu đang chờ, mỗi câu đúng một lần.
- Khi thread đang chờ, client hoàn tất form rồi gọi `/resume`; `/chat` sẽ trả `409`.
- `question_id` và `option_id` phải được lấy từ form server trả về, không nên hard-code phía client.
- Nếu chọn option `other`, phải gửi thêm `custom_answer` khác rỗng.
- Sau khi thread hoàn tất, có thể gửi tin nhắn tiếp theo bằng `POST /chat` với `thread_id` cũ.

## 3. Server-Sent Events

Hai endpoint `POST /chat` và `POST /chat/{thread_id}/resume` trả:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
```

Mỗi event có dạng:

```text
event: token
data: {"delta":"nội dung mới"}

```

Danh sách event:

| Event | Ý nghĩa | Trường dữ liệu chính |
| --- | --- | --- |
| `session` | Event đầu tiên của mỗi stream | `thread_id`, `mode` |
| `progress` | Graph vừa hoàn thành một công đoạn | `stage` |
| `clarification_required` | Graph tạm dừng để chờ người dùng | `thread_id`, `message`, `questions` |
| `token` | Một phần nội dung trả lời của AI | `delta` |
| `completed` | Lượt xử lý đã hoàn tất | `thread_id`, `answer`, `selected_products` |
| `error` | Lỗi xảy ra sau khi stream đã mở | `code`, `message`, `retryable` |

Giá trị `session.mode`:

- `started`: tạo cuộc hội thoại mới.
- `continued`: gửi câu hỏi tiếp theo vào thread đã hoàn tất.
- `resumed`: tiếp tục graph sau khi người dùng trả lời form HITL.

Các `progress.stage` hiện có:

- `intent_detected`
- `need_extracted`
- `clarification_ready`
- `clarification_completed`
- `filter_built`
- `retrieval_completed`
- `ranking_completed`

Server có thể gửi comment `: heartbeat` nếu một công đoạn chạy lâu. Client SSE cần bỏ qua comment này. Nội dung hoàn chỉnh có thể được dựng bằng cách nối tất cả `token.delta`; event `completed.answer` là bản kết quả chuẩn cuối cùng.

Lưu ý: lỗi kiểm tra request xảy ra trước khi mở stream sẽ dùng HTTP status `4xx`. Lỗi trong lúc graph đang chạy được trả bằng event `error` trên một HTTP response đã có status `200`.

## 4. Endpoint

### `GET /healthz`

Kiểm tra process API đã sẵn sàng.

Response `200`:

```json
{
  "status": "ok"
}
```

`status` có thể là `starting` trong lúc ứng dụng chưa sẵn sàng.

### `POST /chat`

Tạo cuộc hội thoại mới hoặc gửi follow-up vào thread đã hoàn tất.

Request tạo thread mới:

```json
{
  "message": "Tôi muốn mua tủ lạnh cho gia đình"
}
```

Request tiếp tục thread cũ:

```json
{
  "message": "Trong ba sản phẩm trên, mẫu nào tiết kiệm điện nhất?",
  "thread_id": "35653af4-43c3-4893-abbe-98d4d08478de"
}
```

| Field | Kiểu | Bắt buộc | Ràng buộc |
| --- | --- | --- | --- |
| `message` | string | Có | Sau khi trim phải có nội dung, tối đa 10.000 ký tự |
| `thread_id` | UUID hoặc `null` | Không | Bỏ trống để server tạo UUID mới; nếu có thì thread phải tồn tại và không đang chạy |

Ví dụ stream yêu cầu làm rõ:

```text
event: session
data: {"thread_id":"35653af4-43c3-4893-abbe-98d4d08478de","mode":"started"}

event: progress
data: {"stage":"intent_detected"}

event: progress
data: {"stage":"need_extracted"}

event: clarification_required
data: {"thread_id":"35653af4-43c3-4893-abbe-98d4d08478de","questions":[...]}

```

Test bằng cURL:

```bash
curl -N http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Tôi muốn mua tủ lạnh cho gia đình"}'
```

### `POST /chat/{thread_id}/resume`

Gửi câu trả lời cho form HITL đang chờ và tiếp tục tìm kiếm, xếp hạng, tư vấn sản phẩm.

Path parameter:

| Field | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `thread_id` | UUID | UUID nhận từ event `session` hoặc `clarification_required` |

Request ví dụ:

```json
{
  "answers": [
    {
      "question_id": "household_size",
      "option_id": "three_four"
    },
    {
      "question_id": "budget",
      "option_id": "10m_20m"
    },
    {
      "question_id": "usage_preferences",
      "option_id": "energy_saving"
    }
  ]
}
```

| Field | Kiểu | Bắt buộc | Ràng buộc |
| --- | --- | --- | --- |
| `answers` | array | Có | Từ 1 đến 3 phần tử; phải khớp toàn bộ câu hỏi đang chờ |
| `answers[].question_id` | string | Có | Phải là ID có trong form đang chờ và không được lặp |
| `answers[].option_id` | string | Có | Phải thuộc câu hỏi tương ứng |
| `answers[].custom_answer` | string hoặc `null` | Khi chọn `other` | Nội dung tự do của người dùng |

Ví dụ chọn `other`:

```json
{
  "answers": [
    {
      "question_id": "household_size",
      "option_id": "other",
      "custom_answer": "Gia đình 7 người"
    },
    {
      "question_id": "budget",
      "option_id": "other",
      "custom_answer": "Tối đa 15 triệu đồng"
    },
    {
      "question_id": "usage_preferences",
      "option_id": "weekly_storage"
    }
  ]
}
```

Test bằng cURL:

```bash
curl -N http://127.0.0.1:8000/chat/35653af4-43c3-4893-abbe-98d4d08478de/resume \
  -H 'Content-Type: application/json' \
  -d '{
    "answers": [
      {"question_id":"household_size","option_id":"three_four"},
      {"question_id":"budget","option_id":"10m_20m"},
      {"question_id":"usage_preferences","option_id":"energy_saving"}
    ]
  }'
```

Ví dụ event kết thúc:

```text
event: token
data: {"delta":"Dựa trên nhu cầu của gia đình bạn, "}

event: token
data: {"delta":"tôi đề xuất ba mẫu sau..."}

event: completed
data: {"thread_id":"35653af4-43c3-4893-abbe-98d4d08478de","answer":"Dựa trên nhu cầu...","selected_products":[{"product_id":"...","reason":"...","trade_off":"..."}]}

```

### `GET /chat/{thread_id}`

Khôi phục trạng thái public của một cuộc hội thoại. Endpoint này hữu ích sau khi reload giao diện hoặc khi client mất kết nối SSE.

Response khi đang chờ HITL:

```json
{
  "thread_id": "35653af4-43c3-4893-abbe-98d4d08478de",
  "status": "waiting_for_clarification",
  "questions": [
    {
      "question_id": "household_size",
      "question_type": "explicit",
      "question": "Gia đình bạn có bao nhiêu người sử dụng tủ lạnh?",
      "options": [
        {"option_id": "one_two", "label": "1–2 người"},
        {"option_id": "three_four", "label": "3–4 người"},
        {"option_id": "other", "label": "Khác – tôi muốn nhập câu trả lời"}
      ]
    }
  ],
  "answer": null,
  "selected_products": []
}
```

Response khi hoàn tất:

```json
{
  "thread_id": "35653af4-43c3-4893-abbe-98d4d08478de",
  "status": "completed",
  "questions": [],
  "answer": "Dựa trên nhu cầu của gia đình bạn...",
  "selected_products": [
    {
      "product_id": "sku-123",
      "reason": "Phù hợp ngân sách và dung tích cho 3–4 người.",
      "trade_off": "Không có tính năng lấy nước ngoài."
    }
  ]
}
```

Giá trị `status` gồm `running`, `waiting_for_clarification` và `completed`.

## 5. Form câu hỏi theo category

Form thực tế do server trả về có thể chỉ chứa những trường còn thiếu. Catalog tủ lạnh hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `household_size` | `one_two`, `three_four`, `five`, `over_five`, `other` |
| `budget` | `under_10m`, `10m_20m`, `20m_30m`, `over_30m`, `other` |
| `usage_preferences` | `daily_shopping`, `weekly_storage`, `frozen_storage`, `energy_saving`, `other` |

Catalog máy lạnh hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `room_size` | `up_to_15`, `15_20`, `20_30`, `30_40`, `other` |
| `budget` | `under_10m`, `10m_15m`, `15m_25m`, `over_25m`, `other` |
| `usage_preferences` | `quiet_sleep`, `energy_saving`, `fast_cooling`, `air_quality`, `smart_control`, `heating`, `other` |

Catalog máy giặt hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `household_size` | `one_two`, `three_five`, `six_seven`, `over_seven`, `other` |
| `budget` | `under_8m`, `8m_12m`, `12m_20m`, `over_20m`, `other` |
| `usage_preferences` | `daily_laundry`, `bulky_items`, `hygiene_care`, `energy_saving`, `quick_wash`, `wash_and_dry`, `other` |

Catalog máy sấy quần áo hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `household_size` | `one_two`, `three_five`, `six_seven`, `over_seven`, `other` |
| `budget` | `under_10m`, `10m_15m`, `15m_20m`, `over_20m`, `other` |
| `usage_preferences` | `rainy_season`, `frequent_drying`, `bulky_items`, `delicate_care`, `energy_saving`, `quick_dry`, `other` |

Catalog máy rửa chén hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `installation` | `freestanding`, `built_in`, `semi_integrated`, `mini`, `flexible`, `other` |
| `capacity` | `compact`, `standard`, `large`, `open`, `other` |
| `budget` | `under_12m`, `12m_18m`, `18m_25m`, `over_25m`, `other` |

Catalog tủ mát, tủ đông hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `product_family` | `cooler`, `freezer`, `open`, `other` |
| `budget` | `under_8m`, `8m_15m`, `15m_30m`, `over_30m`, `other` |
| `usage_preferences` | `display_drinks`, `fresh_food_cooling`, `bulk_frozen_storage`, `commercial_storage`, `convertible_use`, `energy_saving`, `other` |

Catalog máy nước nóng hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `heater_type` | `direct`, `indirect`, `solar`, `direct_multipoint`, `flexible`, `other` |
| `water_supply` | `stable`, `low_pressure`, `multi_outlet`, `open`, `other` |
| `budget` | `under_3m`, `3m_5m`, `5m_9m`, `over_9m`, `other` |

Catalog máy tính bảng hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `primary_usage` | `study_work`, `entertainment`, `gaming`, `drawing_notes`, `children`, `general`, `other` |
| `budget` | `under_10m`, `10m_20m`, `20m_35m`, `over_35m`, `open`, `other` |
| `connectivity` | `wifi_only`, `cellular_4g`, `cellular_5g`, `flexible`, `other` |

Catalog máy in hỗ trợ:

| `question_id` | Các `option_id` hợp lệ |
| --- | --- |
| `print_purpose` | `mono_documents`, `color_documents`, `photo`, `receipt_label`, `general`, `other` |
| `monthly_volume` | `light`, `regular`, `office`, `high`, `open`, `other` |
| `budget` | `under_3m`, `3m_5m`, `5m_10m`, `over_10m`, `open`, `other` |

Client vẫn nên render theo `questions[].options` từ response thay vì phụ thuộc vào bảng này, vì catalog có thể thay đổi.

## 6. Mã lỗi

| HTTP status / event | Trường hợp |
| --- | --- |
| `404 Not Found` | `thread_id` không tồn tại |
| `409 Conflict` | Thread đang chạy, đang chờ HITL nhưng gọi `/chat`, chưa sẵn sàng, hoặc gọi resume khi không chờ HITL |
| `422 Unprocessable Entity` | Body sai schema, thiếu/thừa `question_id`, sai `option_id`, lặp câu hỏi hoặc thiếu `custom_answer` cho `other` |
| SSE `error.code=configuration_error` | Thiếu/sai cấu hình dịch vụ; không nên retry nguyên request cho đến khi server được sửa |
| SSE `error.code=service_error` | Lỗi runtime từ graph hoặc dịch vụ phụ thuộc; có thể retry |
| SSE `error.code=incomplete_run` | Graph dừng mà không tạo form hoặc kết quả cuối; có thể retry |

Ví dụ `422` khi chưa trả lời đủ form:

```json
{
  "detail": {
    "code": "invalid_answers",
    "message": "Answers must match every pending question exactly once.",
    "missing": ["budget", "usage_preferences"],
    "unexpected": []
  }
}
```

Ví dụ `422` khi option không hợp lệ:

```json
{
  "detail": {
    "code": "invalid_options",
    "message": "One or more option IDs do not belong to the pending form.",
    "invalid": [
      {"question_id": "budget", "option_id": "invalid_option"}
    ]
  }
}
```

## 7. Tích hợp client SSE

API chat dùng `POST`, trong khi `EventSource` chuẩn của browser chỉ hỗ trợ `GET`. Frontend nên dùng `fetch()` và đọc `response.body` dạng stream, hoặc dùng một thư viện SSE hỗ trợ POST.

Pseudo-code:

```javascript
const response = await fetch("http://127.0.0.1:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "Tư vấn tủ lạnh cho gia đình tôi" }),
});

if (!response.ok) {
  throw new Error(await response.text());
}

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  // Tách từng block bằng "\n\n", sau đó đọc hai dòng event/data.
}
```

Không được giả định một network chunk tương ứng với một SSE event: một event có thể bị chia qua nhiều chunk, hoặc một chunk có thể chứa nhiều event. Client cần giữ `buffer` cho đến khi gặp dấu phân cách `\n\n`.

Swagger UI phù hợp để kiểm tra schema và request. Tùy trình duyệt, Swagger có thể chỉ hiện response sau khi stream kết thúc; dùng `curl -N` để quan sát token theo thời gian thực.

## 8. Biến môi trường liên quan

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Không có | API key cho Gemini |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model backbone |
| `QDRANT_URL` | Không có | URL Qdrant |
| `QDRANT_API_KEY` | Không có | API key Qdrant |
| `QDRANT_TIMEOUT_SECONDS` | `60` | Timeout Qdrant |
| `CHECKPOINT_DB_PATH` | `.data/checkpoints.sqlite` | SQLite lưu LangGraph checkpoint |
| `API_HOST` | `127.0.0.1` | Host quy ước khi chạy server |
| `API_PORT` | `8000` | Port quy ước khi chạy server |
| `API_CORS_ORIGINS` | localhost ports 3000, 5173 | Danh sách origin, phân cách bằng dấu phẩy |
| `SSE_HEARTBEAT_SECONDS` | `15` | Chu kỳ heartbeat của stream |

Hiện chưa có authentication/authorization. Không expose API MVP trực tiếp ra Internet trước khi bổ sung auth, rate limiting và cấu hình CORS phù hợp.
