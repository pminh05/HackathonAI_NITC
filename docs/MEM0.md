# Long-term memory với Mem0

Tài liệu này giải thích cách Product Advisor dùng Mem0 Platform để cá nhân hóa
giữa nhiều phiên hội thoại. Phần HTTP contract chi tiết nằm tại
[API.md](API.md); tài liệu này tập trung vào kiến trúc, dữ liệu, quy tắc an toàn
và cách vận hành.

## 1. Mem0 giải quyết vấn đề gì?

LangGraph checkpoint và Mem0 có hai trách nhiệm khác nhau:

| Thành phần | Khóa định danh | Phạm vi | Ví dụ dữ liệu |
| --- | --- | --- | --- |
| LangGraph checkpoint | `thread_id` | Trạng thái làm việc của một cuộc hội thoại | câu hỏi HITL đang chờ, hồ sơ nhu cầu, sản phẩm vừa xếp hạng |
| Mem0 | Supabase `user_id` | Fact cá nhân hóa dùng lại giữa nhiều thread | tên gọi, quy mô gia đình, ưu tiên tiết kiệm điện, ngân sách theo ngành hàng |

Mem0 không thay thế checkpoint và không lưu toàn bộ state của graph. Hệ thống chỉ
gửi lượt user/assistant đã hoàn tất cho Mem0 để dịch vụ suy luận các fact hữu ích.
Khi người dùng mở thread mới, các fact liên quan được tìm lại theo ngữ nghĩa.

Memory chỉ được bật khi đồng thời thỏa mãn:

1. `MEMORY_ENABLED=true` ở backend.
2. Mem0 và Supabase đã được cấu hình đầy đủ.
3. Request mang Supabase access token hợp lệ.

Request anonymous vẫn dùng được luồng tư vấn bình thường nhưng không đọc hoặc ghi
Mem0.

## 2. Kiến trúc tổng quan

```text
Authorization: Bearer <Supabase access token>
                    │
                    ▼
          Supabase xác minh người dùng
                    │ verified user UUID
                    ▼
Tin nhắn mới ──> recall Mem0 theo user_id ──> guardrail
                    │
                    ├── style/tên gọi ─────────────> dùng khi soạn câu trả lời
                    │
                    ├── tương tác/feedback ────────> context diễn đạt, không ranking
                    │
                    └── fact ảnh hưởng quyết định ─> schema projection
                                                     │
                              tối đa 3 candidates <──┘
                                      │
                         HITL: use / edit / ignore
                                      │
                                      ▼
                              working profile
                                      │
                         retrieval → ranking → answer
                                      │
                                      ▼
                         queue Mem0 inferred write
                                      │ event_id
                                      ▼
                       poll trạng thái memory-write
```

Các node liên quan trong graph:

```text
analyze_turn
    → recall_memory
    → extract_need
    → project_memory
    → collect_memory_confirmation (chỉ khi có candidate)
    → generate_clarification
    → retrieval/ranking/response
    → queue_memory_write
```

## 3. Dữ liệu nào được ghi nhớ?

Backend gửi `custom_instructions` và `custom_categories` khi gọi Mem0 add. Prompt
hiện tại có version `product-advisor-memory-v3-vi`; version này cũng được gắn vào
metadata để truy vết khi chính sách trích xuất thay đổi.

| Category Mem0 | Nội dung |
| --- | --- |
| `identity_style` | Tên gọi, độ dài và giọng điệu câu trả lời mong muốn |
| `household_context` | Quy mô gia đình, bối cảnh sinh hoạt, thói quen, hạn chế không gian |
| `shopping_preference` | Ưu tiên mua sắm dùng chung như tiết kiệm điện, vận hành êm, thương hiệu thích hoặc tránh |
| `category_need` | Ngân sách và ràng buộc có ghi rõ ngành hàng |
| `product_interaction` | Sản phẩm đã được giới thiệu, xem xét, so sánh hoặc loại bỏ |
| `feedback` | Lý do người dùng thích, không thích hoặc loại bỏ sản phẩm |

Nguyên tắc trích xuất:

- Chỉ ghi thông tin người dùng trực tiếp nói hoặc xác nhận.
- Không biến đề xuất của trợ lý thành sở thích hay quyết định mua của người dùng.
- Không suy luận rằng người dùng đã mua sản phẩm nếu họ chưa xác nhận.
- Memory về ngân sách phải nêu rõ ngành hàng.
- Mỗi fact chỉ nên có một ý chính, ngắn gọn và tự hiểu được độc lập.
- Nội dung memory được viết bằng tiếng Việt; tên riêng, thương hiệu, mã sản phẩm
  và đơn vị đo được giữ nguyên.
- Không lưu mật khẩu, token, API key, dữ liệu thanh toán, địa chỉ chính xác, thông
  tin liên hệ hoặc prompt/hướng dẫn nội bộ.

Ví dụ:

```text
“Nhà tôi có 4 người.”
→ “Gia đình người dùng có 4 người.”

“Tôi cần máy giặt dưới 12 triệu.”
→ “Ngân sách mua máy giặt của người dùng tối đa là 12 triệu đồng.”

“Hãy gọi tôi là Minh và trả lời ngắn gọn.”
→ “Người dùng muốn được gọi là Minh.”
→ “Người dùng muốn câu trả lời ngắn gọn.”
```

## 4. Luồng đọc memory

### 4.1 Search theo người dùng

Ở mỗi lượt, backend tạo một semantic query từ ngành hàng đang active và yêu cầu
mới, sau đó gọi:

```http
POST /v3/memories/search/
Authorization: Token <MEM0_API_KEY>
Content-Type: application/json
```

Payload nội bộ có dạng:

```json
{
  "query": "Ngữ cảnh tư vấn hiện tại: máy giặt...",
  "filters": {"user_id": "verified-supabase-user-uuid"},
  "top_k": 10,
  "threshold": 0.2,
  "rerank": false
}
```

`user_id` luôn do backend lấy từ Supabase token đã xác minh. Client không thể đổi
filter này.

### 4.2 Guardrail và phân luồng sử dụng

Mọi fact recall được xem là natural-language data không đáng tin cậy và phải qua
guardrail chống prompt injection. Fact bị chặn được đưa vào
`blocked_memory_ids` và không đi tiếp vào prompt hoặc profile.

Fact an toàn được sử dụng theo ba mức:

1. `identity_style`: tên gọi, độ dài và tone được áp dụng khi compose response;
   không làm thay đổi quyết định chọn sản phẩm.
2. `product_interaction` và `feedback`: tối đa ba fact được đưa vào context để
   câu trả lời tự nhiên hơn; chúng không được dùng làm filter, điều kiện loại hay
   tín hiệu ranking.
3. Fact có thể ảnh hưởng quyết định: chỉ được chuyển thành candidate theo schema
   và phải qua xác nhận HITL.

## 5. Projection và xác nhận HITL

`projection.py` không đưa nguyên văn memory vào hồ sơ. Nó chỉ nhận diện một số
pattern đã kiểm soát, tạo `ProfilePatch`, giới hạn patch vào các path do câu hỏi
của `CategorySpec` sở hữu, rồi validate lại bằng Pydantic model của ngành hàng.

Projection hiện hỗ trợ:

- Quy mô gia đình → `household_size`.
- Ưu tiên tiết kiệm điện → `usage_preferences += energy_saving`.
- Ngân sách tối đa → `budget_max_vnd`, chỉ khi cả metadata và nội dung memory
  khớp đúng ngành hàng.

Mỗi lượt tạo tối đa ba candidate. Client nhận event
`memory_confirmation_required` và phải quyết định đầy đủ mỗi candidate đúng một
lần:

| Action | Hành vi |
| --- | --- |
| `use` | Dùng `proposed_patch` đã được server validate |
| `edit` | Chọn option khác của cùng câu hỏi; `other` cần `custom_answer` |
| `ignore` | Không dùng memory; clarification thông thường có thể hỏi lại trường này |

Ví dụ resume:

```json
{
  "kind": "memory_confirmation",
  "decisions": [
    {"candidate_id": "mem-...", "action": "use"},
    {
      "candidate_id": "mem-...",
      "action": "edit",
      "option_id": "three_five"
    },
    {"candidate_id": "mem-...", "action": "ignore"}
  ]
}
```

Thứ tự ưu tiên dữ liệu:

1. Thông tin người dùng vừa nói trong lượt hiện tại.
2. Profile đã xác nhận trong thread hiện tại.
3. Candidate từ Mem0 sau khi người dùng chọn `use` hoặc `edit`.
4. Giá trị mặc định hoặc câu hỏi clarification.

Vì vậy memory cũ không ghi đè câu trả lời mới và một candidate đã quyết định sẽ
không bị hỏi lại cho cùng question/category trong thread đó.

## 6. Luồng ghi memory

Sau khi đã tạo câu trả lời hoàn chỉnh, `queue_memory_write` gửi một cặp message
user/assistant đến:

```http
POST /v3/memories/add/
```

Payload nội bộ quan trọng:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "user_id": "verified-supabase-user-uuid",
  "metadata": {
    "thread_id": "...",
    "turn_id": 1,
    "active_category": "washing_machine",
    "prompt_version": "product-advisor-memory-v3-vi"
  },
  "infer": true,
  "custom_instructions": "...",
  "custom_categories": [
    {"identity_style": "Preferred name, answer length and tone."},
    {"household_context": "Household size, routines and space limitations."}
  ]
}
```

Mem0 xử lý inference bất đồng bộ và trả `event_id`. Event `completed` của chat có
`memory_write.status`:

- `skipped`: memory tắt, anonymous hoặc không có câu trả lời để ghi.
- `queued`: Mem0 đã nhận tác vụ và trả `event_id`.
- `failed`: không queue được, nhưng câu trả lời tư vấn vẫn hoàn tất.

Client kiểm tra kết quả cuối bằng:

```http
GET /chat/{thread_id}/memory-write
Authorization: Bearer <supabase_access_token>
```

Kết quả poll có `status` là `pending`, `succeeded`, `failed` hoặc `skipped`.
`added_count=0` khi `succeeded` là hợp lệ: lượt hội thoại có thể không chứa fact
mới, hoặc Mem0 đã có fact tương đương.

## 7. Xác thực và cô lập dữ liệu

- Backend xác minh bearer token qua endpoint user của Supabase Auth.
- Chỉ UUID người dùng đã xác minh được dùng làm Mem0 `user_id`.
- Search, list và delete luôn được scope theo người dùng hiện tại.
- Khi xóa một memory theo ID, backend đọc memory trước và so khớp owner; memory
  không tồn tại và memory của người khác đều trả `404`.
- Thread cũng có owner. Người dùng khác không thể đọc, resume hay poll write event
  của thread đó.
- `MEM0_API_KEY` là secret server-only, tuyệt đối không đặt trong biến `VITE_*`
  hoặc gửi xuống browser.

## 8. Cấu hình

Backend:

```dotenv
MEMORY_ENABLED=true
MEM0_API_KEY=your-server-only-mem0-key
MEM0_BASE_URL=https://api.mem0.ai
MEM0_SEARCH_TOP_K=10
MEM0_SEARCH_THRESHOLD=0.2
MEM0_SEARCH_TIMEOUT_SECONDS=3
SUPABASE_URL=https://PROJECT_REF.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-publishable-key
SUPABASE_AUTH_TIMEOUT_SECONDS=5
```

Frontend chỉ cần cấu hình Supabase public values:

```dotenv
VITE_SUPABASE_URL=https://PROJECT_REF.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-publishable-key
```

Nếu `MEMORY_ENABLED=true` nhưng thiếu `MEM0_API_KEY`, `SUPABASE_URL` hoặc
`SUPABASE_PUBLISHABLE_KEY`, backend từ chối startup để tránh chạy ở trạng thái
cấu hình dở dang.

## 9. API quản lý memory

Tất cả endpoint dưới đây yêu cầu bearer token hợp lệ:

| Endpoint | Mục đích |
| --- | --- |
| `GET /chat/{thread_id}/memory-write` | Kiểm tra event ghi của lượt chat |
| `GET /me/memories?page=1&page_size=50` | Liệt kê memory của chính người dùng |
| `DELETE /me/memories/{memory_id}` | Xóa một memory sau khi kiểm tra ownership |
| `DELETE /me/memories` | Xóa toàn bộ memory của người dùng hiện tại |

Ví dụ:

```bash
curl http://127.0.0.1:8000/me/memories \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

Xóa memory không sửa ngược dữ liệu đã recall vào checkpoint của thread đang chạy.
Sau khi xóa, nên tạo thread mới để chắc chắn phiên tư vấn không còn dùng snapshot
cũ.

## 10. Xử lý lỗi và degraded mode

Luồng Mem0 được thiết kế fail-open cho hoạt động tư vấn:

- Search lỗi hoặc timeout: `recall_status=failed`, graph tiếp tục không có
  long-term memory.
- Queue write lỗi hoặc timeout: `memory_write.status=failed`, answer vẫn được trả.
- Poll event lỗi: API trả trạng thái `failed` cho event đó.
- List/delete lỗi từ Mem0: endpoint quản lý trả `502` vì không thể hoàn tất thao tác.

Không nên dùng retry vô hạn trong request chat. Một lần tư vấn đúng nhưng chưa ghi
được memory tốt hơn việc giữ stream mở hoặc làm mất câu trả lời.

## 11. File triển khai chính

| File | Trách nhiệm |
| --- | --- |
| `src/advisor/memory/mem0.py` | HTTP adapter Mem0, normalization, categories và extraction instructions |
| `src/advisor/memory/projection.py` | Parse fact an toàn, tạo và validate candidate patch |
| `src/advisor/nodes.py` | Recall, HITL projection, áp dụng preference và queue write |
| `src/advisor/graph.py` | Nối các node memory vào LangGraph |
| `src/advisor/auth.py` | Xác minh Supabase access token |
| `src/advisor/api.py` | Ownership, SSE, polling và API quản lý memory |
| `tests/test_memory_personalization.py` | Test adapter, isolation, projection, HITL và write-back |

Chạy nhóm test liên quan:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/test_memory_personalization.py
```

## 12. Giới hạn hiện tại và cách mở rộng

- Projection quyết định hiện chỉ hỗ trợ quy mô gia đình, tiết kiệm điện và ngân
  sách theo ngành hàng. Các category memory khác vẫn có thể được recall cho style
  hoặc context nhưng không tự động trở thành filter/ranking signal.
- Pattern projection hỗ trợ tiếng Việt và một số cách diễn đạt tiếng Anh phổ
  biến; câu quá mơ hồ sẽ bị bỏ qua thay vì suy đoán.
- `rerank=false` ở Mem0 search; chất lượng recall phụ thuộc `top_k`, `threshold`
  và dữ liệu fact đã được inference.
- API chat hiện dùng một worker vì khóa đồng thời theo `thread_id` nằm trong
  process; giới hạn này độc lập với Mem0.

Khi thêm một fact ảnh hưởng quyết định:

1. Thêm hoặc cập nhật category/instruction trong `mem0.py`; tăng
   `MEMORY_PROMPT_VERSION` nếu thay đổi ý nghĩa extraction.
2. Thêm parser có giới hạn rõ ràng trong `projection.py`.
3. Chỉ map vào path thuộc `CategorySpec.question_profile_paths` và
   `valid_patch_paths`.
4. Validate profile bằng Pydantic, giữ nguyên current-turn precedence và HITL.
5. Bổ sung test cho cross-category isolation, dữ liệu mơ hồ, prompt injection và
   use/edit/ignore.
