# Product Advisor

AI chatbot tư vấn và so sánh sản phẩm theo nhu cầu thực tế của khách hàng. Hệ thống sử dụng LangGraph để quản lý luồng hội thoại, Human-in-the-loop để thu thập thông tin còn thiếu và Qdrant để truy xuất sản phẩm theo từng ngành hàng.

Phiên bản hiện tại triển khai đầy đủ cho **tủ lạnh**, **máy lạnh**, **máy giặt**,
**máy sấy quần áo**, **máy rửa chén**, **tủ mát**, **tủ đông**,
**máy nước nóng**, **máy tính bảng** và **máy in**, sau đó có thể mở rộng sang
các sheet sản phẩm khác qua `CategorySpec`.

Tài liệu HTTP API, SSE và Human-in-the-loop: [docs/API.md](docs/API.md).

## Yêu cầu hệ thống

- Python 3.11 trở lên.
- Node.js 20.19 trở lên hoặc Node.js 22.12 trở lên.
- Tài khoản Google AI và `GOOGLE_API_KEY` hợp lệ.
- Qdrant Cloud có bật Cloud Inference, collection `tulanh`, `maylanh`, `maygiat`,
  `maysayquanao`, `mayruachen`, `tumattudong`, `maynuocnong`, `maytinhbang` và
  `mayin` đã chứa dữ liệu sản phẩm.
- npm đi kèm Node.js.

Backend và frontend chạy thành hai process riêng:

| Thành phần | URL mặc định                 |
| ---------- | ---------------------------- |
| FastAPI    | `http://127.0.0.1:8000`      |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| React/Vite | `http://localhost:5173`      |

## Cài đặt và chạy local

Các lệnh dưới đây được chạy từ thư mục gốc của repository.

### 1. Cài dependencies backend

Tạo virtual environment và cài project ở editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Trên Windows PowerShell, kích hoạt virtual environment bằng:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Cấu hình backend

Tạo file môi trường từ template:

```bash
cp .env.example .env
```

Cập nhật tối thiểu ba biến bắt buộc trong `.env`:

```dotenv
GOOGLE_API_KEY=your-google-api-key
GEMINI_MODEL=gemini-3.5-flash
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

Để tự động chuyển sang Qwen3.6-27B trên FPT AI Factory khi Gemini lỗi, cấu
hình thêm API key của FPT. `FPT_BASE_URL` là root URL của Marketplace, không
thêm hậu tố `/v1`:

```dotenv
FPT_API_KEY=your-fpt-api-key
FPT_MODEL=Qwen3.6-27B
FPT_BASE_URL=https://mkp-api.fptcloud.com
FPT_ENABLE_THINKING=false
FPT_TIMEOUT_SECONDS=30
FPT_MAX_RETRIES=0
```

Fallback được áp dụng cho cả phản hồi văn bản và structured output. Nếu chưa
cấu hình `FPT_API_KEY`, ứng dụng tiếp tục chỉ dùng Gemini như trước.
`FPT_ENABLE_THINKING=false` truyền hard switch của Qwen vào chat template để
model trả lời trực tiếp mà không sinh thinking block.
FPT dùng function calling cho các schema quyết định hữu hạn, nhưng riêng
`ProfilePatch` dùng native JSON schema để tránh request bị treo với các trường
dictionary. Mỗi request có timeout 30 giây và không retry mặc định vì đây đã là
nhánh fallback sau Gemini.

Các model đều là LangChain Runnable và có metadata theo provider/model. Khi
cần theo dõi bằng LangSmith, chỉ cần bổ sung các biến môi trường chuẩn; không
cần đổi code gọi model:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=product-advisor
```

Các giá trị local còn lại có thể giữ mặc định:

```dotenv
CHECKPOINT_DB_PATH=.data/checkpoints.sqlite
API_HOST=127.0.0.1
API_PORT=8000
API_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
SSE_HEARTBEAT_SECONDS=15
```

Backend uses SQLite checkpoints by default. For a deployed Supabase/PostgreSQL
backend, configure the Session pooler connection string instead:

```env
CHECKPOINT_BACKEND=postgres
SUPABASE_DATABASE_URL=postgresql://postgres.PROJECT_REF:URL_ENCODED_PASSWORD@HOST:5432/postgres
```

Keep credentials server-side. URL-encode special characters in the database
password (for example, `@` as `%40`). PostgreSQL checkpoint tables are created
automatically on the first backend startup.

Kiểm tra payload indexes của collection `tulanh`:

```bash
PYTHONPATH=src .venv/bin/python -m advisor.categories.refrigerator.setup_indexes
```

Nếu command báo thiếu index, tạo các index còn thiếu một lần:

```bash
PYTHONPATH=src .venv/bin/python -m advisor.categories.refrigerator.setup_indexes --apply
```

Thực hiện kiểm tra và tạo index tương tự cho collection `maylanh`:

```bash
PYTHONPATH=src .venv/bin/python -m advisor.categories.air_conditioner.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.air_conditioner.setup_indexes --apply
```

Với collection `maygiat`:

```bash
PYTHONPATH=src .venv/bin/python -m advisor.categories.washing_machine.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.washing_machine.setup_indexes --apply
```

Với collection `maysayquanao`, `mayruachen` và `tumattudong`:

```bash
PYTHONPATH=src .venv/bin/python -m advisor.categories.dryer.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.dryer.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.dishwasher.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.dishwasher.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.cooler_freezer.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.cooler_freezer.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.water_heater.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.water_heater.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.tablet.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.tablet.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.printer.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.printer.setup_indexes --apply
```

Lệnh này chỉ tạo payload index; không tạo collection và không import catalog sản phẩm.

### 3. Chạy backend

API MVP phải chạy đúng một worker vì khóa chống request đồng thời theo `thread_id`
đang được giữ trong memory của process:

```bash
PYTHONPATH=src .venv/bin/uvicorn advisor.api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

When PostgreSQL checkpoints are enabled on Windows, use the bundled selector
event-loop factory because psycopg async does not support Windows' default
Proactor event loop:

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\uvicorn.exe advisor.api:app `
  --host 127.0.0.1 `
  --port 8000 `
  --workers 1 `
  --loop advisor.event_loop:postgres_compatible_loop_factory
```

Kiểm tra API đã sẵn sàng:

```bash
curl http://127.0.0.1:8000/healthz
```

Kết quả mong đợi:

```json
{ "status": "ok" }
```

### 4. Cài dependencies và chạy frontend

Mở terminal thứ hai, vẫn giữ backend đang chạy:

```bash
cd frontend
cp .env.example .env
npm ci
npm run dev
```

Mở `http://localhost:5173`. Frontend mặc định gọi API qua cấu hình:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Nếu backend chạy ở địa chỉ khác, sửa `frontend/.env` rồi khởi động lại Vite.
Dùng `npm install` thay cho `npm ci` khi chủ động cập nhật dependencies hoặc lockfile.

### 5. Kiểm tra build và test

Backend tests:

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

Frontend production build:

```bash
cd frontend
npm run build
npm run preview
```

Build được tạo tại `frontend/dist/`. Lệnh preview chỉ dùng để kiểm tra build local,
không phải production server.

## Xử lý lỗi thường gặp

- API dừng khi startup: kiểm tra `GOOGLE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` và payload indexes.
- `GET /healthz` trả `starting`: xem log backend để tìm lỗi kết nối Gemini hoặc Qdrant.
- Trình duyệt báo CORS: thêm đúng origin frontend vào `API_CORS_ORIGINS`, sau đó restart backend.
- Frontend không nhận stream: kiểm tra backend ở đúng `VITE_API_BASE_URL` và không có reverse proxy buffer SSE.
- API trả `409`: thread có thể đang chạy hoặc đang chờ câu trả lời làm rõ; không gửi nhiều request đồng thời.
- Đổi biến trong `.env`: luôn restart process tương ứng để cấu hình mới có hiệu lực.

## Luồng xử lý

```text
User message / clarification answer
    ↓
Analyze category, turn action and product references
    ├── Detail / compare / explain → reuse recommendation context
    ├── More options → rerank unseen cached candidates
    └── New or changed needs → apply profile patch
                                  ↓
                            Re-check missing fields
                              ├── Còn thiếu → HITL form
                              └── Đủ → reuse / rerank / retrieve
                                                   ↓
                                      Grounded streamed response
```

HITL đưa tối đa ba thông tin còn thiếu và yêu cầu trả lời đầy đủ form. Giao diện
tự chuyển sang câu tiếp theo sau mỗi lựa chọn và tự động retrieval ngay khi câu
cuối hoàn tất. Lựa chọn `other` cho phép nhập câu trả lời riêng.

## Thiết kế state

State là working memory của một phiên tư vấn:

```text
messages          Lịch sử đầy đủ của user và assistant
conversation      Active category, turn action và execution mode
category_contexts Profile, clarification và recommendations riêng từng category
need_profile      Projection tương thích của profile đang active
clarification     Projection câu hỏi HITL hiện tại
retrieval/ranking Kết quả của lượt hiện tại
response/control  Câu trả lời và trạng thái thực thi
```

Mỗi node đọc state và chỉ cập nhật phần dữ liệu thuộc trách nhiệm của nó. State không chứa toàn bộ catalog hoặc embedding.

## Persistence

LangGraph checkpointer lưu state theo `thread_id`.

```text
thread_id
    ↓
Checkpoint state
    ↓
HITL interrupt
    ↓
Resume cùng thread_id
    ↓
Tiếp tục build filter và search
```

- Graph programmatic mặc định dùng `InMemorySaver`.
- FastAPI dùng `AsyncSqliteSaver` và giữ HITL qua restart.
- Khi cần nhiều worker/replica, thay SQLite bằng PostgreSQL.
- `user_id`: đại diện cho khách hàng.
- `thread_id`: đại diện cho từng cuộc hội thoại.

Long-term memory bằng Mem0 chưa được bật. Checkpoint chỉ là short-term memory của
thread; recommendation cache và clarification không được thiết kế để đưa vào
Mem0. Khi tích hợp sau này, thông tin khách vừa nói sẽ ưu tiên hơn thread context,
và thread context ưu tiên hơn default từ Mem0.

## Qdrant collections

Mỗi sheet hoặc ngành hàng được index thành một collection riêng:

```text
Tủ Lạnh    → tulanh
Máy lạnh   → maylanh
Máy giặt   → maygiat
Máy sấy quần áo → maysayquanao
Máy rửa chén → mayruachen
Tủ mát, tủ đông → tumattudong
Máy nước nóng → maynuocnong
Máy tính bảng → maytinhbang
Máy in → mayin
Laptop     → products_laptop
```

Cách chia này giúp:

- Metadata của từng ngành hàng độc lập.
- Filter đơn giản và ít field rỗng.
- Dễ re-index từng sheet.
- Các thành viên có thể làm category riêng, giảm conflict.

Các trường chuẩn hóa như giá, dung tích, kích thước và tính năng boolean được dùng
cho metadata filter. Thương hiệu cùng các trường mô tả dài như công nghệ và tiện
ích được đưa vào semantic retrieval.

Collection `tulanh`, `maylanh`, `maygiat`, `maysayquanao`, `mayruachen`,
`tumattudong`, `maynuocnong`, `maytinhbang` và `mayin` dùng embedding
`intfloat/multilingual-e5-small`. Trước khi chạy live, kiểm tra payload index
của từng category:

```bash
PYTHONPATH=src .venv/bin/python -m advisor.categories.refrigerator.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.air_conditioner.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.washing_machine.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.dryer.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.dishwasher.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.cooler_freezer.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.water_heater.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.tablet.setup_indexes
PYTHONPATH=src .venv/bin/python -m advisor.categories.printer.setup_indexes
```

Nếu lệnh báo thiếu index, tạo chúng một lần bằng:

```bash
PYTHONPATH=src .venv/bin/python -m advisor.categories.refrigerator.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.air_conditioner.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.washing_machine.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.dryer.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.dishwasher.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.cooler_freezer.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.water_heater.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.tablet.setup_indexes --apply
PYTHONPATH=src .venv/bin/python -m advisor.categories.printer.setup_indexes --apply
```

Runtime chỉ kiểm tra prerequisite và không tự thay đổi schema Qdrant.

## Chạy graph bằng Python

Ngoài FastAPI, graph vẫn có thể được gọi trực tiếp bằng Python:

```python
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from advisor.graph import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "conversation-1"}}

result = graph.invoke(
    {"messages": [HumanMessage(content="Tư vấn tủ lạnh cho gia đình tôi")]},
    config,
)

questions = result["__interrupt__"][0].value["questions"]

result = graph.invoke(
    Command(
        resume={
            "answers": [
                {"question_id": "household_size", "option_id": "three_four"},
                {"question_id": "budget", "option_id": "10m_20m"},
                {"question_id": "usage_preferences", "option_id": "energy_saving"},
            ]
        }
    ),
    config,
)

print(result["response"]["answer"])
```

Phải dùng lại đúng compiled graph/checkpointer và `thread_id` khi resume. Với option `other`, gửi thêm `custom_answer`.

## Gọi FastAPI và SSE bằng cURL

API dùng SQLite checkpoint để giữ Human-in-the-Loop qua nhiều HTTP request. Chạy đúng một worker:

```bash
PYTHONPATH=src .venv/bin/uvicorn advisor.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Mở một cuộc hội thoại mới. Dùng `-N` để `curl` hiển thị SSE ngay khi server gửi:

```bash
curl -N http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Tư vấn giúp tôi một chiếc tủ lạnh"}'
```

Event `session` trả `thread_id`; nếu cần làm rõ, event cuối là `clarification_required`. Gửi toàn bộ câu trả lời bằng đúng thread đó:

```bash
curl -N http://127.0.0.1:8000/chat/THREAD_ID/resume \
  -H 'Content-Type: application/json' \
  -d '{
    "answers": [
      {"question_id":"household_size","option_id":"three_four"},
      {"question_id":"budget","option_id":"10m_20m"},
      {"question_id":"usage_preferences","option_id":"energy_saving"}
    ]
  }'
```

Trong lượt hoàn tất, API phát các event `token` rồi `completed`. Có thể khôi phục form hoặc kết quả sau khi reload bằng:

```bash
curl http://127.0.0.1:8000/chat/THREAD_ID
```

Các endpoint:

- `POST /chat`: tạo thread mới hoặc tiếp tục thread đã hoàn tất.
- `POST /chat/{thread_id}/resume`: resume HITL.
- `GET /chat/{thread_id}`: trạng thái, pending questions hoặc kết quả cuối.
- `GET /healthz`: health check nhẹ.

## Cấu trúc thư mục

```text
product-advisor/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.ts
│
├── src/advisor/
│   ├── graph.py
│   ├── api.py
│   ├── state.py
│   ├── nodes.py
│   ├── schemas.py
│   │
│   ├── categories/
│   │   ├── base.py
│   │   ├── refrigerator/
│   │   │   ├── config.yaml
│   │   │   ├── prompts.py
│   │   │   └── filter_builder.py
│   │   ├── air_conditioner/
│   │   │   ├── config.yaml
│   │   │   ├── schemas.py
│   │   │   ├── prompts.py
│   │   │   ├── filter_builder.py
│   │   │   └── normalizer.py
│   │   ├── washing_machine/
│   │   │   ├── config.yaml
│   │   │   ├── schemas.py
│   │   │   ├── prompts.py
│   │   │   ├── filter_builder.py
│   │   │   └── normalizer.py
│   │   ├── dryer/
│   │   │   ├── config.yaml
│   │   │   ├── schemas.py
│   │   │   ├── prompts.py
│   │   │   ├── filter_builder.py
│   │   │   └── normalizer.py
│   │   ├── dishwasher/
│   │   │   ├── config.yaml
│   │   │   ├── schemas.py
│   │   │   ├── prompts.py
│   │   │   ├── filter_builder.py
│   │   │   └── normalizer.py
│   │   ├── water_heater/
│   │   │   ├── config.yaml
│   │   │   ├── schemas.py
│   │   │   ├── prompts.py
│   │   │   ├── filter_builder.py
│   │   │   └── normalizer.py
│   │   ├── tablet/
│   │   │   ├── config.yaml
│   │   │   ├── schemas.py
│   │   │   ├── prompts.py
│   │   │   ├── filter_builder.py
│   │   │   └── normalizer.py
│   │   ├── printer/
│   │   │   ├── config.yaml
│   │   │   ├── schemas.py
│   │   │   ├── prompts.py
│   │   │   ├── filter_builder.py
│   │   │   └── normalizer.py
│   │   │
│   │   └── registry.py
│   │
│   ├── retrieval/
│   │   └── qdrant.py
│   ├── persistence/
│   │   └── checkpointer.py
│   └── memory/
│       └── mem0.py
│
└── tests/
    ├── test_api.py
    └── categories/
        ├── test_refrigerator.py
        ├── test_air_conditioner.py
        ├── test_washing_machine.py
        ├── test_dryer.py
        ├── test_dishwasher.py
        ├── test_water_heater.py
        ├── test_tablet.py
        └── test_printer.py
```

## Vai trò các thành phần

- `graph.py`: định nghĩa node, edge, interrupt và routing.
- `api.py`: FastAPI, SSE streaming, thread validation và HITL resume.
- `state.py`: định nghĩa LangGraph state.
- `nodes.py`: logic dùng chung như intent, profile extraction, HITL, retrieval, ranking và response.
- `schemas.py`: schema cho state, form câu hỏi và structured output.
- `config.yaml`: collection, required fields và metadata mapping của từng category.
- `prompts.py`: prompt hỏi lại và prompt tư vấn.
- `filter_builder.py`: chuyển `need_profile` thành Qdrant filter.
- `base.py`: contract hành vi bắt buộc của mọi category (`CategorySpec`).
- `registry.py`: lazy-load và validate category spec theo module tương ứng.
- `retrieval/qdrant.py`: kết nối và truy vấn Qdrant.
- `persistence/checkpointer.py`: cấu hình persistence.
- `memory/mem0.py`: placeholder cho long-term memory.
- `tests/categories/`: test rule, filter và output theo từng ngành hàng.
- `frontend/src/App.tsx`: reducer, chat UI, clarification flow và local persistence.
- `frontend/src/api.ts`: HTTP client và SSE parser cho POST stream.
- `frontend/src/styles.css`: giao diện desktop không phụ thuộc UI framework.

## Mở rộng category mới

Khi thêm một sheet mới:

1. Tạo Qdrant collection riêng.
2. Tạo package category mới và export `get_category_spec()`.
3. Khai báo `config.yaml`, schema profile và question rules.
4. Viết prompt, filter builder, search-text builder và candidate normalizer.
5. Thêm test category tương ứng.
6. Để integration owner đăng ký category trong `registry.py`.

Graph, state và retrieval core không cần thay đổi.
Ranh giới file, constraint dữ liệu và merge gate được mô tả trong
[`docs/ADDING_CATEGORIES.md`](docs/ADDING_CATEGORIES.md).
