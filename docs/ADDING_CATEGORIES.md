# Phân chia công việc khi thêm danh mục sản phẩm

Hướng dẫn implement chi tiết và danh sách quyết định nghiệp vụ cho category owner
nằm tại [`src/advisor/categories/README.md`](../src/advisor/categories/README.md).

## Nền tảng đã chốt

Graph dùng chung không còn dispatch trực tiếp sang code tủ lạnh. Mỗi danh
mục cung cấp một `CategorySpec` qua hàm `get_category_spec()` trong package
của nó. Contract ổn định nằm tại `src/advisor/categories/base.py`.

Một category tự sở hữu:

- schema profile và schema diễn giải câu trả lời tự do;
- danh sách patch path, list path, hard-retrieval path và mapping question/path;
- config collection, embedding, question catalog, payload field và payload index;
- prompt extraction, custom answer, ranking và final response;
- filter builder, semantic search-text builder, candidate normalizer và no-match text;
- lệnh setup index idempotent.

Registry lazy-load spec. Qdrant readiness được cache riêng theo category, và graph
dùng collection/prompt/schema/normalizer của active category. API, SSE, checkpoint,
HITL và frontend contract vẫn dùng chung.

## Thứ tự tích hợp

1. Foundation owner merge contract và regression test trước.
2. Các category owner làm song song trong package riêng, chưa đăng ký registry.
3. Integration owner review metadata/index contract, sau đó thêm tất cả entry
   vào `build_default_registry()` trong một commit duy nhất.
4. Chạy merge gate cho từng category, rồi chạy toàn bộ suite.

## Ranh giới thay đổi bắt buộc

### Foundation/integration owner duy nhất

Được sửa:

- `src/advisor/categories/base.py`
- `src/advisor/categories/registry.py`
- `src/advisor/nodes.py`
- `src/advisor/graph.py`
- `src/advisor/api.py`
- `src/advisor/state.py`
- `src/advisor/schemas.py`
- `src/advisor/retrieval/**`
- `tests/test_api.py`, `tests/test_scaffold.py` và shared fixtures
- `docs/API.md`, file này và phần kiến trúc trong `README.md`

Category owner không được sửa các file trên. Nếu contract thiếu capability,
họ ghi issue/PR note cho foundation owner; không thêm `if category == ...` vào core.

### Mỗi category owner

Chỉ được thêm/sửa:

- `src/advisor/categories/<slug>/**`
- `tests/categories/test_<slug>.py`
- `ingestion/<dataset_slug>/**` khi task có giao ingestion rõ ràng

Không được sửa package category khác, `frontend/**`, persistence, shared
ingestion distributor, file data của category khác, hay tự thêm registry entry.

## Cấu trúc bàn giao của mỗi category

```text
src/advisor/categories/<slug>/
├── __init__.py          # load_config(), get_missing..., get_category_spec()
├── config.yaml         # collection/questions/payload fields/indexes
├── schemas.py          # profile + custom-answer models của category
├── prompts.py          # bốn prompt builder theo CategorySpec
├── filter_builder.py   # hard constraints -> Qdrant Filter
├── normalizer.py       # Qdrant point -> common candidate projection
└── setup_indexes.py    # check/apply idempotent, không chạy khi import
```

`__init__.py` bắt buộc export `get_category_spec()` không nhận tham số và trả
về `CategorySpec`. Import package không được gọi network, tạo collection,
ghi file hoặc đọc secret.

## Constraint dữ liệu và hành vi

- Slug phải trùng giữa registry key, `CategorySpec.name` và `config.category`.
- Tên collection phải duy nhất. Metadata path phải khớp chính xác casing và
  type do ingestion ghi vào Qdrant.
- Mọi `payload_fields` dùng trong filter phải có `payload_indexes` tương ứng.
- Chỉ yêu cầu bắt buộc mới thành hard filter. Sở thích và suy luận
  ở semantic query/ranking; không được silently nới hard constraint.
- Patch path trong prompt, schema, option `profile_updates`, allowlist và
  `question_profile_paths` phải khớp nhau.
- Mỗi question có option ID duy nhất và phải có `other`. Mỗi HITL form tối
  đa ba câu; nhiều required field hơn sẽ được chia qua nhiều round.
- Candidate luôn có `product_id`, `name`, `qdrant_score`; nên có `brand`,
  `effective_price_vnd`, `original_price_vnd`, `promotional_price_vnd` và
  `image_url` để frontend render nhất quán. Extra field được phép.
- Prompt/ranking/response chỉ dùng candidate data được cung cấp, không
  bịa metadata. Ranking chỉ trả product ID đã retrieve.
- Không đổi state, API/SSE event, resume payload hay frontend trong category task.
- Không commit secret, cache, vector generated hoặc dataset lớn ngoài scope.

## Test bắt buộc trước bàn giao

Mỗi `test_<slug>.py` phải kiểm tra:

1. package import và `CategorySpec.validate()` thành công;
2. config/slug/collection và exact metadata/index types;
3. valid patch, clear/replace/add/remove và invalid path bị bỏ qua;
4. missing-field/question rule, option thường và `other`;
5. filter chỉ chứa indexed field và cho kết quả đúng trên Qdrant local;
6. candidate normalizer có common fields và không suy đoán dữ liệu thiếu;
7. prompt có tên/rule đúng category và no-match text không nhắc category khác;
8. graph smoke: intent đi qua pipeline, interrupt/resume, retrieve/rank/response;
9. switch A → B → A giữ riêng profile và recommendation context.

Merge gate tối thiểu:

```bash
.venv/bin/python -m pytest -q
```

Test không được phụ thuộc Gemini/Qdrant cloud hoặc secret thật.

## Gợi ý chia nhóm song song

Mỗi thành viên nhận một category nếu metadata đã sạch. Với category còn
phải chuẩn hóa ingestion, tách thành hai task tuần tự: `ingestion contract`
trước, `advisor package` sau. Không cho hai người cùng sửa một package.

Danh sách slug hiện có thể giao: `karaoke_microphone`,
`phone_recording_microphone`, `smartwatch`, `desktop`, `monitor`, `printer` và
`tablet`.
