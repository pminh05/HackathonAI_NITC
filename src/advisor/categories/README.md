# Hướng dẫn implement một product category

Tài liệu này là contract làm việc cho người phụ trách từng danh mục sản phẩm.
Không copy nguyên logic tủ lạnh rồi chỉ đổi prompt. Mỗi category có một collection,
metadata và cách tư vấn riêng; owner phải xác minh và quyết định rõ từng phần dưới đây.

## 1. Luồng dùng chung và phần category sở hữu

Graph dùng chung thực hiện chuỗi sau:

```text
phân tích lượt nói
  -> lấy CategorySpec của active category
  -> cập nhật need profile
  -> xác định thông tin còn thiếu
  -> hỏi tối đa 3 câu mỗi round
  -> build hard filter
  -> semantic retrieval trong collection của category
  -> normalize candidate
  -> rank tối đa 3 sản phẩm
  -> viết câu trả lời có căn cứ
```

Graph chỉ điều phối. Toàn bộ khác biệt nghiệp vụ phải đi qua `CategorySpec` trong
[`base.py`](base.py), không thêm nhánh `if category == ...` vào core.

Mỗi category bắt buộc tự sở hữu:

- một Qdrant collection riêng;
- profile schema và custom-answer schema;
- rule trích xuất nhu cầu và rule hỏi làm rõ;
- hard constraints, soft preferences và implicit needs;
- payload field/index mapping và filter builder;
- semantic search text và candidate normalizer;
- ranking prompt, response prompt và no-match response.

## 2. Phạm vi file

Owner của `<slug>` chỉ được tạo hoặc sửa:

```text
src/advisor/categories/<slug>/**
tests/categories/test_<slug>.py
ingestion/<dataset_slug>/**       # chỉ khi task giao rõ ingestion
```

Không sửa `nodes.py`, `graph.py`, `api.py`, `state.py`, `schemas.py`,
`categories/base.py`, `categories/registry.py`, shared retrieval, frontend,
persistence hay package của category khác. Không tự đăng ký category vào registry;
integration owner sẽ làm việc đó trong một commit riêng.

Nếu contract hiện tại thiếu capability, ghi lại use case và đề nghị foundation
owner mở rộng contract. Không workaround bằng import ngược core hoặc branch theo slug.

## 3. Cấu trúc package đề xuất

```text
<slug>/
├── __init__.py
├── config.yaml
├── schemas.py
├── prompts.py
├── filter_builder.py
├── normalizer.py
└── setup_indexes.py
```

Có thể gộp file với category nhỏ, nhưng `__init__.py` bắt buộc export:

```python
def get_category_spec() -> CategorySpec:
    ...
```

Factory không nhận tham số. Import package/factory không được gọi network, ghi file,
tạo index, đọc secret hay khởi tạo LLM/Qdrant client.

## 4. Các quyết định owner bắt buộc phải đưa ra

Trước khi code, tạo một decision record trong mô tả PR theo bảng sau:

| Nhóm | Owner phải quyết định và chứng minh |
|---|---|
| Collection | Tên collection riêng, embedding model, retrieval limit và dữ liệu đã ingest |
| Metadata | Exact key, casing, kiểu dữ liệu, tỷ lệ missing và ví dụ payload thật |
| Profile | Những thông tin nào thật sự cần để tư vấn category này |
| Required fields | Tối thiểu cần biết gì trước retrieval; field nào có thể bỏ qua |
| Hard constraints | Điều kiện nào phải loại sản phẩm không thỏa |
| Soft preferences | Điều gì chỉ ảnh hưởng ranking, không được loại candidate |
| Implicit needs | Suy luận nào an toàn từ ngữ cảnh và bằng chứng nào cho phép suy luận |
| Questions | Thứ tự ưu tiên hỏi, wording, option và cách option cập nhật profile |
| Filters | Profile path nào map sang index nào và xử lý missing metadata ra sao |
| Search text | Thông tin nào nên đưa vào semantic query, thông tin nào không |
| Candidate shape | Common fields và category-specific fields được phép đưa cho LLM/UI |
| Ranking | Tiêu chí phù hợp, trade-off và cách đa dạng hóa lựa chọn |
| Response | Nội dung phải nêu, điều bị cấm suy đoán và no-match wording |

Không được để các quyết định này hình thành ngẫu nhiên bằng cách copy tủ lạnh.

## 5. Thiết kế profile và schema

`profile_model` là nguồn sự thật cho dữ liệu nhu cầu của category. Owner quyết định:

1. scalar nào cần lưu, ví dụ ngân sách, kích thước, số người dùng;
2. list enum nào có tập giá trị đóng;
3. thông tin nào nằm trong `hard_constraints`;
4. soft/implicit field nào được phép giữ dạng text tự do;
5. giới hạn hợp lệ, đơn vị và cách biểu diễn `None`/empty list.

Không đưa một field vào hard constraints nếu collection không có metadata chuẩn hóa
và index đủ tin cậy để enforce. Một yêu cầu được gọi là “hard” nhưng chỉ nằm trong
prompt là lỗi logic.

Ba nhóm phải phân biệt:

- **Hard constraint:** người dùng nói bắt buộc/phải có/tối đa/tối thiểu. Dùng để
  build Qdrant filter và có thể loại sản phẩm.
- **Soft preference:** thích, ưu tiên, nếu có thì tốt. Dùng trong semantic query
  hoặc ranking, không loại sản phẩm.
- **Implicit need:** hệ thống suy luận trực tiếp từ ngữ cảnh. Phải có evidence,
  không biến thành hard constraint nếu người dùng chưa xác nhận.

`custom_answer_model` phải có ít nhất:

```python
interpretation_status
raw_answer
confidence
```

Ngoài ra model khai báo các field category-specific mà câu trả lời `other` được phép
cập nhật. Không trả dictionary tự do không qua Pydantic validation.

## 6. Patch paths và execution rules

Mỗi category khai báo đồng bộ bốn tập rule:

- `valid_patch_paths`: tất cả dotted paths mà extraction được phép thay đổi;
- `list_patch_paths`: subset chứa list, hỗ trợ add/remove/replace/clear;
- `hard_retrieval_paths`: thay đổi các path này bắt buộc retrieval lại;
- `question_profile_paths`: question ID nào giải quyết hoặc làm thay đổi path nào.

Owner phải kiểm tra cùng một path xuất hiện nhất quán trong:

- Pydantic schema;
- extraction prompt;
- `valid_patch_paths`;
- option `profile_updates` trong config;
- filter builder nếu đó là hard constraint;
- `question_profile_paths` nếu được thu thập qua form.

Không thêm soft preference vào `hard_retrieval_paths` nếu chỉ cần rerank candidate
đã cache. Ngược lại, thay đổi budget/kích thước/tính năng bắt buộc phải retrieval lại.

## 7. Question catalog và rule hỏi

`required_profile_fields` không nhất thiết là toàn bộ profile. Nó chỉ chứa những
thông tin tối thiểu mà category quyết định cần hỏi trước khi tư vấn.

`get_missing_profile_fields(profile, config)` phải deterministic và trả question ID,
không trả raw profile path. Ví dụ một question `budget` có thể được xem là resolved
khi có `budget_max_vnd` hoặc một open-ended `budget_segment`.

Mỗi question trong `question_catalog` phải có:

- ID duy nhất, ổn định qua các version;
- `question_type`: `explicit` hoặc `implicit`;
- câu hỏi đúng category, ngắn và không dẫn dắt;
- các option ID duy nhất;
- `profile_updates` hợp lệ;
- option `other` để người dùng nhập tự do.

Graph hỏi tối đa 3 câu trong một form. Nếu có hơn 3 required fields, owner phải xác
định thứ tự ưu tiên trong `required_profile_fields`; các câu còn lại được hỏi round sau.

Không hỏi thông tin chỉ “hay thì có” nếu nó làm tăng ma sát nhưng không cải thiện
filter/ranking đáng kể. Không hỏi lại field đã resolved trừ khi người dùng thay đổi nó.

## 8. `config.yaml`, collection và indexes

Các key tối thiểu:

```yaml
category: <slug>
display_name: <Vietnamese name>
collection: <unique_qdrant_collection>
embedding_model: <model used by the collection>
retrieval_limit: 12

required_profile_fields: []
payload_fields: {}
payload_indexes: {}
question_catalog: {}
```

Mỗi category dùng một collection riêng. Không tái sử dụng collection của category
khác dù metadata trông giống nhau. Registry sẽ từ chối hai category cùng collection.

Với từng `payload_fields`, owner phải kiểm tra bằng payload thật:

- path chính xác, bao gồm chữ hoa/thường, dấu và quoting;
- type thực tế đồng nhất với `keyword`, `integer`, `float` hoặc `bool`;
- field có được populate đủ nhiều hay không;
- payload index tương ứng tồn tại;
- ingestion và runtime dùng cùng đơn vị.

Không đoán metadata key từ tên cột Excel. Phải kiểm tra JSON sau processing hoặc point
thật trong collection. Không dùng raw text field làm numeric/range filter.

`setup_indexes.py` phải idempotent: mặc định chỉ report, chỉ tạo index khi có
`--apply`, và không được xóa/recreate collection.

## 9. Filter builder

`build_filter(profile, payload_fields)` chỉ dịch hard constraints thành Qdrant filter.

Owner phải quyết định rõ cho từng rule:

- dùng `must`, `should` hay nested filter;
- equality, keyword set hay numeric range;
- boundary có inclusive không;
- missing metadata được xem là fail hay được phép pass;
- nhiều giá/biến thể giá được kết hợp ra sao;
- đơn vị có cần normalize trước khi so sánh không.

Nguyên tắc an toàn:

- hard constraint không được tự ý bỏ khi field thiếu;
- không build condition trên field chưa index;
- không filter soft/implicit preference;
- không map enum sang metadata value nếu chưa có test chứng minh;
- filter phải serialize được để lưu trong LangGraph checkpoint.

Ví dụ tủ lạnh chỉ là tham khảo. Máy lạnh, máy giặt, màn hình hoặc smartwatch phải
có filter semantics riêng theo metadata thật của collection tương ứng.

## 10. Semantic search text

`build_search_text(profile, user_query)` tạo text gửi cho embedding search. Owner chọn:

- giữ phần nào của user query;
- thêm soft preferences và implicit needs nào;
- biểu diễn thuật ngữ/đơn vị category ra sao;
- có đưa hard constraints vào text để hỗ trợ relevance hay không.

Hard constraint vẫn phải được enforce bằng filter; xuất hiện trong search text không
thay thế filter. Không nhồi toàn bộ profile JSON nếu tạo nhiễu hoặc lộ field nội bộ.

Với action `more_options`, core sẽ tái sử dụng discovery query ban đầu và loại các
product ID đã trình bày. Builder không cần tự xử lý cache hoặc exclusion.

## 11. Candidate normalizer

`normalize_candidate(point)` là ranh giới giữa metadata raw và phần còn lại của app.
Nó phải luôn trả:

```python
{
    "product_id": "...",
    "name": "...",
    "qdrant_score": 0.0,
}
```

Nên trả thêm khi dữ liệu có thật:

```python
brand
image_url
effective_price_vnd
original_price_vnd
promotional_price_vnd
description
```

Các thông số riêng như BTU, khối lượng giặt, kích thước màn hình, pin hoặc chống nước
được phép thêm dưới tên rõ đơn vị. Không đổi tên cùng một khái niệm giữa các product
trong cùng category.

Normalizer phải:

- whitelist field hữu ích, không đẩy toàn bộ payload vào prompt;
- giữ `None` khi thiếu, không tạo default mang ý nghĩa giả;
- không chuyển text không parse được thành số 0;
- không suy ra tính năng từ tên sản phẩm nếu ingestion chưa xác nhận;
- tính `effective_price_vnd` theo một rule đã test.

## 12. Prompt logic

Mỗi category cung cấp bốn prompt builder.

### Need extraction

Phải liệt kê exact valid paths, enum và rule hard/soft/implicit. Chỉ cập nhật phần thay
đổi so với profile hiện tại, giữ evidence và không tự điền field còn thiếu.

### Custom answer

Chỉ diễn giải nội dung người dùng thật sự nói. Nếu không hiểu, trả `unresolved`; không
ép câu trả lời vào option gần nhất và không tự sinh câu hỏi mới.

### Ranking

Chỉ chọn ID trong candidate list, tối đa 3 sản phẩm. Owner định nghĩa tiêu chí phù hợp
và trade-off quan trọng của category. Khi có nhiều lựa chọn, nên đại diện các hướng
khác nhau thay vì chọn ba model gần như giống nhau.

### Final response

Chỉ dùng normalized candidate data và ranking result. Không bịa giá, tính năng,
khuyến mãi hoặc kết luận “tốt nhất tuyệt đối”. Phải nói rõ dữ liệu thiếu và ít nhất
một đánh đổi có căn cứ cho mỗi lựa chọn.

## 13. No-match logic

`no_match_answer(profile)` phải nhắc đúng category và các hard constraints quan trọng
đã khiến candidate set rỗng. Không tự nới constraint, không khẳng định toàn thị trường
không có sản phẩm và không nhắc Qdrant/prompt/filter nội bộ.

Owner quyết định constraint nào hữu ích để giải thích cho người dùng; không dump toàn
bộ profile kỹ thuật.

## 14. Tạo `CategorySpec`

Skeleton tối thiểu:

```python
def get_category_spec() -> CategorySpec:
    config = load_config()
    return CategorySpec(
        name="<slug>",
        display_name="<Tên danh mục>",
        config=config,
        profile_model=CategoryNeedProfile,
        custom_answer_model=CategoryCustomAnswer,
        build_need_extraction_prompt=build_need_extraction_prompt,
        build_custom_answer_prompt=build_custom_answer_prompt,
        build_ranking_prompt=build_ranking_prompt,
        build_response_prompt=build_response_prompt,
        build_filter=build_filter,
        build_search_text=build_search_text,
        no_match_answer=no_match_answer,
        normalize_candidate=normalize_candidate,
        get_missing_profile_fields=get_missing_profile_fields,
        valid_patch_paths=frozenset({...}),
        list_patch_paths=frozenset({...}),
        hard_retrieval_paths=frozenset({...}),
        question_profile_paths={...},
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.<slug>.setup_indexes --apply"
        ),
    )
```

Chạy trực tiếp `get_category_spec().validate()` trước khi bàn giao. Validation hiện
kiểm tra config keys, slug, required question, patch subsets, question paths,
payload indexes, custom-answer fields, duplicate options và option `other`.

## 15. Test bắt buộc

`tests/categories/test_<slug>.py` tối thiểu phải có:

1. import package không tạo external side effect;
2. `CategorySpec.validate()` pass;
3. slug, display name, collection và embedding model đúng;
4. payload fields/indexes khớp exact metadata thật;
5. profile schema chấp nhận case đúng và từ chối case sai;
6. patch set/replace/add/remove/clear và invalid path;
7. missing-field resolution và thứ tự question;
8. option thường và option `other` cập nhật đúng profile;
9. từng hard constraint tạo đúng Qdrant condition;
10. soft/implicit preference không tạo hard filter;
11. filter chạy đúng trên Qdrant local/in-memory fixture;
12. normalizer có common fields, xử lý price/missing data đúng;
13. prompt chứa rule đúng category và không nhắc category khác;
14. no-match response đúng category;
15. graph smoke qua interrupt/resume/retrieve/rank/response;
16. switch category A -> B -> A không trộn profile/candidate context.

Test không gọi Gemini hoặc Qdrant cloud và không dùng secret thật.

Chạy merge gate:

```bash
.venv/bin/python -m pytest -q tests/categories/test_<slug>.py
.venv/bin/python -m pytest -q
```

## 16. Checklist bàn giao

- [ ] Decision record đã trả lời đủ bảng ở mục 4.
- [ ] Collection riêng đã có dữ liệu và tên không trùng category khác.
- [ ] Metadata paths/types được đối chiếu với payload thật.
- [ ] Profile schema và patch paths đồng bộ.
- [ ] Hard constraints đều có indexed filter thực sự.
- [ ] Soft/implicit rules không bị biến thành hard filter.
- [ ] Required questions là tối thiểu cần thiết và có thứ tự ưu tiên.
- [ ] Tất cả question có option `other`.
- [ ] Candidate normalizer không bịa hoặc leak payload thừa.
- [ ] Bốn prompt không nhắc category khác và không cho phép hallucination.
- [ ] Setup index idempotent, không phá collection.
- [ ] Category tests và toàn bộ regression suite đều pass.
- [ ] Không sửa file ngoài ownership scope.
- [ ] Chưa tự thêm registry entry; integration owner sẽ đăng ký sau review.

Package [`refrigerator`](refrigerator/) là implementation tham chiếu về cách nối
`CategorySpec`, không phải template nghiệp vụ để sao chép nguyên trạng.
