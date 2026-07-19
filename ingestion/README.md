# Ingestion Pipeline

Pipeline làm sạch, phân phối, xử lý semantic, embedding và đưa dữ liệu sản phẩm vào Qdrant phục vụ RAG.

## Luồng dữ liệu tổng thể

```text
clean_data/Spec_cate_gia.xlsx
        ↓ chỉnh sửa thủ công số liệu lỗi
clean_data/Clean_data.xlsx
        ↓ tách dữ liệu theo danh mục bằng clean_data/Final.ipynb
clean_data/Final/<Tên thiết bị>.xlsx
        ↓ xuất JSON
clean_data/Final_json/<Tên thiết bị>.json
        ↓ chạy clean_data/distribute_final_json.py
<ten_thiet_bi>/data/<ten_thiet_bi>.json
        ↓ processing.py → changeName.py → dictionary.py → [embedding.py] → qdrant.py
Qdrant
```

## 1. Làm sạch dữ liệu thủ công

File đầu vào là:

```text
clean_data/Spec_cate_gia.xlsx
```

Kiểm tra và chỉnh tay các số liệu sai, thiếu đơn vị hoặc sai định dạng. Lưu kết quả vào:

```text
clean_data/Clean_data.xlsx
```

Không chỉnh trực tiếp các JSON trong folder thiết bị vì chúng được tạo lại ở các bước sau.

## 2. Tách dữ liệu và xuất JSON

Chạy notebook:

```text
clean_data/Final.ipynb
```

Notebook đọc `Clean_data.xlsx`, tách dữ liệu theo danh mục vào `clean_data/Final`, sau đó xuất JSON tiếng Việt vào `clean_data/Final_json`.

## 3. Đổi tên và phân phối JSON

Chạy từ thư mục `ingestion`:

```powershell
python clean_data/distribute_final_json.py --dry-run
python clean_data/distribute_final_json.py
```

`--dry-run` kiểm tra đủ 14 file, kiểm tra cú pháp JSON và hiển thị đích đến nhưng không ghi file. Khi chạy thật, script đổi tên và phân phối theo mapping cố định, ví dụ:

```text
Máy lạnh.json              → may_lanh/data/may_lanh.json
Đồng hồ thông minh.json    → dong_ho_thong_minh/data/dong_ho_thong_minh.json
Tủ mát, tủ đông.json       → tu_mat_tu_dong/data/tu_mat_tu_dong.json
```

Nếu xuất hiện file JSON không có trong mapping hoặc thiếu một danh mục, script dừng trước khi thay dữ liệu đích.

## 4. Chạy pipeline cho từng thiết bị

Ví dụ với máy lạnh:

```powershell
python may_lanh/processing.py
python may_lanh/changeName.py
python may_lanh/dictionary.py
python may_lanh/embedding.py
python may_lanh/qdrant.py
```

Thứ tự và output:

| Bước | Input | Output |
|---|---|---|
| `processing.py` | `<dataset>.json` | `<dataset>_processed.json` |
| `changeName.py` | `<dataset>_processed.json` | `<dataset>_processed_vi.json` |
| `dictionary.py` | `<dataset>_processed_vi.json` | `<dataset>_dictionary.json` |
| `embedding.py` | `<dataset>_processed_vi.json` | `<dataset>_embedded.json` |
| `qdrant.py` | `<dataset>_processed_vi.json` (Cloud) hoặc `<dataset>_embedded.json` (local) | Collection trên Qdrant |

Máy giặt, máy sấy quần áo, máy rửa chén, máy nước nóng, micro karaoke,
micro thu âm, đồng hồ thông minh, máy tính để bàn, màn hình máy tính, máy tính
bảng và máy in dùng Qdrant Cloud Inference trực tiếp, nên không cần cài `torch`
hoặc chạy embedding local:

```powershell
python may_giat/processing.py
python may_giat/changeName.py
python may_giat/dictionary.py
python may_giat/qdrant.py
```

Với máy giặt, `qdrant.py` nhận `may_giat_processed_vi.json`, gửi semantic text
cho model `intfloat/multilingual-e5-small` trên Qdrant Cloud và upsert point theo
UUID ổn định. Metadata chuẩn hóa gồm loại máy/lồng, giá, tải giặt, số người,
inverter, khả năng sấy và kích thước cm.

Máy rửa chén chạy tương tự:

```powershell
python may_rua_chen/processing.py
python may_rua_chen/changeName.py
python may_rua_chen/dictionary.py
python may_rua_chen/qdrant.py
```

Collection `mayruachen` dùng UUID ổn định và metadata canonical cho loại lắp
đặt, giá VND, sức chứa theo bữa ăn Việt/bộ châu Âu, lượng nước, độ ồn và kích
thước. `category_scope` tách máy rửa chén khỏi sản phẩm máy sấy chén nằm chung
trong dữ liệu nguồn.

Máy nước nóng dùng Qdrant Cloud Inference và metadata canonical:

```powershell
python may_nuoc_nong/processing.py
python may_nuoc_nong/changeName.py
python may_nuoc_nong/dictionary.py
python may_nuoc_nong/qdrant.py
```

Collection `maynuocnong` dùng UUID ổn định, Qdrant Cloud Inference và vector
cosine 384 chiều. Giá VND được parse từ cột hiển thị thay vì cột sinh sai hệ số;
áp lực nước được chuẩn hóa từ Bar/kPa/MPa. Metadata canonical gồm loại làm nóng,
giá, dung tích, công suất, thời gian làm nóng, bơm trợ lực, vòi sen, IP, safety
tags và kích thước cm.

Máy tính bảng chạy trực tiếp qua Cloud Inference:

```powershell
python may_tinh_bang/processing.py
python may_tinh_bang/changeName.py
python may_tinh_bang/dictionary.py
python may_tinh_bang/qdrant.py
```

Collection `maytinhbang` dùng UUID ổn định và metadata canonical cho giá VND,
hệ điều hành, RAM, bộ nhớ, màn hình, khối lượng, kết nối Wi-Fi/di động, thế hệ
mạng, gọi điện và thẻ nhớ. Giá được parse lại từ cột hiển thị vì cột sinh sẵn
trong dataset đang lớn hơn thực tế 10 lần.

Máy tính để bàn dùng metadata canonical và Qdrant Cloud Inference:

```powershell
python may_tinh_de_ban/processing.py
python may_tinh_de_ban/changeName.py
python may_tinh_de_ban/dictionary.py
python may_tinh_de_ban/qdrant.py
```

Collection `maytinhdeban` dùng UUID ổn định và vector cosine 384 chiều. Giá VND
được parse lại từ cột hiển thị vì cột sinh sẵn đang lớn hơn thực tế 10 lần.
Metadata canonical gồm kiểu all-in-one/bộ máy riêng, CPU vendor, hệ điều hành,
RAM và khả năng nâng cấp, tổng dung lượng/loại ổ, GPU tích hợp/rời, Wi-Fi và màn
hình tích hợp. Xung nhịp CPU, model GPU, cổng và kích thước chỉ dùng để semantic
retrieval/ranking; không được suy thành capability khi nguồn dữ liệu thiếu.
Production upload đọc `may_tinh_de_ban_processed_vi.json`; `embedding.py` chỉ là
local fallback.

Màn hình máy tính dùng metadata canonical và Qdrant Cloud Inference:

```powershell
python man_hinh_may_tinh/processing.py
python man_hinh_may_tinh/changeName.py
python man_hinh_may_tinh/dictionary.py
python man_hinh_may_tinh/qdrant.py
```

Collection `manhinhmaytinh` dùng UUID ổn định và vector cosine 384 chiều. Giá
VND được parse lại từ cột hiển thị vì cột sinh sẵn đang lớn hơn thực tế 10 lần.
Metadata canonical gồm kích thước, độ phân giải, family tấm nền, dạng phẳng/cong,
response time kèm metric, độ sáng, gamut màu, kết nối, feature tags, loa, VESA,
cảm ứng và kích thước vật lý. Dataset hiện không có trường tần số quét độc lập;
không được suy refresh rate từ giới hạn Hz nằm trong mô tả cổng kết nối.

Đồng hồ thông minh cũng chạy trực tiếp qua Cloud Inference:

```powershell
python dong_ho_thong_minh/processing.py
python dong_ho_thong_minh/changeName.py
python dong_ho_thong_minh/dictionary.py
python dong_ho_thong_minh/qdrant.py
```

Collection `donghothongminh` dùng UUID ổn định và metadata canonical cho giá,
tương thích iOS/Android, nhóm màn hình và chất liệu dây, kích thước đeo, khối
lượng, thời lượng pin ở chế độ thường và ATM. Các capability quan trọng được
tách thành `call_mode`, `has_cellular`, `has_gps`, `has_notifications`,
`swim_ready` và `health_feature_tags` thay vì gom chung một field. Không suy thời
lượng từ mAh, không quy đổi IP sang ATM và không suy tương thích chỉ dựa trên hãng.
`embedding.py` được giữ làm công cụ local tùy chọn, không phải input production.

Máy in chạy tương tự:

```powershell
python may_in/processing.py
python may_in/changeName.py
python may_in/dictionary.py
python may_in/qdrant.py
```

Collection `mayin` chuẩn hóa công nghệ laser/phun/nhiệt, màu/đơn sắc, giá VND,
tốc độ, công suất tháng, kết nối, khổ giấy, in hai mặt và kích thước mm. Semantic
text chỉ lấy thông số máy in và không dùng các trường máy lạnh có trong pipeline
cũ.

Micro karaoke dùng Qdrant Cloud Inference và metadata canonical:

```powershell
python micro_karaoke/processing.py
python micro_karaoke/changeName.py
python micro_karaoke/dictionary.py
python micro_karaoke/qdrant.py
```

`qdrant.py` đọc trực tiếp `micro_karaoke_processed_vi.json`, gửi semantic text
cho model `intfloat/multilingual-e5-small` trên Qdrant Cloud và upsert point theo
UUID ổn định. Collection `microkaraoke` dùng vector cosine 384 chiều. Metadata
canonical gồm `brand_key`, loại có dây/không dây, băng tần, giá VND, dải tần số
sóng, dải tần âm thanh, độ méo và năm sản xuất. Các tần số có cảnh báo chất lượng
không được ghi vào field canonical; cảnh báo được giữ trong `data_quality_flags`.
Do phần lớn catalog hiện chưa có giá xác minh, người dùng có thể chọn ngân sách
`open`; nếu chọn mức trần cụ thể thì sản phẩm có giá vượt trần sẽ bị loại, còn
sản phẩm chưa có giá vẫn được giữ làm phương án tham khảo và phải hiển thị cảnh
báo chưa thể xác nhận phù hợp ngân sách.

Micro thu âm dùng Qdrant Cloud Inference và metadata canonical riêng:

```powershell
python micro_thu_am_dien_thoai/processing.py
python micro_thu_am_dien_thoai/changeName.py
python micro_thu_am_dien_thoai/dictionary.py
python micro_thu_am_dien_thoai/qdrant.py
```

Collection `microthuamdienthoai` dùng UUID ổn định và vector cosine 384 chiều.
Metadata canonical gồm loại micro, giá VND, thiết bị tương thích, cổng kết nối,
feature tags, hướng thu, băng tần, số bộ phát/thu, thời lượng và phạm vi truyền.
Các cột sinh sai hoặc không đáng tin cậy không được chép vào canonical metadata;
cảnh báo được giữ trong `data_quality_flags`. Production upload đọc trực tiếp
`micro_thu_am_dien_thoai_processed_vi.json` và dùng Cloud Inference; script
`embedding.py` chỉ là local fallback.

Tủ mát và tủ đông dùng pipeline local embedding đầy đủ:

```powershell
python tu_mat_tu_dong/processing.py
python tu_mat_tu_dong/changeName.py
python tu_mat_tu_dong/dictionary.py
python tu_mat_tu_dong/embedding.py
python tu_mat_tu_dong/qdrant.py
```

Nếu collection đã có vector nhưng payload còn theo schema cũ, có thể cập nhật
payload canonical mà không cần chạy lại embedding:

```powershell
python tu_mat_tu_dong/qdrant.py --reuse-existing-vectors
```

Collection `tumattudong` dùng UUID ổn định và vector cosine 384 chiều. Metadata
canonical tách `product_family` thành `cooler`/`freezer`, giữ loại mini, giá hiệu
lực, dung tích, nhiệt độ, Inverter, gas, kích thước và các `feature_tags` có thể
lọc. Hai family dùng chung collection nhưng luôn được lọc bằng metadata khi nhu
cầu người dùng đã xác định loại tủ.

Máy sấy quần áo dùng metadata canonical và Qdrant Cloud Inference:

```powershell
python may_say_quan_ao/processing.py
python may_say_quan_ao/changeName.py
python may_say_quan_ao/dictionary.py
python may_say_quan_ao/qdrant.py
```

`qdrant.py` đọc trực tiếp `may_say_quan_ao_processed_vi.json` và từ chối upload
nếu metadata canonical còn thiếu, tránh dùng nhầm file embedding cũ. Collection
`maysayquanao` dùng vector cosine 384 chiều. Metadata canonical gồm `dryer_type`,
giá VND, tải sấy kg, khoảng số người, kích thước cm, công suất W, inverter và cảm
biến. Các trường công nghệ/tiện ích tiếng Việt vẫn được giữ cho semantic
retrieval và ranking.

`processing.py` tạo `text` semantic tối đa 480 token và thêm `image_path` ở cấp đối tượng. Ví dụ:

```json
{
  "id": "...",
  "name": "...",
  "text": "...",
  "image_path": "/public/may_lanh.jpg",
  "metadata": {}
}
```

Tên file tương ứng nằm trong `frontend/public`. Frontend có thể dùng trực tiếp
đường dẫn `/public/...`; API cũng phục vụ cùng asset qua `/product-images/...`.

## Cấu hình

- Ảnh danh mục nằm trong `frontend/public` và có tên trùng dataset.
- Model embedding: `intfloat/multilingual-e5-small` hoặc bản model local đã cấu hình.
- Trước khi chạy `qdrant.py`, khai báo `QDRANT_URL` và `QDRANT_API_KEY` trong môi trường hoặc file `.env`.

## Lưu ý vận hành

- Luôn chạy `distribute_final_json.py --dry-run` trước khi ghi đè dữ liệu category.
- Sau khi sửa `Clean_data.xlsx`, phải chạy lại bước tách/xuất JSON và phân phối.
- Chạy các script category đúng thứ tự; không chạy embedding trước processing.
- `qdrant.py` dùng upsert nên cùng một SKU sẽ cập nhật point hiện có.
