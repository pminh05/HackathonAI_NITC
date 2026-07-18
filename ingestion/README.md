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
        ↓ processing.py → changeName.py → dictionary.py → embedding.py → qdrant.py
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
| `qdrant.py` | `<dataset>_embedded.json` | Collection trên Qdrant |

Máy giặt dùng Qdrant Cloud Inference trực tiếp, nên không cần cài `torch` hoặc
chạy embedding local:

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

Máy sấy quần áo dùng pipeline local embedding đầy đủ:

```powershell
python may_say_quan_ao/processing.py
python may_say_quan_ao/changeName.py
python may_say_quan_ao/dictionary.py
python may_say_quan_ao/embedding.py
python may_say_quan_ao/qdrant.py
```

Collection `maysayquanao` dùng vector cosine 384 chiều. Metadata canonical gồm
`dryer_type`, giá VND, tải sấy kg, khoảng số người, kích thước cm, công suất W,
inverter và cảm biến. Các trường công nghệ/tiện ích tiếng Việt vẫn được giữ cho
semantic retrieval và ranking.

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
