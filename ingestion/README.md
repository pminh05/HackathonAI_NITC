# Ingestion Pipeline

Pipeline xử lý dữ liệu sản phẩm để đưa vào Qdrant phục vụ RAG.

## Cấu trúc thư mục

```
ingestion/
├── <category>/
│   ├── data/
│   │   ├── <dataset>.json
│   │   ├── <dataset>_processed.json
│   │   ├── <dataset>_processed_vi.json
│   │   ├── <dataset>_dictionary.json
│   │   └── <dataset>_embedded.json
│   ├── processing.py
│   ├── changeName.py
│   ├── dictionary.py
│   ├── embedding.py
│   └── qdrant.py
└── README.md
```

---

# Quy trình xử lý

Chạy các script theo đúng thứ tự:

```
processing.py
      ↓
changeName.py
      ↓
dictionary.py
      ↓
embedding.py
      ↓
qdrant.py
```

---

## 1. processing.py

Đọc dữ liệu gốc

```
data/<dataset>.json
```

Chuẩn hóa dữ liệu:

- Làm sạch metadata
- Sinh trường `text`
- Chuẩn hóa cấu trúc dữ liệu

Output:

```
data/<dataset>_processed.json
```

---

## 2. changeName.py

Đổi tên các thuộc tính metadata từ tiếng Anh sang tiếng Việt.

Ví dụ:

```
ram_gb
↓
RAM GB
```

Output:

```
data/<dataset>_processed_vi.json
```

---

## 3. dictionary.py

Sinh dictionary mô tả các trường metadata.

Ví dụ:

```json
{
    "RAM GB": {
        "description": "...",
        "possible_values": [...]
    }
}
```

Output:

```
data/<dataset>_dictionary.json
```

---

## 4. embedding.py

Sinh vector embedding cho toàn bộ dữ liệu.

Input:

```
data/<dataset>_processed_vi.json
```

Output:

```
data/<dataset>_embedded.json
```

### Model

Project sử dụng model:

```
models/
└── multilingual-e5-small/
```

Model cần được tải trước khi chạy.

---

## 5. qdrant.py

Upload dữ liệu đã embedding lên Qdrant.

Input:

```
data/<dataset>_embedded.json
```

Output:

- Collection trên Qdrant

---

# File dữ liệu

| File                          | Vai trò                         |
| ----------------------------- | ------------------------------- |
| `<dataset>.json`              | Dữ liệu gốc                     |
| `<dataset>_processed.json`    | Dữ liệu đã chuẩn hóa            |
| `<dataset>_processed_vi.json` | Metadata đã đổi sang tiếng Việt |
| `<dataset>_dictionary.json`   | Dictionary thuộc tính           |
| `<dataset>_embedded.json`     | Dữ liệu đã embedding            |

---

# Lưu ý

- Luôn chạy đúng thứ tự các script.
- Model `multilingual-e5-small` phải được tải sẵn trong thư mục:

```
models/multilingual-e5-small/
```

- Cần cấu hình Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`) trước khi chạy `qdrant.py`.
