"""Build semantic documents for the recording-microphone catalog."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None


CATEGORY = "micro thu âm"
DATASET = "micro_thu_am_dien_thoai"
MAX_TOKENS = 480
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = "/public/micro_thu_am_dien_thoai.jpg"
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"


def clean(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    return value


def meaningful(value: Any) -> bool:
    value = clean(value)
    return value is not None and str(value).casefold() not in {
        "không",
        "không có",
        "none",
        "null",
        "nan",
    }


def load_products() -> list[dict[str, Any]]:
    payload = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data_clean", "Sheet1", "products", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Không tìm thấy danh sách sản phẩm trong {INPUT_FILE}")


def limit_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        shortened = encoding.decode(tokens[:max_tokens])
    else:
        max_characters = max_tokens * 3
        if len(text) <= max_characters:
            return text
        shortened = text[:max_characters]
    last_period = shortened.rfind(".")
    if last_period > 0:
        shortened = shortened[: last_period + 1]
    return shortened.strip()


def _sentence(label: str, value: Any) -> str | None:
    return f"{label}: {clean(value)}." if meaningful(value) else None


def create_semantic_text(product: dict[str, Any], sku: str) -> str:
    """Describe microphone facts without relying on faulty derived columns."""
    brand = clean(product.get("brand")) or "không xác định"
    model = clean(product.get("model_code"))
    introduction = f"Micro thu âm {brand}"
    if model:
        introduction += f" model {model}"
    introduction += f", mã sản phẩm {sku}."
    sentences = [introduction]

    fields = (
        ("Loại sản phẩm", product.get("Loại sản phẩm")),
        ("Thiết bị tương thích", product.get("Tương thích")),
        ("Cổng kết nối", product.get("Cổng tai nghe, headphone")),
        ("Tính năng", product.get("Tính năng cơ bản")),
        ("Hướng thu âm", product.get("Hướng thu âm")),
        ("Thời gian sử dụng", product.get("Thời gian sử dụng")),
        ("Khoảng cách truyền", product.get("Khoảng cách truyền")),
        ("Băng tần", product.get("Băng tần")),
        ("Phụ kiện đi kèm", product.get("Phụ kiện đi kèm")),
    )
    sentences.extend(
        sentence
        for label, value in fields
        if (sentence := _sentence(label, value)) is not None
    )

    origin = _sentence("Sản xuất tại", product.get("Sản xuất tại"))
    year = _sentence(
        "Năm sản phẩm",
        product.get("Năm sản xuất") or product.get("nam_san_xuat_chuan"),
    )
    if origin:
        sentences.append(origin)
    if year:
        sentences.append(year)
    return limit_tokens(" ".join(sentences))


def create_metadata(product: dict[str, Any], source_index: int) -> dict[str, Any]:
    metadata = {
        key: cleaned
        for key, value in product.items()
        if (cleaned := clean(value)) is not None
    }
    metadata["category"] = CATEGORY
    metadata["source_index"] = source_index
    return metadata


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    products = load_products()
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, product in enumerate(products):
        if not isinstance(product, dict):
            raise ValueError(f"Sản phẩm index {index} không phải object")
        sku = str(clean(product.get("sku")) or "").strip()
        if not sku:
            raise ValueError(f"Sản phẩm index {index} thiếu sku")
        if sku in seen_ids:
            raise ValueError(f"SKU bị trùng: {sku}")
        seen_ids.add(sku)
        brand = clean(product.get("brand")) or "không xác định"
        model = clean(product.get("model_code"))
        name = f"Micro thu âm {brand}"
        if model:
            name += f" {model}"
        output.append(
            {
                "id": sku,
                "name": name,
                "text": create_semantic_text(product, sku),
                "image_path": IMAGE_PATH,
                "metadata": create_metadata(product, index),
            }
        )
    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã xử lý {len(output)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
