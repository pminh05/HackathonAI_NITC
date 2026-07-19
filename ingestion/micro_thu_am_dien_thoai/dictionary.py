"""Build a dictionary from canonical recording-microphone metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DATASET = "micro_thu_am_dien_thoai"
CATEGORY = "micro thu âm"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_dictionary.json"

FIELDS = {
    "product_type": "Loại micro chuẩn hóa.",
    "compatibility_tags": "Thiết bị và hệ điều hành tương thích.",
    "connector_tags": "Cổng kết nối có bằng chứng trong catalog.",
    "feature_tags": "Tính năng micro đã được chuẩn hóa.",
    "pickup_pattern": "Hướng hoặc kiểu thu âm.",
    "wireless_band": "Băng tần không dây chuẩn hóa.",
}


def _values(metadata: dict[str, Any], field: str) -> list[str]:
    value = metadata.get(field)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value is not None and str(value).strip() else []


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu micro thu âm đã chuẩn hóa")
    result_fields: dict[str, Any] = {}
    for field, description in FIELDS.items():
        possible_values = {
            value
            for product in products
            for value in _values(product.get("metadata") or {}, field)
        }
        result_fields[field] = {
            "description": description,
            "data_type": "string",
            "possible_values": sorted(possible_values, key=str.casefold),
        }
    OUTPUT_FILE.write_text(
        json.dumps(
            {"dataset": DATASET, "category": CATEGORY, "fields": result_fields},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Đã tạo dictionary -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
