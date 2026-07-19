"""Build a compact dictionary for normalized printer metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DATASET = "may_in"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_dictionary.json"
FIELDS = {
    "brand": "Hãng máy in.",
    "print_technology": "Công nghệ in đã chuẩn hóa.",
    "color_mode": "Khả năng in màu hoặc đơn sắc.",
    "connection_tags": "Các phương thức kết nối đã chuẩn hóa.",
    "paper_size_tags": "Các khổ giấy được nhận diện.",
    "os_tags": "Các hệ điều hành tương thích được nhận diện.",
}


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split("|") if part.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    result: dict[str, Any] = {
        "dataset": DATASET,
        "category": "máy in",
        "fields": {},
    }
    for field, description in FIELDS.items():
        values: set[str] = set()
        for product in products:
            values.update(_values((product.get("metadata") or {}).get(field)))
        result["fields"][field] = {
            "description": description,
            "data_type": "string",
            "possible_values": sorted(values, key=str.casefold),
        }
    OUTPUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã tạo dictionary -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
