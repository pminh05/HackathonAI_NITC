"""Build a compact dictionary for cooler/freezer descriptive metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


DATASET = "tu_mat_tu_dong"
CATEGORY = "tủ mát, tủ đông"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_dictionary.json"

FIELDS = {
    "product_type": {
        "source_fields": ["product_type"],
        "description": "Loại canonical của tủ mát hoặc tủ đông.",
        "split_by_pipe": False,
    },
    "gas_type": {
        "source_fields": ["gas_type"],
        "description": "Loại gas làm lạnh được công bố.",
        "split_by_pipe": False,
    },
    "feature_tags": {
        "source_fields": ["feature_tags"],
        "description": "Các tính năng canonical dùng cho tìm kiếm và lọc.",
        "split_by_pipe": False,
    },
    "Công nghệ": {
        "source_fields": ["Công nghệ"],
        "description": "Công nghệ làm lạnh hoặc bảo quản của sản phẩm.",
        "split_by_pipe": True,
    },
    "Tiện ích": {
        "source_fields": ["Tiện ích"],
        "description": "Tiện ích sử dụng của tủ mát hoặc tủ đông.",
        "split_by_pipe": True,
    },
    "Chất liệu mặt": {
        "source_fields": ["Chất liệu mặt"],
        "description": "Vật liệu hoặc kiểu hoàn thiện mặt tủ.",
        "split_by_pipe": True,
    },
    "Chất liệu ruột": {
        "source_fields": ["Chất liệu ruột"],
        "description": "Vật liệu bên trong tủ.",
        "split_by_pipe": True,
    },
}


def clean(value: Any) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def _values(metadata: dict[str, Any], config: dict[str, Any]) -> list[str]:
    raw: Any = None
    for key in config["source_fields"]:
        if metadata.get(key) is not None:
            raw = metadata[key]
            break
    if isinstance(raw, list):
        return [value for item in raw if (value := clean(item))]
    value = clean(raw)
    if not value:
        return []
    if config["split_by_pipe"]:
        return [part for item in value.split("|") if (part := clean(item))]
    return [value]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu tủ mát, tủ đông đã chuẩn hóa")

    fields: dict[str, Any] = {}
    for output_field, config in FIELDS.items():
        possible_values: set[str] = set()
        for product in products:
            metadata = product.get("metadata") or {}
            possible_values.update(_values(metadata, config))
        fields[output_field] = {
            "description": config["description"],
            "data_type": "string",
            "possible_values": sorted(possible_values, key=str.casefold),
        }

    result = {"dataset": DATASET, "category": CATEGORY, "fields": fields}
    OUTPUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã tạo dictionary -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
