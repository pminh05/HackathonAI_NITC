"""Build a compact dictionary for washing-machine descriptive metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "may_giat_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / "may_giat_dictionary.json"
FIELDS = {
    "product_type": "Loại máy giặt đã chuẩn hóa.",
    "drum_type": "Kiểu lồng đứng hoặc lồng ngang đã chuẩn hóa.",
    "Loại Inverter": "Tên công nghệ Inverter theo dữ liệu nguồn.",
    "Công nghệ": "Các công nghệ giặt được công bố.",
    "Công nghệ sấy": "Công nghệ sấy được công bố.",
    "Tiện ích": "Các tiện ích của máy giặt.",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    result: dict[str, object] = {"dataset": "may_giat", "fields": {}}
    for field, description in FIELDS.items():
        values: set[str] = set()
        for product in products:
            value = (product.get("metadata") or {}).get(field)
            if value is None:
                continue
            for item in str(value).split("|"):
                item = item.strip()
                if item:
                    values.add(item)
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
