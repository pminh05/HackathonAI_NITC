"""Generate a dictionary for canonical desktop Qdrant metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DATASET = "may_tinh_de_ban"
CATEGORY = "máy tính để bàn"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_dictionary.json"

FIELD_DEFINITIONS = {
    "category_scope": ("keyword", "Phạm vi cố định của catalog desktop."),
    "brand_key": ("keyword", "Hãng chuẩn hóa lowercase để lọc chính xác."),
    "original_price_vnd": ("integer", "Giá gốc VND parse từ cột hiển thị."),
    "promotional_price_vnd": ("integer", "Giá khuyến mãi VND đã xác minh."),
    "desktop_form": ("keyword", "Máy liền màn hình hoặc bộ máy dùng màn hình riêng."),
    "cpu_vendor": ("keyword", "Nhà cung cấp CPU được xác nhận từ thông số CPU."),
    "os_family_tags": ("keyword[]", "Các họ hệ điều hành đã chuẩn hóa."),
    "ram_gb": ("integer", "Dung lượng RAM lắp sẵn theo GB."),
    "ram_max_gb": ("integer", "Dung lượng RAM tối đa được nguồn dữ liệu công bố."),
    "storage_total_gb": ("integer", "Tổng dung lượng ổ đã lắp theo GB."),
    "storage_type_tags": ("keyword[]", "Loại ổ đã lắp: NVMe, SSD hoặc HDD."),
    "gpu_type": ("keyword", "Card đồ họa tích hợp hoặc card rời."),
    "has_wifi": ("bool", "Khả năng Wi-Fi khi nguồn dữ liệu xác nhận."),
    "screen_size_inch": ("float", "Kích thước màn hình tích hợp theo inch."),
    "release_year": ("integer", "Năm ra mắt."),
    "data_quality_flags": ("keyword[]", "Cảnh báo chất lượng hoặc phép chuẩn hóa dữ liệu."),
}


def _values(metadata: dict[str, Any], field: str) -> list[Any]:
    value = metadata.get(field)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu desktop đã chuẩn hóa")

    fields: dict[str, Any] = {}
    for field, (data_type, description) in FIELD_DEFINITIONS.items():
        populated = 0
        distinct: set[str] = set()
        for product in products:
            metadata = product.get("metadata") or {}
            values = _values(metadata, field)
            if values:
                populated += 1
                distinct.update(str(value) for value in values)
        entry: dict[str, Any] = {
            "description": description,
            "data_type": data_type,
            "coverage": {
                "populated": populated,
                "total": len(products),
                "percent": round(populated * 100 / len(products), 1),
            },
        }
        if data_type.startswith("keyword") or data_type == "bool":
            entry["possible_values"] = sorted(distinct, key=str.casefold)
        fields[field] = entry

    OUTPUT_FILE.write_text(
        json.dumps(
            {"dataset": DATASET, "category": CATEGORY, "fields": fields},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Đã tạo dictionary -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
