"""Build a data dictionary for raw and canonical smartwatch metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


DATASET = "dong_ho_thong_minh"
CATEGORY = "đồng hồ thông minh"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_dictionary.json"
RAW_FIELDS = [
    "Màn hình hiển thị",
    "Độ phân giải",
    "Chất liệu dây",
    "Kết nối",
    "Tương thích",
    "Theo dõi sức khoẻ",
    "Chuẩn chống nước, bụi",
]
CANONICAL_FIELDS = {
    "category_scope": ("keyword", "Slug phạm vi collection, luôn là smartwatch."),
    "brand_key": ("keyword", "Tên hãng casefold dùng cho hard filter."),
    "original_price_vnd": ("integer", "Giá gốc đã chuẩn hóa theo VND."),
    "promotional_price_vnd": ("integer", "Giá khuyến mãi đã chuẩn hóa theo VND."),
    "compatible_platforms": ("keyword[]", "Nền tảng được nguồn xác nhận: ios/android."),
    "call_mode": (
        "keyword",
        "Kiểu cuộc gọi: none, on_wrist hoặc standalone.",
    ),
    "has_cellular": ("bool", "Nguồn SIM xác nhận có kết nối di động."),
    "has_gps": ("bool", "Nguồn định vị xác nhận GPS/GNSS vệ tinh."),
    "has_notifications": ("bool", "Nguồn xác nhận khả năng hiển thị thông báo."),
    "swim_ready": ("bool", "Mô tả chống nước nói rõ dùng khi bơi hoặc lặn."),
    "has_sos": ("bool", "Tiện ích nguồn ghi rõ SOS hoặc cuộc gọi khẩn cấp."),
    "health_feature_tags": (
        "keyword[]",
        "Tính năng sức khỏe có bằng chứng trực tiếp trong catalog.",
    ),
    "display_family": ("keyword", "Nhóm công nghệ màn hình chuẩn hóa."),
    "strap_material_family": ("keyword", "Nhóm chất liệu dây chuẩn hóa."),
    "screen_size_inch": ("float", "Kích thước màn hình theo inch."),
    "case_length_mm": ("float", "Chiều dài mặt đồng hồ theo mm."),
    "case_width_mm": ("float", "Chiều rộng mặt đồng hồ theo mm."),
    "case_thickness_mm": ("float", "Độ dày mặt đồng hồ theo mm."),
    "weight_g": ("float", "Khối lượng theo gram."),
    "wrist_min_cm": ("float", "Chu vi cổ tay nhỏ nhất theo cm."),
    "wrist_max_cm": ("float", "Chu vi cổ tay lớn nhất theo cm."),
    "typical_battery_hours": (
        "float",
        "Thời lượng chế độ thường/smartwatch; không dùng GPS hay tiết kiệm pin.",
    ),
    "battery_mah": (
        "integer",
        "Dung lượng pin theo mAh, không dùng để suy thời lượng.",
    ),
    "charging_time_hours": ("float", "Thời gian sạc được chuẩn hóa theo giờ."),
    "water_resistance_atm": ("float", "Mức ATM được ghi trực tiếp trong nguồn."),
}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _values(products: list[dict[str, Any]], field: str) -> list[str]:
    values: set[str] = set()
    for product in products:
        metadata = product.get("metadata") or {}
        value = metadata.get(field)
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            cleaned = _clean(entry)
            if cleaned and cleaned.casefold() not in {"không", "không có", "none"}:
                if field in RAW_FIELDS:
                    values.update(
                        part for part in (_clean(item) for item in cleaned.split("|")) if part
                    )
                else:
                    values.add(cleaned)
    return sorted(values, key=str.casefold)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu đồng hồ thông minh đã chuẩn hóa")

    fields: dict[str, dict[str, Any]] = {}
    for field in RAW_FIELDS:
        fields[field] = {
            "description": f"Giá trị mô tả gốc cho {field.lower()}.",
            "data_type": "string",
            "possible_values": _values(products, field),
        }
    for field, (data_type, description) in CANONICAL_FIELDS.items():
        fields[field] = {
            "description": description,
            "data_type": data_type,
            "possible_values": (
                _values(products, field)
                if data_type in {"keyword", "keyword[]"}
                else []
            ),
        }

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
