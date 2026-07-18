import json
import re
import sys
from pathlib import Path

DATASET = "may_tinh_de_ban"
CATEGORY = "máy tính để bàn"
DICTIONARY_FIELDS = ["Công nghệ CPU","Loại CPU","Loại RAM","Ổ cứng","Thiết kế card","Hệ điều hành"]
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_dictionary.json"


def clean(value):
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def candidate_keys(field):
    return [field, field.replace("_", " ").capitalize()]


def find_value(metadata, field):
    for key in candidate_keys(field):
        value = clean(metadata.get(key))
        if value:
            return value
    return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    fields = {}

    for field in DICTIONARY_FIELDS:
        values = set()
        for product in products:
            metadata = product.get("metadata", {})
            value = find_value(metadata, field)
            if value and value.casefold() not in {"không", "không có", "none"}:
                for part in value.split("|"):
                    part = clean(part)
                    if part:
                        values.add(part)

        fields[field] = {
            "description": (
                f"Các giá trị {field.lower()} dùng để tìm kiếm, "
                f"lọc và hiểu sản phẩm {CATEGORY}."
            ),
            "data_type": "string",
            "possible_values": sorted(values, key=str.casefold),
        }

    result = {
        "dataset": DATASET,
        "category": CATEGORY,
        "fields": fields,
    }
    OUTPUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã tạo dictionary -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
