"""Build a value dictionary from canonical micro-karaoke metadata."""

import json
import sys
from pathlib import Path


DATASET = "micro_karaoke"
CATEGORY = "micro karaoke"
DICTIONARY_FIELDS = {
    "microphone_type": "Loại micro chuẩn hóa",
    "wireless_band": "Băng tần không dây chuẩn hóa",
    "frequency_data_type": "Loại dữ liệu tần số",
    "distortion_operator": "Cách diễn giải độ méo tiếng",
}
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_dictionary.json"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu micro karaoke đã chuẩn hóa")
    fields = {}
    for field, description in DICTIONARY_FIELDS.items():
        values = {
            str(value)
            for product in products
            if (value := (product.get("metadata") or {}).get(field)) is not None
        }
        fields[field] = {
            "description": description,
            "data_type": "string",
            "possible_values": sorted(values, key=str.casefold),
        }
    result = {"dataset": DATASET, "category": CATEGORY, "fields": fields}
    OUTPUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã tạo dictionary -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
