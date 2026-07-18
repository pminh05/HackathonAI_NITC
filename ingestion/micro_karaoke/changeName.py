import json
import re
import sys
from pathlib import Path

DATASET = "micro_karaoke"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm micro karaoke.
KEY_MAPPING = KEY_MAPPING = {
    "loai_san_pham_chuan": "Loại sản phẩm chuẩn",
    "bang_tan_chuan": "Băng tần chuẩn",
    "loai_du_lieu_tan_so": "Loại dữ liệu tần số",
    "tan_so_song_min_mhz": "Tần số sóng min MHz",
    "tan_so_song_max_mhz": "Tần số sóng max MHz",
    "tan_so_am_thanh_min_hz": "Tần số âm thanh min Hz",
    "tan_so_am_thanh_max_hz": "Tần số âm thanh max Hz",
    "do_meo_tieng_pct": "Độ méo tiếng %",
    "toan_tu_do_meo": "Toán tử độ méo",
    "nam_san_xuat_chuan": "Năm sản xuất chuẩn",
}


def vietnamese_label(key):
    if key in KEY_MAPPING:
        return KEY_MAPPING[key]
    # Giữ nguyên các khóa gốc tiếng Việt; chỉ làm đẹp khóa snake_case.
    if "_" not in key:
        return key
    return re.sub(r"\s+", " ", key.replace("_", " ")).strip().capitalize()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    for item in data:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            item["metadata"] = {
                vietnamese_label(key): value
                for key, value in metadata.items()
            }

    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã chuẩn hóa metadata -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
