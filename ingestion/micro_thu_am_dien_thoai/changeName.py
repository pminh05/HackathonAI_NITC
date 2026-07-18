import json
import re
import sys
from pathlib import Path

DATASET = "micro_thu_am_dien_thoai"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm micro thu âm điện thoại.
KEY_MAPPING = {
    "bang_tan_min_ghz": "Băng tần min GHz",
    "bang_tan_max_ghz": "Băng tần max GHz",
    "thoi_gian_bo_thu_min_gio": "Thời gian bộ thu min giờ",
    "thoi_gian_bo_thu_max_gio": "Thời gian bộ thu max giờ",
    "thoi_gian_su_dung_min_gio": "Thời gian sử dụng min giờ",
    "thoi_gian_su_dung_max_gio": "Thời gian sử dụng max giờ",
    "sac_day_bo_thu_phut": "Sạc đầy bộ thu phút",
    "so_chu_ky_sac": "Số chu kỳ sạc",
    "khoang_cach_truyen_m": "Khoảng cách truyền m",
    "so_mic_phat": "Số mic phát",
    "so_mic_thu": "Số mic thu",
    "loai_san_pham_chuan": "Loại sản phẩm",
    "nam_san_xuat_chuan": "Năm sản xuất",
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
