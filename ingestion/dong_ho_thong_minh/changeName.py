import json
import re
import sys
from pathlib import Path

DATASET = "dong_ho_thong_minh"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm đồng hồ thông minh.
KEY_MAPPING = {
    "kich_thuoc_man_hinh_inch": "Kích thước màn hình inch",
    "do_phan_giai_chuan": "Độ phân giải chuẩn",
    "khoi_luong_g": "Khối lượng g",

      "kich_thuoc_mat_mm": "Chiều dài x rộng x dày mm",
      "dai_mm": "Chiều dài mm",
      "ngang_mm": "Chiều rộng mm",
      "day_mm": "Chiều dày mm",
      "do_rong_day_cm": "Độ rộng dây cm",
      "chu_vi_co_tay_min_cm": "Chu vi cổ tay min cm",
      "chu_vi_co_tay_max_cm": "Chu vi cổ tay max cm",
      "khoi_luong_g": "Khối lượng g",
      "dung_luong_pin_mah": "Dung lượng pin mAh",
      "thoi_gian_sac_gio": "Thời gian sạc giờ",
      "bo_nho_mb": "Bộ nhớ MB",
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
