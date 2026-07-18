import json
import re
import sys
from pathlib import Path

DATASET = "may_say_quan_ao"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm máy sấy quần áo.
KEY_MAPPING = {
  "khoi_luong_say_kg": "Khối lượng sấy (kg)",
  "so_nguoi_min": "Số người sử dụng tối thiểu",
  "so_nguoi_max": "Số người sử dụng tối đa",
  "tai_say_khuyen_nghi_min_kg": "Tải sấy khuyến nghị tối thiểu (kg)",
  "tai_say_khuyen_nghi_max_kg": "Tải sấy khuyến nghị tối đa (kg)",
  "nhiet_do_min_c": "Nhiệt độ tối thiểu (°C)",
  "nhiet_do_max_c": "Nhiệt độ tối đa (°C)",
  "cong_suat_min_w": "Công suất tối thiểu (W)",
  "cong_suat_max_w": "Công suất tối đa (W)",
  "cao_cm": "Chiều cao (cm)",
  "ngang_cm": "Chiều ngang (cm)",
  "sau_cm": "Chiều sâu (cm)",
  "khoi_luong_may_kg": "Khối lượng máy (kg)",
  "nam_dong_san_pham": "Năm dòng sản phẩm",
  "dai_ong_thoat_khi_cm": "Chiều dài ống thoát khí (cm)",
  "so_tien_ich": "Số tiện ích"
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
