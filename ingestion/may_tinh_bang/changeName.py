import json
import re
import sys
from pathlib import Path

DATASET = "may_tinh_bang"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm máy tính bảng.
KEY_MAPPING = {
  "so_nhan_cpu": "Số nhân CPU",
  "ram_gb": "RAM GB",
  "kich_thuoc_man_hinh_inch": "Kích thước màn hình inch",
  "bo_nho_luu_tru_gb": "Bộ nhớ lưu trữ GB",
  "bo_nho_kha_dung_gb": "Bộ nhớ khả dụng GB",
  "khoi_luong_g": "Khối lượng g",

  "dung_luong_pin_mah": "Dung lượng pin mAh",
  "thoi_gian_ra_mat_chuan": "Thời gian ra mắt",
  "dai_mm": "Dài mm",
  "ngang_mm": "Ngang mm",
  "day_mm": "Dày mm",
  "toc_do_cpu_min_ghz": "Tốc độ CPU min GHz",
  "toc_do_cpu_max_ghz": "Tốc độ CPU max GHz",
  "cong_suat_sac_toi_da_w": "Công suất sạc tối đa W",
  "phien_ban_bluetooth": "Phiên bản Bluetooth",
  "quay_video_toi_da_p": "Quay video tối đa P",
  "quay_video_toi_da_fps": "Quay video tối đa FPS",
  "co_goi_dien": "Có gọi điện",
  "co_ghi_am": "Có ghi âm",
  "co_radio": "Có radio",
  "gia_goc_vnd": "Giá gốc VND"
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
