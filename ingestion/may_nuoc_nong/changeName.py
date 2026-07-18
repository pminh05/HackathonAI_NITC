import json
import re
import sys
from pathlib import Path

DATASET = "may_nuoc_nong"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm máy nước nóng.
KEY_MAPPING = {
  "dung_tich_lit": "Dung tích lít",
  "cong_suat_w": "Công suất W",
  "thoi_gian_lam_nong_min_phut": "Thời gian làm nóng min phút",
  "thoi_gian_lam_nong_max_phut": "Thời gian làm nóng max phút",
  "nhiet_do_toi_da_c": "Nhiệt độ tối đa °C",
  "co_bom_tro_luc": "Có bơm trợ lực",
  "co_kem_voi_sen": "Có kèm vòi sen",
  "cao_cm": "Cao cm",
  "rong_cm": "Rộng cm",
  "day_cm": "Dày cm",
  "khoi_luong_may_kg": "Khối lượng máy kg",
  "ap_luc_nuoc_min_mpa": "Áp lực nước min MPa",
  "ap_luc_nuoc_max_mpa": "Áp lực nước max MPa",
  "chi_so_chong_nuoc_ip": "Chỉ số chống nước IP",
  "nam_dong_san_pham": "Năm dòng sản phẩm",
  "gia_goc_vnd": "Giá gốc VND",
  "kiem_tra_du_lieu": "Kiểm tra dữ liệu"
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
