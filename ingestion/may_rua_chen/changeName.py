import json
import re
import sys
from pathlib import Path

DATASET = "may_rua_chen"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm máy rửa chén.
KEY_MAPPING = {
  "bua_an_viet_min": "Bữa ăn Việt min",
  "bua_an_viet_max": "Bữa ăn Việt max",
  "bo_chau_au_min": "Bộ châu Âu min",
  "bo_chau_au_max": "Bộ châu Âu max",
  "so_chuong_trinh": "Số chương trình",
  "so_cong_nghe_rua": "Số công nghệ rửa",
  "tieu_thu_nuoc_min_lit_lan": "Tiêu thụ nước min lít/lần",
  "tieu_thu_nuoc_max_lit_lan": "Tiêu thụ nước max lít/lần",
  "cong_suat_min_w": "Công suất min W",
  "cong_suat_max_w": "Công suất max W",
  "do_on_db": "Độ ồn dB",
  "cao_cm": "Cao cm",
  "ngang_cm": "Ngang cm",
  "sau_cm": "Sâu cm",
  "khoi_luong_may_kg": "Khối lượng máy kg",
  "nam_dong_san_pham": "Năm dòng sản phẩm",
  "dai_ong_cap_nuoc_cm": "Dài ống cấp nước cm",
  "dai_ong_xa_nuoc_cm": "Dài ống xả nước cm",
  "so_tien_ich": "Số tiện ích",
  "so_cong_nghe_say": "Số công nghệ sấy",
  "so_loai_khay": "Số loại khay"
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
