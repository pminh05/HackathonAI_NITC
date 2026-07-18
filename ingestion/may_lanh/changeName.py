import json
import re
import sys
from pathlib import Path

DATASET = "may_lanh"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm máy lạnh.
KEY_MAPPING = {
  "dien_tich_min_m2": "Diện tích min m2",
  "dien_tich_max_m2": "Diện tích max m2",
  "the_tich_min_m3": "Thể tích min m3",
  "the_tich_max_m3": "Thể tích max m3",
  "bao_hanh_may_nen_nam": "Bảo hành máy nén năm",
  "khoi_luong_phu_kien_chinh_kg": "Khối lượng phụ kiện chính kg",
  "khoi_luong_phu_kien_phu_kg": "Khối lượng phụ kiện phụ kg",

  "ong_dong_min_m": "Ống đồng min m",
  "ong_dong_max_m": "Ống đồng max m",
  "cao_lap_dat_m": "Cao lắp đặt m",
  "do_on_min_db": "Độ ồn min dB",
  "do_on_max_db": "Độ ồn max dB",
  "do_on_dan_lanh_db": "Độ ồn dàn lạnh dB",
  "do_on_dan_nong_db": "Độ ồn dàn nóng dB",
  "ong_dong_nho_mm": "Ống đồng nhỏ mm",
  "ong_dong_lon_mm": "Ống đồng lớn mm",
  "dai_phu_kien_chinh_cm": "Dài phụ kiện chính cm",
  "day_phu_kien_chinh_cm": "Dày phụ kiện chính cm",
  "cao_phu_kien_chinh_cm": "Cao phụ kiện chính cm",
  "cao_phu_kien_phu_cm": "Cao phụ kiện phụ cm",
  "dai_phu_kien_phu_cm": "Dài phụ kiện phụ cm",
  "day_phu_kien_phu_cm": "Dày phụ kiện phụ cm",
  "so_sao_nang_luong": "Số sao năng lượng",
  "hieu_suat_nang_luong": "Hiệu suất năng lượng",
  "vi_tri_cap_dien": "Vị trí cấp điện",
  "so_pha": "Số pha",
  "nam_dong_san_pham": "Năm dòng sản phẩm",
  "bao_hanh_bo_phan_nam": "Bảo hành bộ phận năm",
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
