import json
import re
import sys
from pathlib import Path

DATASET = "man_hinh_may_tinh"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm màn hình máy tính.
KEY_MAPPING = {
    "kich_thuoc_man_hinh_inch": "Kích thước màn hình inch",
    "do_phan_giai_chuan": "Độ phân giải chuẩn",
    "do_sang_cd_m2": "Độ sáng Cd M2",
    "so_luong_mau": "Số lượng màu",
    "tan_so_quet_hz": "ần số quét Hz",
    "thoi_gian_dap_ung_ms": "Thời gian đáp ứng Ms",
    "loai_thoi_gian_dap_ung": "Loại thời gian đáp ứng",
    "ngang_mm": "Ngang mm",
    "cao_min_mm": "Cao Min mm",
    "cao_max_mm": "Cao Max mm",
    "day_mm": "Dày mm",
    "khoi_luong_kg": "Khối lượng kg",
    "dien_nang_tieu_thu_w": "Điện năng tiêu thụ W",
    "do_phan_giai_ngang_px": "Độ phân giải ngang px",
    "do_phan_giai_doc_px": "Độ phân giải dọc px",
    "ngang_module_phu_mm": "Ngang module phụ mm",
    "cao_khong_chan_mm": "Cao không chăn mm",
    "do_day_khong_chan_mm": "Độ dày không chăn mm",
    "do_tuong_phan_tinh": "Độ tường phân tinh",
    "do_phu_srgb_pct": "Độ phủ sRGB %",
    "gia_goc_vnd": "Giá gốc VND",
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
