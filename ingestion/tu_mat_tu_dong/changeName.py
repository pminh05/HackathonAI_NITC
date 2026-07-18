import json
import re
import sys
from pathlib import Path

DATASET = "tu_mat_tu_dong"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm tủ mát tủ đông.
KEY_MAPPING = {
    "loai_san_pham_chuan": "Loại sản phẩm",
    "thoi_gian_ra_mat_chuan": "Thời gian ra mắt",
    "nam_ra_mat": "Năm ra mắt",
    "dung_tich_tong_lit": "Dung tích tổng lít",
    "dien_nang_kwh_ngay": "Điện năng kWh/ngày",
    "dien_nang_kwh_nam": "Điện năng kWh/năm",
    "co_inverter": "Có Inverter",
    "so_cua_chuan": "Số cửa",
    "cao_cm": "Cao cm",
    "ngang_cm": "Ngang cm",
    "sau_cm": "Sâu cm",
    "khoi_luong_may_kg": "Khối lượng máy kg",
    "do_on_max_db": "Độ ồn max dB",
    "loai_gas_chuan": "Loại gas",
    "dung_tich_ngan_dong_mem_lit": "Dung tích ngăn đông mềm lít",
    "tong_so_ngan": "Tổng số ngăn",
    "so_ngan_dong": "Số ngăn đông",
    "so_ngan_mat": "Số ngăn mát",
    "nhiet_do_min_c": "Nhiệt độ min °C",
    "nhiet_do_max_c": "Nhiệt độ max °C",
    "so_tien_ich": "Số tiện ích",
    "so_cong_nghe": "Số công nghệ",
    "gia_goc_vnd": "Giá gốc VND",
    "dien_nang_kwh_nam_moi_lit": "Điện năng kWh/năm mỗi lít",
    "the_tich_ngoai_m3": "Thể tích ngoài m³",
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
