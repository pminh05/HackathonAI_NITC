import json
import re
import sys
from pathlib import Path

DATASET = "may_in"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm máy in.
KEY_MAPPING = {
    "dai_mm": "Dài mm",
    "rong_mm": "Rộng mm",
    "cao_mm": "Cao mm",
    "khoi_luong_kg": "Khối lượng kg",
    "bo_nho_mb": "Bộ nhớ MB",
    "do_phan_giai_chuan": "Độ phân giải",
    "do_phan_giai_ngang_dpi": "Độ phân giải ngang DPI",
    "do_phan_giai_doc_dpi": "Độ phân giải dọc DPI",
    "thoi_gian_trang_dau_giay": "Thời gian in trang đầu giây",
    "toc_do_in_trang_phut": "Tốc độ in trang/phút",
    "cong_suat_thang_min_trang": "Công suất tháng min trang",
    "cong_suat_thang_max_trang": "Công suất tháng max trang",
    "so_trang_muc_min": "Số trang mực min",
    "so_trang_muc_max": "Số trang mực max",
    "khay_giay_ra_so_to": "Khay giấy ra số tờ",
    "khay_nap_giay_so_to": "Khay nạp giấy số tờ",
    "nam_ra_mat": "Năm ra mắt",
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
