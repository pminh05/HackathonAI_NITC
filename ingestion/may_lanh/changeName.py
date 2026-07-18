import json
import re
import sys
from decimal import Decimal, InvalidOperation
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
    "cong_suat_lanh_btu_h": "Công suất lạnh BTU/h",
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
    "gia_goc_vnd": "Giá gốc vnd",
    "gia_khuyen_mai_vnd": "Giá khuyến mãi vnd",
}

INTEGER_METADATA_KEYS = {
    "dien_tich_min_m2",
    "dien_tich_max_m2",
    "the_tich_min_m3",
    "the_tich_max_m3",
    "cong_suat_lanh_btu_h",
    "so_sao_nang_luong",
    "bao_hanh_bo_phan_nam",
    "nam_dong_san_pham",
    "so_pha",
}

FLOAT_METADATA_KEYS = {
    "hieu_suat_nang_luong",
    "do_on_min_db",
    "do_on_max_db",
    "do_on_dan_lanh_db",
    "do_on_dan_nong_db",
    "bao_hanh_may_nen_nam",
    "ong_dong_min_m",
    "ong_dong_max_m",
    "cao_lap_dat_m",
}



def vietnamese_label(key):
    if key in KEY_MAPPING:
        return KEY_MAPPING[key]
    # Giữ nguyên các khóa gốc tiếng Việt; chỉ làm đẹp khóa snake_case.
    if "_" not in key:
        return key
    return re.sub(r"\s+", " ", key.replace("_", " ")).strip().capitalize()


def parse_vnd(value):
    """Parse a displayed VND value without trusting the faulty derived column."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return int(Decimal(str(value)))

    text = re.sub(r"(?i)(vnd|₫|đ)", "", str(value)).strip()
    if not text:
        return None
    try:
        return int(Decimal(text.replace(",", "")))
    except InvalidOperation:
        digits = re.sub(r"\D", "", text)
        return int(digits) if digits else None


def normalize_metadata(metadata):
    """Return stable, typed metadata consumed by the advisor and Qdrant indexes."""
    normalized = {}
    for key, value in metadata.items():
        if key in INTEGER_METADATA_KEYS and isinstance(value, (int, float)):
            value = int(value)
        elif key in FLOAT_METADATA_KEYS and isinstance(value, (int, float)):
            value = float(value)
        normalized[vietnamese_label(key)] = value

    # The current source's derived price columns are ten times the displayed
    # prices. Prefer the source display values and keep the canonical field name
    # stable for both local data and Qdrant payload indexes.
    original_price = parse_vnd(metadata.get("giá gốc"))
    if original_price is None:
        original_price = parse_vnd(metadata.get("gia_goc_vnd"))
    promotional_price = parse_vnd(metadata.get("giá khuyến mãi"))
    if promotional_price is None:
        promotional_price = parse_vnd(metadata.get("gia_khuyen_mai_vnd"))
    if original_price is not None:
        normalized["Giá gốc vnd"] = original_price
    if promotional_price is not None:
        normalized["Giá khuyến mãi vnd"] = promotional_price
    return normalized


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    for item in data:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            item["metadata"] = normalize_metadata(metadata)

    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã chuẩn hóa metadata -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
