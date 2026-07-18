import json

KEY_MAPPING = {
    "kieu_dang_chuan": "Kiểu dáng chuẩn",
    "thoi_gian_ra_mat_chuan": "Thời gian ra mắt chuẩn",
    "nam_ra_mat": "Năm ra mắt",

    "dung_tich_tong_lit": "Dung tích tổng lít",
    "dung_tich_ngan_da_lit": "Dung tích ngăn đá lít",
    "dung_tich_ngan_lanh_lit": "Dung tích ngăn lạnh lít",
    "dung_tich_su_dung_lit": "Dung tích sử dụng lít",

    "dien_nang_kwh_nam": "Điện năng kWh năm",
    "dien_nang_kwh_ngay": "Điện năng kWh ngày",

    "so_nguoi_min": "Số người tối thiểu",
    "so_nguoi_max": "Số người tối đa",

    "so_cua_chuan": "Số cửa chuẩn",

    "cao_cm": "Cao cm",
    "ngang_cm": "Ngang cm",
    "sau_cm": "Sâu cm",

    "khoi_luong_may_kg": "Khối lượng máy kg",

    "co_lay_nuoc_ngoai": "Có lấy nước ngoài",
    "co_che_do_tu_dong": "Có chế độ tự động",
    "co_inverter": "Có inverter",

    "so_cong_nghe_lam_lanh": "Số công nghệ làm lạnh",
    "so_cong_nghe_tiet_kiem_dien": "Số công nghệ tiết kiệm điện",
    "so_cong_nghe_bao_quan": "Số công nghệ bảo quản",
    "so_tien_ich": "Số tiện ích",

    "gia_goc_vnd": "Giá gốc vnd",
    "gia_khuyen_mai_vnd": "Giá khuyến mãi vnd",

    "tong_dung_tich_cac_ngan_da_biet_lit": "Tổng dung tích các ngăn đã biết lít",
    "ty_le_ngan_da_pct": "Tỷ lệ ngăn đá pct",
    "dien_nang_kwh_nam_moi_lit": "Điện năng kWh năm mỗi lít",

    "phan_tram_giam_gia": "Phần trăm giảm giá",

    "dung_tich_ngan_chuyen_doi_lit": "Dung tích ngăn chuyển đổi lít",
}

INPUT_FILE = "data/tu_lanh_processed.json"
OUTPUT_FILE = "data/tu_lanh_processed_vi.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "metadata" not in item:
        continue

    metadata = item["metadata"]

    new_metadata = {}
    for key, value in metadata.items():
        new_key = KEY_MAPPING.get(key, key)
        new_metadata[new_key] = value

    item["metadata"] = new_metadata

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Đã lưu file: {OUTPUT_FILE}")