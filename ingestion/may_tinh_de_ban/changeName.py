import json
import re
import sys
from pathlib import Path

DATASET = "may_tinh_de_ban"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

# Các trường chuẩn hóa quan trọng của riêng nhóm máy tính để bàn.
KEY_MAPPING = {
  "toc_do_cpu_co_ban_ghz": "Tốc độ CPU cơ bản GHz",
  "toc_do_cpu_toi_da_ghz": "Tốc độ CPU tối đa GHz",
  "ram_gb": "RAM GB",
  "bus_ram_mainboard_mhz": "Bus RAM mainboard MHz",
  "so_khe_ram_chuan": "Số khe RAM chuẩn",
  "dung_luong_o_cung_gb": "Dung lượng ổ cứng GB",
  "loai_o_cung_chuan": "Loại ổ cứng chuẩn",
  "nguon_dien_min_w": "Nguồn điện min W",
  "nguon_dien_max_w": "Nguồn điện max W",
  "dai_mm": "Dài mm",
  "rong_mm": "Rộng mm",
  "day_mm": "Dày mm",
  "khoi_luong_kg": "Khối lượng kg",

  "nam_ra_mat": "Năm ra mắt",
  "so_nhan_cpu": "Số nhân CPU",
  "so_luong_cpu_luong": "Số luồng CPU",
  "bo_nho_dem_mb": "Bộ nhớ đệm MB",
  "ram_toi_da_gb": "RAM tối đa GB",
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
