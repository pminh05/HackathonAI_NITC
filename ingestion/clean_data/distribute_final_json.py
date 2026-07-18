"""Đổi tên và phân phối JSON từ clean_data/Final_json vào từng category/data."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


FILE_NAME_TO_DATASET = {
    "Đồng hồ thông minh": "dong_ho_thong_minh",
    "Màn hình máy tính": "man_hinh_may_tinh",
    "Máy giặt": "may_giat",
    "Máy in": "may_in",
    "Máy lạnh": "may_lanh",
    "Máy nước nóng": "may_nuoc_nong",
    "Máy rửa chén": "may_rua_chen",
    "Máy sấy quần áo": "may_say_quan_ao",
    "Máy tính bảng": "may_tinh_bang",
    "Máy tính để bàn": "may_tinh_de_ban",
    "Micro karaoke": "micro_karaoke",
    "Micro thu âm điện thoại": "micro_thu_am_dien_thoai",
    "Tủ lạnh": "tu_lanh",
    "Tủ mát, tủ đông": "tu_mat_tu_dong",
}

CLEAN_DATA_DIR = Path(__file__).resolve().parent
INGESTION_DIR = CLEAN_DATA_DIR.parent
SOURCE_DIR = CLEAN_DATA_DIR / "Final_json"


def validate_json(path: Path) -> None:
    """Dừng trước khi ghi nếu JSON nguồn bị lỗi cú pháp."""
    with path.open("r", encoding="utf-8-sig") as file:
        json.load(file)


def build_jobs() -> list[tuple[Path, Path]]:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {SOURCE_DIR}")

    source_files = {path.stem: path for path in SOURCE_DIR.glob("*.json")}
    missing = sorted(set(FILE_NAME_TO_DATASET) - set(source_files))
    unknown = sorted(set(source_files) - set(FILE_NAME_TO_DATASET))

    if missing:
        raise FileNotFoundError("Thiếu file trong Final_json: " + ", ".join(missing))
    if unknown:
        raise ValueError("Có file JSON chưa được khai báo mapping: " + ", ".join(unknown))

    jobs = []
    for vietnamese_name, dataset in FILE_NAME_TO_DATASET.items():
        source = source_files[vietnamese_name]
        destination = INGESTION_DIR / dataset / "data" / f"{dataset}.json"
        jobs.append((source, destination))
    return jobs


def distribute(dry_run: bool = False) -> None:
    jobs = build_jobs()

    # Kiểm tra toàn bộ nguồn trước, tránh phân phối dở dang khi một file lỗi.
    for source, _ in jobs:
        validate_json(source)

    for source, destination in jobs:
        print(f"{source.name} -> {destination.relative_to(INGESTION_DIR)}")
        if dry_run:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    action = "Kiểm tra" if dry_run else "Phân phối"
    print(f"{action} thành công {len(jobs)} file JSON.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Đổi tên và phân phối JSON sạch vào các category/data."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra và in kế hoạch, không ghi file.",
    )
    args = parser.parse_args()
    distribute(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
