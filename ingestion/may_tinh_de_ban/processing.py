"""Build desktop-computer semantic passages and stable raw payloads."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None


CATEGORY = "máy tính để bàn"
DATASET = "may_tinh_de_ban"
GROUPS = [
    (
        "Hiệu năng xử lý",
        [
            "Công nghệ CPU",
            "Loại CPU",
            "toc_do_cpu_co_ban_ghz",
            "toc_do_cpu_toi_da_ghz",
            "so_nhan_cpu",
            "so_luong_cpu_luong",
        ],
    ),
    (
        "Bộ nhớ và khả năng nâng cấp",
        [
            "ram_gb",
            "Loại RAM",
            "ram_toi_da_gb",
            "ram_mainboard_toi_da_gb",
            "so_khe_ram_chuan",
            "bus_ram_mainboard_mhz",
        ],
    ),
    (
        "Lưu trữ và đồ họa",
        [
            "Ổ cứng",
            "dung_luong_o_cung_gb",
            "loai_o_cung_chuan",
            "Thiết kế card",
            "Chip đồ họa (GPU)",
            "vram_gpu_gb",
        ],
    ),
    (
        "Kết nối và phần mềm",
        [
            "Hệ điều hành",
            "Wifi",
            "Cổng kết nối",
            "Cổng giao tiếp",
            "Cổng I/O mặt sau",
        ],
    ),
    (
        "Màn hình tích hợp",
        [
            "Kích thước màn hình",
            "Màn hình hiển thị",
            "Độ phân giải",
            "Màn hình cảm ứng",
            "Webcam",
        ],
    ),
    (
        "Nguồn và kích thước",
        [
            "Nguồn điện",
            "dai_mm",
            "rong_mm",
            "day_mm",
            "khoi_luong_kg",
        ],
    ),
]
MAX_TOKENS = 480
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = "/public/may_tinh_de_ban.jpg"
INPUT_FILE = BASE_DIR / "data" / "may_tinh_de_ban.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"


def clean(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    return value


def meaningful(value: Any) -> bool:
    value = clean(value)
    return value is not None and str(value).casefold() not in {
        "không",
        "không có",
        "none",
        "null",
        "nan",
    }


def load_products() -> list[dict[str, Any]]:
    payload = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data_clean", "Sheet1", "products", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Không tìm thấy danh sách sản phẩm trong {INPUT_FILE}")


def display_value(value: Any) -> str:
    if value is True:
        return "có"
    if value is False:
        return "không"
    return str(clean(value))


def limit_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        shortened = encoding.decode(tokens[:max_tokens])
    else:
        max_characters = max_tokens * 3
        if len(text) <= max_characters:
            return text
        shortened = text[:max_characters]

    last_period = shortened.rfind(".")
    if last_period > 0:
        shortened = shortened[: last_period + 1]
    return shortened.strip()


def product_name(product: dict[str, Any], sku: str) -> str:
    brand = clean(product.get("brand")) or "không xác định"
    model = clean(product.get("model_code"))
    identifier = str(model or sku)
    return f"{CATEGORY.capitalize()} {brand} {identifier}"


def create_semantic_text(product: dict[str, Any], sku: str) -> str:
    sentences = [f"{product_name(product, sku)}, mã sản phẩm {sku}."]
    used: set[tuple[str, str]] = set()

    for heading, fields in GROUPS:
        facts: list[str] = []
        for field in fields:
            value = product.get(field)
            signature = (field, str(value))
            if meaningful(value) and signature not in used:
                used.add(signature)
                label = field.replace("_", " ")
                facts.append(f"{label}: {display_value(value)}")
        if facts:
            sentences.append(f"{heading}: " + "; ".join(facts[:10]) + ".")

    origin = clean(product.get("Sản xuất tại"))
    year = clean(product.get("nam_ra_mat") or product.get("nam_san_xuat_chuan"))
    if origin:
        sentences.append(f"Sản xuất tại {origin}.")
    if year:
        sentences.append(f"Năm sản phẩm: {year}.")
    return limit_tokens(" ".join(sentences))


def create_metadata(product: dict[str, Any], source_index: int) -> dict[str, Any]:
    metadata = {
        key: clean(value)
        for key, value in product.items()
        if clean(value) is not None
    }
    metadata["category"] = CATEGORY
    metadata["source_index"] = source_index
    return metadata


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    products = load_products()
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, product in enumerate(products):
        if not isinstance(product, dict):
            raise ValueError(f"Sản phẩm index {index} không phải object")
        sku = str(clean(product.get("sku")) or "").strip()
        if not sku:
            raise ValueError(f"Sản phẩm index {index} thiếu sku")
        if sku in seen_ids:
            raise ValueError(f"SKU bị trùng: {sku}")
        seen_ids.add(sku)
        output.append(
            {
                "id": sku,
                "name": product_name(product, sku),
                "text": create_semantic_text(product, sku),
                "image_path": IMAGE_PATH,
                "metadata": create_metadata(product, index),
            }
        )

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã xử lý {len(output)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
