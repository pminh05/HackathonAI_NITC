"""Build category-specific semantic documents for coolers and freezers."""

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


DATASET = "tu_mat_tu_dong"
CATEGORY = "tủ mát, tủ đông"
MAX_TOKENS = 480
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = "/public/tu_mat_tu_dong.jpg"
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"

PRODUCT_TYPES = {
    "tủ mát": ("Tủ mát", "cooler", False),
    "tủ mát mini": ("Tủ mát mini", "cooler", True),
    "tủ đông": ("Tủ đông", "freezer", False),
    "tủ đông mini": ("Tủ đông mini", "freezer", True),
}


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


def split_features(value: Any) -> list[str]:
    if not meaningful(value):
        return []
    return [
        item
        for part in str(value).split("|")
        if (item := clean(part)) is not None
    ]


def load_products() -> list[dict[str, Any]]:
    payload = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data_clean", "Sheet1", "products", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Không tìm thấy danh sách sản phẩm trong {INPUT_FILE}")


def product_type_details(product: dict[str, Any]) -> tuple[str, str, bool]:
    raw_type = clean(
        product.get("loai_san_pham_chuan") or product.get("Loại sản phẩm")
    )
    details = PRODUCT_TYPES.get(str(raw_type or "").casefold())
    if details is None:
        sku = clean(product.get("sku")) or "không xác định"
        raise ValueError(f"Sản phẩm {sku} thiếu loại tủ mát/tủ đông hợp lệ")
    return details


def _format_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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


def _append(sentences: list[str], sentence: str | None) -> None:
    sentence = clean(sentence)
    if sentence and sentence not in sentences:
        sentences.append(sentence)


def create_semantic_text(product: dict[str, Any], sku: str) -> str:
    type_label, family, is_mini = product_type_details(product)
    brand = clean(product.get("brand")) or "không xác định"
    model = clean(product.get("model_code"))

    name_parts = [type_label, str(brand)]
    if model:
        name_parts.append(f"model {model}")
    name_parts.append(f"mã sản phẩm {sku}")
    sentences = [" ".join(name_parts) + "."]

    capacity = clean(product.get("dung_tich_tong_lit"))
    if capacity is not None:
        scale = " nhỏ gọn" if is_mini else ""
        _append(
            sentences,
            f"Dung tích tổng{scale} {_format_number(capacity)} lít.",
        )

    total_compartments = clean(product.get("tong_so_ngan"))
    freezer_compartments = clean(product.get("so_ngan_dong"))
    cooler_compartments = clean(product.get("so_ngan_mat"))
    compartment_parts: list[str] = []
    if total_compartments is not None:
        compartment_parts.append(f"tổng {_format_number(total_compartments)} ngăn")
    if freezer_compartments is not None:
        compartment_parts.append(
            f"{_format_number(freezer_compartments)} ngăn đông"
        )
    if cooler_compartments is not None:
        compartment_parts.append(f"{_format_number(cooler_compartments)} ngăn mát")
    if compartment_parts:
        _append(sentences, "Bố trí " + ", ".join(compartment_parts) + ".")

    raw_temperature = clean(product.get("Nhiệt độ ngăn đông (độ C)"))
    if meaningful(raw_temperature):
        if family == "freezer":
            _append(sentences, f"Nhiệt độ vận hành được công bố: {raw_temperature}.")
        else:
            _append(sentences, f"Khoảng nhiệt độ làm mát được công bố: {raw_temperature}.")

    technology = split_features(product.get("Công nghệ"))
    if technology:
        _append(sentences, "Công nghệ làm lạnh: " + ", ".join(technology) + ".")

    energy_technology = split_features(product.get("Công nghệ tiết kiệm điện"))
    if energy_technology:
        _append(
            sentences,
            "Công nghệ tiết kiệm điện: " + ", ".join(energy_technology) + ".",
        )
    elif product.get("co_inverter") is True:
        _append(sentences, "Có Inverter hỗ trợ vận hành ổn định và tiết kiệm điện.")

    energy_day = clean(product.get("dien_nang_kwh_ngay"))
    if energy_day is not None:
        _append(
            sentences,
            f"Điện năng tiêu thụ tham khảo {_format_number(energy_day)} kWh mỗi ngày.",
        )

    noise_min = clean(product.get("do_on_min_db"))
    noise_max = clean(product.get("do_on_max_db"))
    if noise_min is not None and noise_max is not None:
        _append(
            sentences,
            f"Độ ồn công bố từ {_format_number(noise_min)} đến "
            f"{_format_number(noise_max)} dB.",
        )
    elif noise_max is not None:
        _append(sentences, f"Độ ồn tối đa công bố {_format_number(noise_max)} dB.")

    utilities = split_features(product.get("Tiện ích"))
    if utilities:
        _append(sentences, "Tiện ích: " + ", ".join(utilities) + ".")

    face_material = clean(product.get("Chất liệu mặt"))
    inner_material = clean(product.get("Chất liệu ruột"))
    material_parts: list[str] = []
    if meaningful(face_material):
        material_parts.append(f"mặt tủ {face_material}")
    if meaningful(inner_material):
        material_parts.append(f"lòng tủ {inner_material}")
    if material_parts:
        _append(sentences, "Vật liệu gồm " + ", ".join(material_parts) + ".")

    width = clean(product.get("ngang_cm"))
    height = clean(product.get("cao_cm"))
    depth = clean(product.get("sau_cm"))
    dimensions: list[str] = []
    if width is not None:
        dimensions.append(f"ngang {_format_number(width)} cm")
    if height is not None:
        dimensions.append(f"cao {_format_number(height)} cm")
    if depth is not None:
        dimensions.append(f"sâu {_format_number(depth)} cm")
    if dimensions:
        _append(sentences, "Kích thước " + ", ".join(dimensions) + ".")

    gas = clean(product.get("loai_gas_chuan") or product.get("Loại Gas"))
    if gas:
        _append(sentences, f"Sử dụng gas {gas}.")

    origin = clean(product.get("Sản xuất tại"))
    year = clean(product.get("nam_ra_mat"))
    if origin and year is not None:
        _append(sentences, f"Sản xuất tại {origin}, ra mắt năm {year}.")
    elif origin:
        _append(sentences, f"Sản xuất tại {origin}.")
    elif year is not None:
        _append(sentences, f"Ra mắt năm {year}.")

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

        type_label, _, _ = product_type_details(product)
        brand = clean(product.get("brand")) or "không xác định"
        model = clean(product.get("model_code"))
        name = " ".join(
            part
            for part in (type_label, str(brand), str(model) if model else None, sku)
            if part
        )
        text = create_semantic_text(product, sku)
        if not text:
            raise ValueError(f"Sản phẩm {sku} thiếu semantic text")

        output.append(
            {
                "id": sku,
                "name": name,
                "text": text,
                "image_path": IMAGE_PATH,
                "metadata": create_metadata(product, index),
            }
        )

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã xử lý {len(output)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
