"""Build semantic documents for the micro-karaoke catalog."""

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


CATEGORY = "micro karaoke"
DATASET = "micro_karaoke"
MAX_TOKENS = 480
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = "/public/micro_karaoke.jpg"
INPUT_FILE = BASE_DIR / "data" / "micro_karaoke.json"
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


def create_semantic_text(product: dict[str, Any], sku: str) -> str:
    brand = clean(product.get("brand")) or "không xác định"
    model = clean(product.get("model_code"))
    introduction = f"Micro karaoke {brand}"
    if model:
        introduction += f" model {model}"
    introduction += f", mã sản phẩm {sku}."
    sentences = [introduction]

    microphone_type = clean(
        product.get("loai_san_pham_chuan") or product.get("Loại sản phẩm")
    )
    band = clean(product.get("bang_tan_chuan") or product.get("Băng tần"))
    if meaningful(microphone_type):
        sentence = f"Đây là micro {str(microphone_type).casefold()}"
        if meaningful(band):
            sentence += f", sử dụng băng tần {band}"
        sentences.append(sentence + ".")

    warning = str(clean(product.get("kiem_tra_du_lieu")) or "").casefold()
    if "chưa tách được tần số sóng" not in warning and "có dây" not in warning:
        rf_min = clean(product.get("tan_so_song_min_mhz"))
        rf_max = clean(product.get("tan_so_song_max_mhz"))
        if rf_min is not None and rf_max is not None:
            sentences.append(f"Dải tần số sóng từ {rf_min} đến {rf_max} MHz.")

    if "dải tần âm thanh bất thường" not in warning:
        audio_min = clean(product.get("tan_so_am_thanh_min_hz"))
        audio_max = clean(product.get("tan_so_am_thanh_max_hz"))
        if audio_min is not None and audio_max is not None:
            sentences.append(f"Dải tần âm thanh từ {audio_min} đến {audio_max} Hz.")

    distortion = clean(product.get("do_meo_tieng_pct"))
    operator = clean(product.get("toan_tu_do_meo"))
    if distortion is not None:
        prefix = f"{operator} " if meaningful(operator) else ""
        sentences.append(f"Độ méo tiếng {prefix}{distortion}%.")

    origin = clean(product.get("Sản xuất tại"))
    year = clean(product.get("nam_san_xuat_chuan") or product.get("Năm sản xuất"))
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
        brand = clean(product.get("brand")) or "không xác định"
        output.append(
            {
                "id": sku,
                "name": f"Micro karaoke {brand} {sku}",
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
