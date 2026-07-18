import json
import re
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    tiktoken = None

CATEGORY = "máy in"
DATASET = "may_in"
GROUPS = [["Công nghệ và chức năng in",["Loại máy","Công nghệ in","Chức năng","In hai mặt","In màu","Độ phân giải in"]],["Hiệu suất in",["Tốc độ in","Khổ giấy","Công suất in","Khay giấy"]],["Kết nối",["Kết nối","Wifi","Cổng giao tiếp","Hệ điều hành tương thích"]],["Thông số khác có trong nguồn",["Phạm vi sử dụng","Công nghệ làm lạnh","Loại Inverter","dien_tich_min_m2","dien_tich_max_m2","Điện năng tiêu thụ"]]]
MAX_TOKENS = 480
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = str((BASE_DIR.parent.parent / "public" / f"{DATASET}.jpg").resolve())
INPUT_FILE = BASE_DIR / "data" / "may_in.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"


def clean(value):
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    return value


def meaningful(value):
    value = clean(value)
    return value is not None and str(value).casefold() not in {
        "không", "không có", "none", "null", "nan"
    }


def load_products():
    payload = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data_clean", "Sheet1", "products", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Không tìm thấy danh sách sản phẩm trong {INPUT_FILE}")


def display_value(value):
    if value is True:
        return "có"
    if value is False:
        return "không"
    return str(clean(value))


def limit_tokens(text, max_tokens=MAX_TOKENS):
    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        shortened = encoding.decode(tokens[:max_tokens])
    else:
        # Fallback bảo thủ khi máy chưa cài tiktoken.
        max_characters = max_tokens * 3
        if len(text) <= max_characters:
            return text
        shortened = text[:max_characters]

    last_period = shortened.rfind(".")
    if last_period > 0:
        shortened = shortened[:last_period + 1]
    return shortened.strip()


def create_semantic_text(product, sku):
    brand = clean(product.get("brand")) or "không xác định"
    model = clean(product.get("model_code"))
    introduction = f"{CATEGORY.capitalize()} {brand}"
    if model:
        introduction += f" model {model}"
    introduction += f", mã sản phẩm {sku}."

    sentences = [introduction]
    used = set()

    # Mỗi nhóm dưới đây phản ánh đúng ý định tìm kiếm của loại sản phẩm này.
    for heading, fields in GROUPS:
        facts = []
        for field in fields:
            value = product.get(field)
            signature = (field, str(value))
            if meaningful(value) and signature not in used:
                used.add(signature)
                label = field.replace("_", " ")
                facts.append(f"{label}: {display_value(value)}")
        if facts:
            sentences.append(f"{heading}: " + "; ".join(facts[:8]) + ".")

    origin = clean(product.get("Sản xuất tại"))
    year = clean(product.get("nam_ra_mat") or product.get("nam_san_xuat_chuan"))
    if origin:
        sentences.append(f"Sản xuất tại {origin}.")
    if year:
        sentences.append(f"Năm sản phẩm: {year}.")

    return limit_tokens(" ".join(sentences))


def create_metadata(product, source_index):
    metadata = {
        key: clean(value)
        for key, value in product.items()
        if clean(value) is not None
    }
    metadata["category"] = CATEGORY
    metadata["source_index"] = source_index
    return metadata


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    products = load_products()
    output, seen_ids = [], set()

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
        output.append({
            "id": sku,
            "name": f"{CATEGORY} {brand} {sku}",
            "text": create_semantic_text(product, sku),
            "image_path": IMAGE_PATH,
            "metadata": create_metadata(product, index),
        })

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã xử lý {len(output)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


