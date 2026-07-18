import json
import re
from pathlib import Path

input_file = Path(r"data\tu_lanh_processed_vi.json")
output_file = Path(r"data\tu_lanh_dictionary.json")


FIELDS = {
    "Kiểu dáng chuẩn": {
        "source_fields": ["Kiểu dáng chuẩn", "Kiểu dáng"],
        "description": "Kiểu dáng hoặc thiết kế của tủ lạnh.",
        "split_by_pipe": False,
    },
    "Công nghệ làm lạnh": {
        "source_fields": ["Công nghệ làm lạnh"],
        "description": "Các công nghệ hỗ trợ làm lạnh và phân phối hơi lạnh.",
        "split_by_pipe": True,
    },
    "Chất liệu khay ngăn lạnh": {
        "source_fields": ["Chất liệu khay ngăn lạnh"],
        "description": "Chất liệu của khay chứa trong ngăn lạnh.",
        "split_by_pipe": True,
    },
    "Chất liệu thân vỏ": {
        "source_fields": ["Chất liệu thân vỏ"],
        "description": "Chất liệu hoặc kiểu hoàn thiện bên ngoài của tủ lạnh.",
        "split_by_pipe": True,
    },
    "Công nghệ tiết kiệm điện": {
        "source_fields": ["Công nghệ tiết kiệm điện"],
        "description": "Các công nghệ hỗ trợ giảm điện năng tiêu thụ.",
        "split_by_pipe": True,
    },
    "Công nghệ bảo quản thực phẩm": {
        "source_fields": ["Công nghệ bảo quản thực phẩm"],
        "description": "Các công nghệ hỗ trợ giữ thực phẩm tươi lâu và hạn chế mất độ ẩm.",
        "split_by_pipe": True,
    },
}


def clean(value):
    if value is None:
        return None

    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def extract_products(payload):
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        raise ValueError("Cấu trúc JSON không hợp lệ.")

    for key in ("data_clean", "products", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]

    raise ValueError("Không tìm thấy danh sách sản phẩm trong JSON.")


def get_source_value(product, source_fields):
    metadata = product.get("metadata", {})

    for source_field in source_fields:
        value = clean(metadata.get(source_field))
        if value:
            return value

    return None


def extract_values(products, config):
    possible_values = set()

    for product in products:
        value = get_source_value(
            product=product,
            source_fields=config["source_fields"],
        )

        if not value:
            continue

        if config["split_by_pipe"]:
            values = value.split("|")
        else:
            values = [value]

        for item in values:
            item = clean(item)
            if item:
                possible_values.add(item)

    return sorted(
        possible_values,
        key=lambda item: item.casefold(),
    )


def main():
    with input_file.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    products = extract_products(payload)

    print(f"Tìm thấy {len(products)} sản phẩm")

    fields_dictionary = {}

    for output_field, config in FIELDS.items():
        values = extract_values(
            products=products,
            config=config,
        )

        print(f"{output_field}: {len(values)} giá trị")

        fields_dictionary[output_field] = {
            "description": config["description"],
            "data_type": "string",
            "possible_values": values,
        }

    result = {
        "dataset": "tu_lanh",
        "fields": fields_dictionary,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nĐã tạo file: {output_file}")


if __name__ == "__main__":
    main()