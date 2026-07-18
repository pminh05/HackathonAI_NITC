import json
import re
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    tiktoken = None


input_file = Path(r"data\tu_lanh.json")
output_file = Path(r"data\tu_lanh_processed.json")

MAX_TOKENS = 450


def is_blank(value):
    return value is None or (
        isinstance(value, str) and not value.strip()
    )


def clean(value):
    if is_blank(value):
        return None

    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()

    return value


def split_features(value):
    if is_blank(value):
        return []

    return [
        clean(item)
        for item in str(value).split("|")
        if not is_blank(item)
    ]


def add_sentence(sentences, sentence):
    sentence = clean(sentence)

    if sentence and sentence not in sentences:
        sentences.append(sentence)


def describe_capacity(product):
    capacity = (
        product.get("dung_tich_su_dung_lit")
        or product.get("dung_tich_tong_lit")
    )

    if capacity is None:
        return None

    if capacity < 200:
        return (
            f"Dung tích khoảng {capacity} lít, phù hợp người sống "
            "một mình, sinh viên, phòng trọ hoặc gia đình ít người."
        )

    if capacity < 300:
        return (
            f"Dung tích khoảng {capacity} lít, phù hợp gia đình nhỏ "
            "từ 2 đến 3 người."
        )

    if capacity < 400:
        return (
            f"Dung tích khoảng {capacity} lít, phù hợp gia đình "
            "từ 3 đến 4 người."
        )

    if capacity < 550:
        return (
            f"Dung tích lớn khoảng {capacity} lít, phù hợp gia đình "
            "từ 4 đến 5 người và nhu cầu lưu trữ nhiều thực phẩm."
        )

    return (
        f"Dung tích rất lớn khoảng {capacity} lít, phù hợp gia đình "
        "đông người, thường xuyên dự trữ hoặc bảo quản nhiều thực phẩm."
    )


def describe_users(product):
    minimum = product.get("so_nguoi_min")
    maximum = product.get("so_nguoi_max")

    if minimum is not None and maximum is not None:
        return (
            f"Sản phẩm phù hợp gia đình khoảng {minimum} đến "
            f"{maximum} người."
        )

    raw_value = product.get("Số người sử dụng")

    if not is_blank(raw_value):
        return f"Phù hợp nhu cầu sử dụng của {clean(raw_value)}."

    return None


def describe_design(product):
    design = (
        product.get("kieu_dang_chuan")
        or product.get("Kiểu dáng")
    )

    if is_blank(design):
        return None

    design = clean(design)
    design_lower = design.lower()

    if "ngăn đá dưới" in design_lower:
        return (
            "Thiết kế ngăn đá dưới giúp ngăn mát nằm ngang tầm sử dụng, "
            "thuận tiện lấy thực phẩm dùng hằng ngày."
        )

    if "ngăn đá trên" in design_lower:
        return (
            "Thiết kế ngăn đá trên truyền thống, dễ sử dụng và phù hợp "
            "nhiều không gian bếp gia đình."
        )

    if "side by side" in design_lower:
        return (
            "Thiết kế Side by Side hiện đại, không gian lưu trữ rộng, "
            "phù hợp gia đình đông người và căn bếp lớn."
        )

    if "multi door" in design_lower or "nhiều cửa" in design_lower:
        return (
            "Thiết kế nhiều cửa Multi Door giúp phân chia thực phẩm "
            "khoa học và hạn chế thất thoát hơi lạnh khi mở tủ."
        )

    if "mini" in design_lower:
        return (
            "Thiết kế tủ lạnh mini nhỏ gọn, phù hợp phòng ngủ, "
            "phòng trọ, văn phòng hoặc nhu cầu cá nhân."
        )

    return f"Tủ lạnh có kiểu dáng {design}."


def describe_storage(product):
    details = []

    fridge_capacity = product.get("dung_tich_ngan_lanh_lit")
    freezer_capacity = product.get("dung_tich_ngan_da_lit")
    convertible_capacity = product.get(
        "dung_tich_ngan_chuyen_doi_lit"
    )

    if fridge_capacity is not None:
        details.append(f"ngăn lạnh {fridge_capacity} lít")

    if freezer_capacity is not None:
        details.append(f"ngăn đá {freezer_capacity} lít")

    if convertible_capacity is not None:
        details.append(
            f"ngăn chuyển đổi {convertible_capacity} lít"
        )

    if not details:
        return None

    return (
        "Không gian bảo quản gồm "
        + ", ".join(details)
        + ", hỗ trợ phân chia thực phẩm theo từng nhu cầu sử dụng."
    )


def describe_energy(product):
    sentences = []

    if product.get("co_inverter") is True:
        sentences.append(
            "Có công nghệ Inverter giúp vận hành ổn định, "
            "tiết kiệm điện và giảm tiếng ồn."
        )

    energy_technology = product.get(
        "Công nghệ tiết kiệm điện"
    )

    if not is_blank(energy_technology):
        sentences.append(
            "Công nghệ tiết kiệm điện gồm "
            f"{clean(energy_technology)}."
        )

    annual_energy = product.get("dien_nang_kwh_nam")

    if annual_energy is not None:
        sentences.append(
            "Mức tiêu thụ điện tham khảo khoảng "
            f"{annual_energy} kWh mỗi năm."
        )

    return sentences


def describe_cooling(product):
    sentences = []

    cooling = product.get("Công nghệ làm lạnh")

    if not is_blank(cooling):
        sentences.append(
            f"Công nghệ làm lạnh {clean(cooling)} hỗ trợ phân phối "
            "hơi lạnh và duy trì nhiệt độ ổn định."
        )

    preservation = product.get(
        "Công nghệ bảo quản thực phẩm"
    )

    if not is_blank(preservation):
        sentences.append(
            f"Công nghệ bảo quản {clean(preservation)} hỗ trợ giữ "
            "thực phẩm tươi lâu và hạn chế mất độ ẩm."
        )

    return sentences


def describe_conveniences(product):
    sentences = []

    if product.get("co_lay_nuoc_ngoai") is True:
        sentences.append(
            "Có lấy nước bên ngoài, thuận tiện uống nước "
            "mà không cần mở cửa tủ."
        )

    if product.get("co_che_do_tu_dong") is True:
        sentences.append(
            "Có chế độ tự động giúp quá trình sử dụng thuận tiện hơn."
        )

    features = split_features(product.get("Tiện ích"))

    # Giới hạn tiện ích để tránh embedding bị nhiễu.
    if features:
        sentences.append(
            "Các tiện ích nổi bật gồm "
            + ", ".join(features[:6])
            + "."
        )

    return sentences


def describe_material(product):
    sentences = []

    tray_material = product.get(
        "Chất liệu khay ngăn lạnh"
    )

    if not is_blank(tray_material):
        sentences.append(
            f"Khay ngăn lạnh làm từ {clean(tray_material)}, "
            "phù hợp lưu trữ thực phẩm hằng ngày."
        )

    body_material = product.get("Chất liệu thân vỏ")

    if not is_blank(body_material):
        sentences.append(
            f"Thân vỏ sử dụng chất liệu {clean(body_material)}."
        )

    return sentences


def describe_dimensions(product):
    dimensions = []

    mappings = [
        ("cao", "cao_cm"),
        ("ngang", "ngang_cm"),
        ("sâu", "sau_cm"),
    ]

    for label, key in mappings:
        value = product.get(key)

        if value is not None:
            dimensions.append(f"{label} {value} cm")

    if not dimensions:
        return None

    text = "Kích thước tủ: " + ", ".join(dimensions)

    weight = product.get("khoi_luong_may_kg")

    if weight is not None:
        text += f", khối lượng khoảng {weight} kg"

    return text + "."


def create_text(product, name):
    sentences = []

    brand = clean(product.get("brand")) or "không xác định"
    model = clean(product.get("model_code"))

    design = (
        product.get("kieu_dang_chuan")
        or product.get("Kiểu dáng")
    )

    introduction = (
        f"{name} là sản phẩm tủ lạnh của hãng {brand}"
    )

    if model:
        introduction += f", model {model}"

    if not is_blank(design):
        introduction += f", kiểu {clean(design)}"

    introduction += "."

    add_sentence(sentences, introduction)
    add_sentence(sentences, describe_capacity(product))
    add_sentence(sentences, describe_users(product))
    add_sentence(sentences, describe_design(product))
    add_sentence(sentences, describe_storage(product))

    for sentence in describe_cooling(product):
        add_sentence(sentences, sentence)

    for sentence in describe_energy(product):
        add_sentence(sentences, sentence)

    for sentence in describe_conveniences(product):
        add_sentence(sentences, sentence)

    for sentence in describe_material(product):
        add_sentence(sentences, sentence)

    add_sentence(sentences, describe_dimensions(product))

    origin = product.get("Sản xuất tại")
    year = product.get("nam_ra_mat")

    if not is_blank(origin) and year is not None:
        add_sentence(
            sentences,
            f"Sản phẩm được sản xuất tại {clean(origin)}, "
            f"ra mắt năm {year}."
        )
    elif not is_blank(origin):
        add_sentence(
            sentences,
            f"Sản phẩm được sản xuất tại {clean(origin)}."
        )
    elif year is not None:
        add_sentence(
            sentences,
            f"Sản phẩm ra mắt năm {year}."
        )

    return " ".join(sentences)


def limit_tokens(text, max_tokens=MAX_TOKENS):
    if tiktoken is None:
        # Ước lượng an toàn khi chưa cài tiktoken.
        # Tiếng Việt thường có thể tốn nhiều token hơn tiếng Anh.
        max_characters = max_tokens * 3
        if len(text) <= max_characters:
            return text

        shortened = text[:max_characters]
        last_period = shortened.rfind(".")

        if last_period > 0:
            shortened = shortened[:last_period + 1]

        return shortened.strip()

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    shortened = encoding.decode(tokens[:max_tokens])
    last_period = shortened.rfind(".")

    if last_period > 0:
        shortened = shortened[:last_period + 1]

    return shortened.strip()


def create_metadata(product, source_index):
    metadata = {}

    for key, value in product.items():
        value = clean(value)

        if value is not None:
            metadata[key] = value

    metadata["category"] = "tủ lạnh"
    metadata["source_index"] = source_index

    return metadata


def extract_products(payload):
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        raise ValueError(
            "Cấu trúc JSON không hợp lệ."
        )

    possible_keys = [
        "data_clean",
        "products",
        "data",
    ]

    for key in possible_keys:
        value = payload.get(key)

        if isinstance(value, list):
            return value

    raise ValueError(
        "Không tìm thấy danh sách sản phẩm trong JSON."
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(file)

    source_products = extract_products(payload)

    output_products = []
    seen_ids = set()

    for index, product in enumerate(source_products):
        if not isinstance(product, dict):
            raise ValueError(
                f"Sản phẩm tại index {index} không phải object."
            )

        product_id = str(
            product.get("sku") or ""
        ).strip()

        brand = (
            clean(product.get("brand"))
            or "không xác định"
        )

        if not product_id:
            raise ValueError(
                f"Sản phẩm tại index {index} thiếu sku/id."
            )

        if product_id in seen_ids:
            raise ValueError(
                f"sku/id bị trùng: {product_id}"
            )

        seen_ids.add(product_id)

        name = f"tủ lạnh {brand} {product_id}"

        semantic_text = create_text(
            product=product,
            name=name,
        )

        semantic_text = limit_tokens(
            semantic_text,
            max_tokens=MAX_TOKENS,
        )

        output_products.append(
            {
                "id": product_id,
                "name": name,
                "text": semantic_text,
                "metadata": create_metadata(
                    product=product,
                    source_index=index,
                ),
            }
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_products,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Đã xử lý {len(output_products)} sản phẩm."
    )
    print(
        f"Số ID duy nhất: {len(seen_ids)}."
    )
    print(
        f"File đầu ra: {output_file}"
    )


if __name__ == "__main__":
    main()