import json
import math
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models

load_dotenv()

INPUT_FILE = Path(r"data\tu_lanh_embedded.json")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "tulanh"
BATCH_SIZE = 64
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Namespace cố định để cùng một SKU luôn tạo ra cùng một UUID.
POINT_NAMESPACE = uuid.UUID("7ba04ffd-2fa8-4e32-b0c6-ec25739120d2")


def load_products():
    with INPUT_FILE.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        products = payload
    elif isinstance(payload, dict):
        products = (
            payload.get("products")
            or payload.get("data")
            or payload.get("data_clean")
        )
    else:
        products = None

    if not isinstance(products, list) or not products:
        raise ValueError("Không tìm thấy danh sách sản phẩm trong file JSON.")

    return products


def validate_vector(vector, index, expected_size=None):
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"Sản phẩm tại index {index} thiếu vector hợp lệ.")

    if expected_size is not None and len(vector) != expected_size:
        raise ValueError(
            f"Vector tại index {index} có {len(vector)} chiều; "
            f"yêu cầu {expected_size} chiều."
        )

    cleaned_vector = []

    for position, value in enumerate(vector):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Vector tại index {index}, vị trí {position} không phải số."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"Vector tại index {index}, vị trí {position} chứa NaN hoặc Infinity."
            )

        cleaned_vector.append(value)

    return cleaned_vector


def get_product_id(product, index):
    product_id = product.get("id")

    if product_id is None and isinstance(product.get("metadata"), dict):
        product_id = product["metadata"].get("sku")

    product_id = str(product_id or "").strip()

    if not product_id:
        raise ValueError(f"Sản phẩm tại index {index} thiếu id hoặc metadata.sku.")

    return product_id


def create_payload(product, product_id):
    metadata = product.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "product_id": product_id,
        "name": product.get("name"),
        "text": product.get("text"),
        "metadata": metadata,
    }


def point_generator(products, vector_size):
    seen_ids = set()

    for index, product in enumerate(products):
        if not isinstance(product, dict):
            raise ValueError(f"Sản phẩm tại index {index} không phải object JSON.")

        product_id = get_product_id(product, index)

        if product_id in seen_ids:
            raise ValueError(f"ID sản phẩm bị trùng: {product_id}")

        seen_ids.add(product_id)

        vector = validate_vector(
            vector=product.get("vector"),
            index=index,
            expected_size=vector_size,
        )

        # Qdrant chỉ chấp nhận point ID dạng số nguyên dương hoặc UUID.
        point_id = str(uuid.uuid5(POINT_NAMESPACE, product_id))

        yield models.PointStruct(
            id=point_id,
            vector=vector,
            payload=create_payload(product, product_id),
        )


def main():
    api_key = QDRANT_API_KEY 

    if not api_key:
        raise RuntimeError(
            "Thiếu biến môi trường QDRANT_API_KEY. "
            "Không nên ghi API key trực tiếp trong source code."
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")

    products = load_products()
    first_vector = validate_vector(products[0].get("vector"), index=0)
    vector_size = len(first_vector)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=api_key,
        timeout=120,
    )

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(
            f"Đã tạo collection '{COLLECTION_NAME}' "
            f"với vector {vector_size} chiều."
        )
    else:
        collection_info = client.get_collection(COLLECTION_NAME)
        vectors_config = collection_info.config.params.vectors

        existing_size = getattr(vectors_config, "size", None)

        if existing_size is not None and existing_size != vector_size:
            raise ValueError(
                f"Collection hiện có dùng vector {existing_size} chiều, "
                f"nhưng dữ liệu mới có {vector_size} chiều."
            )

        print(f"Collection '{COLLECTION_NAME}' đã tồn tại; tiếp tục upsert.")

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=point_generator(products, vector_size),
        batch_size=BATCH_SIZE,
        parallel=1,
        max_retries=3,
        wait=True,
    )

    collection_info = client.get_collection(COLLECTION_NAME)

    print(f"Đã upload/upsert {len(products)} sản phẩm.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Số point hiện có: {collection_info.points_count}")


if __name__ == "__main__":
    main()
