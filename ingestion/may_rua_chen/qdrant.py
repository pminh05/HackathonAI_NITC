import json
import math
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

DATASET = "may_rua_chen"
COLLECTION_NAME = "mayruachen"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_embedded.json"
BATCH_SIZE = 64
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"ingestion:{DATASET}")


def validate_vector(vector, index, expected_size):
    if not isinstance(vector, list) or len(vector) != expected_size:
        raise ValueError(f"Vector không hợp lệ tại index {index}")
    cleaned = []
    for position, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Vector index {index}, vị trí {position} không phải số")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Vector index {index} chứa NaN/Infinity")
        cleaned.append(value)
    return cleaned


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY")

    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu để upload")

    first_vector = products[0].get("vector")
    if not isinstance(first_vector, list) or not first_vector:
        raise ValueError("Dữ liệu chưa được embedding")
    vector_size = len(first_vector)

    client = QdrantClient(url=url, api_key=api_key, timeout=120)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    points, seen_ids = [], set()
    for index, product in enumerate(products):
        product_id = str(product.get("id") or "").strip()
        if not product_id:
            raise ValueError(f"Sản phẩm index {index} thiếu id")
        if product_id in seen_ids:
            raise ValueError(f"ID bị trùng: {product_id}")
        seen_ids.add(product_id)

        points.append(models.PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, product_id)),
            vector=validate_vector(product.get("vector"), index, vector_size),
            payload={
                "product_id": product_id,
                "name": product.get("name"),
                "text": product.get("text"),
                "image_path": product.get("image_path"),
                "metadata": product.get("metadata", {}),
            },
        ))

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=BATCH_SIZE,
        parallel=1,
        max_retries=3,
        wait=True,
    )
    print(f"Đã upsert {len(points)} điểm vào collection {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
