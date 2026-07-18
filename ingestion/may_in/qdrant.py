"""Upsert normalized printer points into Qdrant Cloud."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


DATASET = "may_in"
COLLECTION_NAME = "mayin"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384
BATCH_SIZE = 64
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"ingestion:{DATASET}")


def load_products() -> list[dict]:
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu máy in đã chuẩn hóa để upload")
    return products


def point_generator(products: list[dict]) -> Iterator[models.PointStruct]:
    """Yield deterministic points and let Qdrant Cloud embed passage text."""
    seen_ids: set[str] = set()
    for index, product in enumerate(products):
        product_id = str(product.get("id") or "").strip()
        if not product_id:
            raise ValueError(f"Sản phẩm index {index} thiếu id")
        if product_id in seen_ids:
            raise ValueError(f"ID bị trùng: {product_id}")
        text = str(product.get("text") or "").strip()
        if not text:
            raise ValueError(f"Sản phẩm index {index} thiếu semantic text")
        seen_ids.add(product_id)
        yield models.PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, product_id)),
            vector=models.Document(text=f"passage: {text}", model=EMBEDDING_MODEL),
            payload={
                "product_id": product_id,
                "name": product.get("name"),
                "text": text,
                "image_path": product.get("image_path"),
                "metadata": product.get("metadata", {}),
            },
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY")
    products = load_products()
    client = QdrantClient(
        url=url, api_key=api_key, cloud_inference=True, timeout=120
    )
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
    else:
        info = client.get_collection(COLLECTION_NAME)
        existing_size = getattr(info.config.params.vectors, "size", None)
        if existing_size is not None and existing_size != VECTOR_SIZE:
            raise ValueError(
                f"Collection dùng vector {existing_size} chiều, yêu cầu {VECTOR_SIZE}."
            )
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=point_generator(products),
        batch_size=BATCH_SIZE,
        parallel=1,
        max_retries=3,
        wait=True,
    )
    print(f"Đã upsert {len(products)} điểm vào collection {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
