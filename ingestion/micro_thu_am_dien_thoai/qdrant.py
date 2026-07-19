"""Upload normalized recording-microphone points with Qdrant Cloud Inference."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


DATASET = "micro_thu_am_dien_thoai"
COLLECTION_NAME = "microthuamdienthoai"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
BATCH_SIZE = 64
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"ingestion:{DATASET}")

CANONICAL_TYPES: dict[str, type | tuple[type, ...]] = {
    "category_scope": str,
    "brand": str,
    "brand_key": str,
    "model_code": str,
    "product_type": str,
    "original_price_vnd": int,
    "promotional_price_vnd": int,
    "compatibility_tags": list,
    "connector_tags": list,
    "feature_tags": list,
    "pickup_pattern": str,
    "wireless_band": str,
    "transmitter_count": int,
    "receiver_count": int,
    "runtime_min_hours": (int, float),
    "runtime_max_hours": (int, float),
    "transmission_range_m": (int, float),
    "audio_frequency_min_hz": int,
    "audio_frequency_max_hz": int,
    "max_spl_db": int,
    "total_weight_g": (int, float),
    "manufacture_year": int,
    "origin": str,
    "data_quality_flags": list,
}


def validate_metadata(metadata: Any, index: int) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError(f"Sản phẩm index {index} thiếu metadata")
    for required in ("category_scope", "brand", "brand_key"):
        if not metadata.get(required):
            raise ValueError(f"Sản phẩm index {index} thiếu metadata.{required}")
    if metadata["category_scope"] != "phone_recording_microphone":
        raise ValueError(f"Sản phẩm index {index} sai category_scope")
    for field, expected in CANONICAL_TYPES.items():
        value = metadata.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ValueError(f"metadata.{field} sai kiểu tại index {index}")
        if isinstance(value, list) and not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"metadata.{field} phải là danh sách chuỗi tại index {index}")
    return metadata


def load_products() -> list[dict[str, Any]]:
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu micro thu âm đã chuẩn hóa để upload")
    return products


def point_generator(
    products: list[dict[str, Any]],
) -> Iterator[models.PointStruct]:
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
            vector=models.Document(
                text=f"passage: {text}",
                model=EMBEDDING_MODEL,
            ),
            payload={
                "product_id": product_id,
                "name": product.get("name"),
                "text": text,
                "image_path": product.get("image_path"),
                "metadata": validate_metadata(product.get("metadata"), index),
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
        url=url,
        api_key=api_key,
        cloud_inference=True,
        timeout=120,
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
