"""Upsert normalized smartwatch points into Qdrant Cloud."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


DATASET = "dong_ho_thong_minh"
COLLECTION_NAME = "donghothongminh"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384
BATCH_SIZE = 64
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"ingestion:{DATASET}")
REQUIRED_METADATA_FIELDS = {
    "category_scope",
    "brand_key",
    "compatible_platforms",
    "health_feature_tags",
}
INTEGER_METADATA_FIELDS = {
    "original_price_vnd",
    "promotional_price_vnd",
    "battery_mah",
}
NUMBER_METADATA_FIELDS = {
    "screen_size_inch",
    "case_length_mm",
    "case_width_mm",
    "case_thickness_mm",
    "weight_g",
    "wrist_min_cm",
    "wrist_max_cm",
    "charging_time_hours",
    "typical_battery_hours",
    "water_resistance_atm",
}
BOOLEAN_METADATA_FIELDS = {
    "has_cellular",
    "has_gps",
    "has_notifications",
    "swim_ready",
    "has_sos",
}


def validate_metadata(metadata: Any, index: int) -> dict[str, Any]:
    """Reject stale or incorrectly typed payloads before they reach Qdrant."""
    if not isinstance(metadata, dict):
        raise ValueError(f"Sản phẩm index {index} thiếu metadata")
    missing = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing:
        raise ValueError(
            f"Sản phẩm index {index} thiếu metadata canonical: {', '.join(missing)}. "
            "Hãy chạy lại processing.py và changeName.py trước qdrant.py."
        )
    if metadata["category_scope"] != "smartwatch":
        raise ValueError(
            f"Sản phẩm index {index} có category_scope không hợp lệ: "
            f"{metadata['category_scope']!r}"
        )
    if not isinstance(metadata["brand_key"], str) or not metadata["brand_key"].strip():
        raise ValueError(f"Sản phẩm index {index} có brand_key không hợp lệ")
    for field in ("compatible_platforms", "health_feature_tags"):
        value = metadata[field]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Sản phẩm index {index} có {field} không phải list[str]")
    call_mode = metadata.get("call_mode")
    if call_mode is not None and call_mode not in {"none", "on_wrist", "standalone"}:
        raise ValueError(f"Sản phẩm index {index} có call_mode không hợp lệ")
    for field in INTEGER_METADATA_FIELDS:
        value = metadata.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"Sản phẩm index {index} có {field} không phải integer")
    for field in NUMBER_METADATA_FIELDS:
        value = metadata.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError(f"Sản phẩm index {index} có {field} không phải number")
    for field in BOOLEAN_METADATA_FIELDS:
        value = metadata.get(field)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"Sản phẩm index {index} có {field} không phải bool")
    return metadata


def load_products() -> list[dict[str, Any]]:
    products = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError("Không có dữ liệu đồng hồ thông minh đã chuẩn hóa để upload")
    return products


def point_generator(products: list[dict[str, Any]]) -> Iterator[models.PointStruct]:
    """Yield deterministic canonical points and let Qdrant embed passage text."""
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
        metadata = validate_metadata(product.get("metadata"), index)
        seen_ids.add(product_id)
        yield models.PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, product_id)),
            vector=models.Document(text=f"passage: {text}", model=EMBEDDING_MODEL),
            payload={
                "product_id": product_id,
                "name": product.get("name"),
                "text": text,
                "image_path": product.get("image_path"),
                "metadata": metadata,
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
        vectors = info.config.params.vectors
        existing_size = getattr(vectors, "size", None)
        existing_distance = getattr(vectors, "distance", None)
        if existing_size is not None and existing_size != VECTOR_SIZE:
            raise ValueError(
                f"Collection dùng vector {existing_size} chiều, yêu cầu {VECTOR_SIZE}."
            )
        if existing_distance is not None and str(existing_distance).casefold() not in {
            "cosine",
            "distance.cosine",
        }:
            raise ValueError("Collection hiện có không dùng khoảng cách cosine.")
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
