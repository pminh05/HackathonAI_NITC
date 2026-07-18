"""Upload deterministic cooler/freezer points to Qdrant."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


DATASET = "tu_mat_tu_dong"
COLLECTION_NAME = "tumattudong"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_embedded.json"
PROCESSED_INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384
BATCH_SIZE = 64
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"ingestion:{DATASET}")


def validate_vector(
    vector: Any, index: int, expected_size: int = VECTOR_SIZE
) -> list[float]:
    if not isinstance(vector, list) or len(vector) != expected_size:
        raise ValueError(
            f"Vector không hợp lệ tại index {index}; yêu cầu {expected_size} chiều"
        )
    cleaned: list[float] = []
    for position, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Vector index {index}, vị trí {position} không phải số")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Vector index {index} chứa NaN/Infinity")
        cleaned.append(value)
    return cleaned


def load_products(path: Path = INPUT_FILE) -> list[dict[str, Any]]:
    products = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(products, list) or not products:
        raise ValueError(f"Không có dữ liệu tủ mát, tủ đông trong {path}")
    return products


def create_payload(product: dict[str, Any], index: int) -> dict[str, Any]:
    product_id = str(product.get("id") or "").strip()
    name = str(product.get("name") or "").strip()
    text = str(product.get("text") or "").strip()
    metadata = product.get("metadata")
    if not product_id:
        raise ValueError(f"Sản phẩm index {index} thiếu id")
    if not name or not text:
        raise ValueError(f"Sản phẩm {product_id} thiếu name hoặc semantic text")
    if not isinstance(metadata, dict):
        raise ValueError(f"Sản phẩm {product_id} thiếu metadata object")
    if metadata.get("category_scope") != "cooler_freezer":
        raise ValueError(f"Sản phẩm {product_id} sai category_scope")
    if metadata.get("product_family") not in {"cooler", "freezer"}:
        raise ValueError(f"Sản phẩm {product_id} sai product_family")
    return {
        "product_id": product_id,
        "name": name,
        "text": text,
        "image_path": product.get("image_path"),
        "metadata": metadata,
    }


def point_generator(
    products: list[dict[str, Any]], expected_size: int = VECTOR_SIZE
) -> Iterator[models.PointStruct]:
    seen_ids: set[str] = set()
    for index, product in enumerate(products):
        payload = create_payload(product, index)
        product_id = payload["product_id"]
        if product_id in seen_ids:
            raise ValueError(f"ID bị trùng: {product_id}")
        seen_ids.add(product_id)

        yield models.PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, product_id)),
            vector=validate_vector(product.get("vector"), index, expected_size),
            payload=payload,
        )


def payload_refresh_generator(
    products: list[dict[str, Any]], existing_points: list[Any]
) -> Iterator[models.PointStruct]:
    """Reuse existing vectors while replacing legacy payloads by product ID."""
    existing_by_product_id: dict[str, Any] = {}
    for point in existing_points:
        product_id = str((point.payload or {}).get("product_id") or "").strip()
        if not product_id:
            raise ValueError(f"Point Qdrant {point.id} thiếu payload.product_id")
        if product_id in existing_by_product_id:
            raise ValueError(f"Qdrant có product_id bị trùng: {product_id}")
        existing_by_product_id[product_id] = point

    seen_ids: set[str] = set()
    for index, product in enumerate(products):
        payload = create_payload(product, index)
        product_id = payload["product_id"]
        if product_id in seen_ids:
            raise ValueError(f"ID dữ liệu chuẩn hóa bị trùng: {product_id}")
        seen_ids.add(product_id)
        existing = existing_by_product_id.get(product_id)
        if existing is None:
            raise ValueError(
                f"Không tìm thấy vector hiện có cho product_id {product_id}"
            )
        yield models.PointStruct(
            id=existing.id,
            vector=validate_vector(existing.vector, index),
            payload=payload,
        )


def load_existing_points(client: QdrantClient) -> list[Any]:
    points: list[Any] = []
    offset: Any = None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(page)
        if offset is None:
            return points


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-existing-vectors",
        action="store_true",
        help=(
            "Refresh canonical payloads from the processed file while preserving "
            "vectors already stored in Qdrant."
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY")

    client = QdrantClient(url=url, api_key=api_key, timeout=120)
    if args.reuse_existing_vectors:
        if not client.collection_exists(COLLECTION_NAME):
            raise ValueError(
                f"Collection {COLLECTION_NAME} chưa tồn tại để tái sử dụng vector"
            )
        products = load_products(PROCESSED_INPUT_FILE)
        existing_points = load_existing_points(client)
        if len(existing_points) != len(products):
            raise ValueError(
                f"Collection có {len(existing_points)} point nhưng dữ liệu chuẩn hóa "
                f"có {len(products)} sản phẩm"
            )
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=payload_refresh_generator(products, existing_points),
            batch_size=BATCH_SIZE,
            parallel=1,
            max_retries=3,
            wait=True,
        )
        print(
            f"Đã cập nhật payload cho {len(products)} point trong {COLLECTION_NAME} "
            "và giữ nguyên vector hiện có"
        )
        return

    products = load_products()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
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
    info = client.get_collection(COLLECTION_NAME)
    print(f"Đã upsert {len(products)} điểm vào collection {COLLECTION_NAME}")
    print(f"Số point hiện có: {info.points_count}")


if __name__ == "__main__":
    main()
