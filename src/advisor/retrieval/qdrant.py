"""Qdrant client and refrigerator semantic-search helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from advisor.schemas import ApplicationSettings, ProductCandidate

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, models


class AdvisorConfigurationError(RuntimeError):
    """Raised when an external service is configured incompatibly."""


def _schema_name(value: Any) -> str:
    data_type = getattr(value, "data_type", value)
    raw = getattr(data_type, "value", data_type)
    return str(raw).lower()


def find_missing_indexes(
    client: Any, collection: str, required: dict[str, str]
) -> dict[str, str]:
    """Return absent or mismatched payload index fields for any category."""
    existing = client.get_collection(collection).payload_schema or {}
    return {
        field: schema
        for field, schema in required.items()
        if field not in existing or _schema_name(existing[field]) != schema
    }


def create_qdrant_client(settings: ApplicationSettings) -> QdrantClient:
    """Create a Cloud-Inference-enabled Qdrant client lazily."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before creating Qdrant.") from exc

    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise AdvisorConfigurationError("QDRANT_URL and QDRANT_API_KEY are required")
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        cloud_inference=True,
        timeout=settings.qdrant_timeout_seconds,
    )


def query_products(
    client: QdrantClient,
    *,
    collection: str,
    embedding_model: str,
    query_text: str,
    query_filter: models.Filter | None,
    limit: int,
    timeout: int,
) -> list[Any]:
    """Run filtered dense retrieval using the collection's embedding model."""
    from qdrant_client import models

    result = client.query_points(
        collection_name=collection,
        query=models.Document(text=query_text, model=embedding_model),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        timeout=timeout,
    )
    return list(result.points)


def normalize_candidate(point: Any) -> dict[str, Any]:
    """Whitelist useful catalog fields and omit noisy promotion metadata."""
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = metadata.get("Giá khuyến mãi vnd")
    original_price = metadata.get("Giá gốc vnd")
    candidate = ProductCandidate(
        product_id=str(payload.get("product_id") or point.id),
        name=str(payload.get("name") or "Sản phẩm chưa có tên"),
        qdrant_score=float(point.score),
        brand=metadata.get("brand"),
        style=metadata.get("Kiểu dáng chuẩn"),
        effective_price_vnd=promotional_price or original_price,
        original_price_vnd=original_price,
        promotional_price_vnd=promotional_price,
        capacity_lit=metadata.get("Dung tích sử dụng lít"),
        suitable_for=metadata.get("Số người sử dụng"),
        description=payload.get("text"),
        image_path=payload.get("image_path") or metadata.get("image_path"),
    ).model_dump(mode="json")
    candidate.update(
        {
            "freezer_capacity_lit": metadata.get("Dung tích ngăn đá lít"),
            "annual_energy_kwh": metadata.get("Điện năng kWh năm"),
            "inverter": metadata.get("Có inverter"),
            "external_water": metadata.get("Có lấy nước ngoài"),
            "automatic_mode": metadata.get("Có chế độ tự động"),
            "dimensions_cm": {
                "width": metadata.get("Ngang cm"),
                "height": metadata.get("Cao cm"),
                "depth": metadata.get("Sâu cm"),
            },
            "utilities": metadata.get("Tiện ích"),
            "preservation": metadata.get("Công nghệ bảo quản thực phẩm"),
            "energy_saving_technology": metadata.get("Công nghệ tiết kiệm điện"),
        }
    )
    return candidate
