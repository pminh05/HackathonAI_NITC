"""Explicit, idempotent payload-index setup for the ``tulanh`` collection."""

from __future__ import annotations

import argparse
from typing import Any

from qdrant_client import models

from advisor.categories.refrigerator import load_config
from advisor.retrieval.qdrant import create_qdrant_client
from advisor.schemas import ApplicationSettings


SCHEMA_TYPES = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "bool": models.PayloadSchemaType.BOOL,
}


def _schema_name(value: Any) -> str:
    data_type = getattr(value, "data_type", value)
    raw = getattr(data_type, "value", data_type)
    return str(raw).lower()


def find_missing_indexes(
    client: Any, collection: str, required: dict[str, str]
) -> dict[str, str]:
    """Return absent or mismatched payload index fields."""
    existing = client.get_collection(collection).payload_schema or {}
    return {
        field: schema
        for field, schema in required.items()
        if field not in existing or _schema_name(existing[field]) != schema
    }


def ensure_payload_indexes(
    client: Any,
    collection: str,
    required: dict[str, str],
    *,
    apply: bool = False,
) -> dict[str, str]:
    """Check indexes and optionally create only the missing definitions."""
    missing = find_missing_indexes(client, collection, required)
    if apply:
        for field, schema in missing.items():
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=SCHEMA_TYPES[schema],
                wait=True,
            )
        return find_missing_indexes(client, collection, required)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create missing indexes. Without this flag the command is read-only.",
    )
    args = parser.parse_args()
    settings = ApplicationSettings()
    config = load_config()
    client = create_qdrant_client(settings)
    missing = ensure_payload_indexes(
        client,
        config["collection"],
        config["payload_indexes"],
        apply=args.apply,
    )
    if missing:
        print("Missing or mismatched payload indexes:")
        for field, schema in missing.items():
            print(f"- {field}: {schema}")
        print("Run again with --apply to create them.")
        return 1
    print(f"Collection {config['collection']} has all required payload indexes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
