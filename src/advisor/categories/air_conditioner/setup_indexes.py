"""Explicit, idempotent payload-index setup for the ``maylanh`` collection."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from qdrant_client import models

from advisor.categories.air_conditioner import load_config
from advisor.retrieval.qdrant import create_qdrant_client, find_missing_indexes
from advisor.schemas import ApplicationSettings


SCHEMA_TYPES = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "float": models.PayloadSchemaType.FLOAT,
    "bool": models.PayloadSchemaType.BOOL,
}


def ensure_payload_indexes(
    client: Any,
    collection: str,
    required: dict[str, str],
    *,
    apply: bool = False,
) -> dict[str, str]:
    """Check indexes and optionally create only absent/mismatched definitions."""
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create missing indexes. Without this flag the command is read-only.",
    )
    args = parser.parse_args()
    config = load_config()
    client = create_qdrant_client(ApplicationSettings())
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
