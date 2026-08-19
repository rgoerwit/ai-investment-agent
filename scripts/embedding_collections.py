#!/usr/bin/env python3
"""Inspect embedding collections or initialize the configured empty target.

This command never deletes or re-embeds data. Rebuilding historical vectors must
be a separate, explicit workflow with a reviewed source corpus.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import config
from src.embeddings import (
    fingerprinted_collection_name,
    resolve_embedding_binding,
)


def inspect_collections(client: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in client.list_collections():
        collection = client.get_collection(item) if isinstance(item, str) else item
        rows.append(
            {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata or {},
            }
        )
    return sorted(rows, key=lambda row: str(row["name"]))


def initialize_current(client: Any, base_name: str) -> str:
    binding = resolve_embedding_binding(config)
    name = fingerprinted_collection_name(base_name, binding)
    client.get_or_create_collection(
        name=name,
        metadata={
            **binding.metadata(),
            "legacy_base_name": base_name,
            "version": "3.0",
        },
    )
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-current", metavar="BASE_NAME")
    parser.add_argument("--persist-dir", type=Path)
    args = parser.parse_args()

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(args.persist_dir or config.chroma_persist_directory),
        settings=Settings(anonymized_telemetry=False),
    )
    initialized = (
        initialize_current(client, args.initialize_current)
        if args.initialize_current
        else None
    )
    print(
        json.dumps(
            {
                "initialized": initialized,
                "collections": inspect_collections(client),
                "destructive_actions": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
