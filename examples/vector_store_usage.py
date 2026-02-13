#!/usr/bin/env python3
"""Usage examples for the Milvus Vector Store SDK.

This script demonstrates the main capabilities of :class:`MilvusVectorStore`.
It expects a running Milvus instance with the ``kubeflow_docs`` collection
already populated by the ingestion pipeline.

Run::

    # Basic usage (defaults to localhost:19530)
    python examples/vector_store_usage.py

    # Custom host
    python examples/vector_store_usage.py --host milvus.example.com --port 19530
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from pprint import pprint

import numpy as np

# Make the SDK importable when running from the repo root
SDK_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SDK_ROOT))

from vector_store import (
    EmbeddingConfig,
    MilvusConnectionConfig,
    MilvusVectorStore,
    RetryConfig,
    SearchConfig,
    VectorStoreConfig,
)


# ---------------------------------------------------------------------------
# 1. Basic initialisation
# ---------------------------------------------------------------------------
def example_basic_usage(host: str, port: int, collection: str) -> None:
    """Connect, check stats, and run a text search."""
    print("\n=== 1. Basic Usage ===\n")

    # Simplest constructor – just pass host, port, collection
    store = MilvusVectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    # Check collection info
    stats = store.get_stats()
    print("Collection statistics:")
    pprint(stats)

    # Text-based semantic search (embeds the query automatically)
    results = store.search_by_text("How do I deploy a Kubeflow pipeline?", top_k=3)
    print("\nTop-3 results for 'How do I deploy a Kubeflow pipeline?':")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['text'][:100]}...")
        print(f"     metadata: {r['metadata']}")

    store.close()


# ---------------------------------------------------------------------------
# 2. Using the config object
# ---------------------------------------------------------------------------
def example_custom_config(host: str, port: int, collection: str) -> None:
    """Demonstrate how to override every config knob."""
    print("\n=== 2. Custom Configuration ===\n")

    config = VectorStoreConfig(
        connection=MilvusConnectionConfig(
            host=host,
            port=port,
            timeout=60.0,
        ),
        retry=RetryConfig(max_retries=5, backoff_base=1.5),
        search=SearchConfig(
            metric_type="IP",
            top_k=10,
            search_params={"nprobe": 32},
        ),
        embedding=EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            normalize=True,
        ),
        collection_name=collection,
        batch_size=200,
    )

    store = MilvusVectorStore(
        host=host,
        port=port,
        collection_name=collection,
        config=config,
    )

    print(f"Store: {store!r}")
    print(f"Stats: {store.get_stats()}")
    store.close()


# ---------------------------------------------------------------------------
# 3. Context manager
# ---------------------------------------------------------------------------
def example_context_manager(host: str, port: int, collection: str) -> None:
    """Use as a context manager for automatic cleanup."""
    print("\n=== 3. Context Manager ===\n")

    with MilvusVectorStore(host, port, collection) as store:
        stats = store.get_stats()
        print(f"Entities: {stats['total_entities']}")
    # connection is released automatically here
    print("Connection closed automatically.")


# ---------------------------------------------------------------------------
# 4. Insert, search, delete lifecycle
# ---------------------------------------------------------------------------
def example_insert_search_delete(host: str, port: int, collection: str) -> None:
    """Full lifecycle: insert → search → delete."""
    print("\n=== 4. Insert → Search → Delete ===\n")

    with MilvusVectorStore(host, port, collection) as store:
        dim = store.get_stats()["embedding_dim"]
        rng = np.random.default_rng(42)

        # --- Insert ---
        chunks = [
            {
                "chunk_id": f"example_chunk_{i}",
                "embedding": rng.random(dim, dtype=np.float32).tolist(),
                "text": f"This is example chunk number {i}.",
                "metadata": {"source": "example.py", "index": i},
            }
            for i in range(5)
        ]

        inserted_ids = store.insert(chunks)
        print(f"Inserted {len(inserted_ids)} chunks: {inserted_ids}")

        # --- Search by raw vector ---
        query_vector = rng.random(dim, dtype=np.float32)
        results = store.search(query_vector, top_k=3)
        print("\nSearch results (raw vector):")
        for r in results:
            print(f"  {r['chunk_id']} – score={r['score']:.4f}")

        # --- Search with metadata filter ---
        filtered = store.search(
            query_vector,
            top_k=3,
            filter={"source": "example.py"},
        )
        print(f"\nFiltered results (source=example.py): {len(filtered)} hits")

        # --- Delete ---
        deleted = store.delete(inserted_ids)
        print(f"\nDeleted {deleted} chunks.")


# ---------------------------------------------------------------------------
# 5. Async usage
# ---------------------------------------------------------------------------
async def example_async(host: str, port: int, collection: str) -> None:
    """Demonstrate async wrappers."""
    print("\n=== 5. Async Usage ===\n")

    store = MilvusVectorStore(host, port, collection)
    dim = store.get_stats()["embedding_dim"]
    rng = np.random.default_rng(99)

    # Async insert
    chunks = [
        {
            "chunk_id": f"async_chunk_{i}",
            "embedding": rng.random(dim, dtype=np.float32).tolist(),
            "text": f"Async example chunk {i}.",
            "metadata": {"async": True},
        }
        for i in range(3)
    ]
    ids = await store.ainsert(chunks)
    print(f"Async inserted: {ids}")

    # Async search
    query = rng.random(dim, dtype=np.float32)
    results = await store.asearch(query, top_k=2)
    print(f"Async search returned {len(results)} results")

    # Async delete
    deleted = await store.adelete(ids)
    print(f"Async deleted: {deleted}")

    store.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus Vector Store SDK examples")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument(
        "--collection", default="kubeflow_docs", help="Collection name"
    )
    parser.add_argument(
        "--example",
        choices=["basic", "config", "context", "lifecycle", "async", "all"],
        default="all",
        help="Which example to run",
    )
    args = parser.parse_args()

    examples = {
        "basic": lambda: example_basic_usage(args.host, args.port, args.collection),
        "config": lambda: example_custom_config(args.host, args.port, args.collection),
        "context": lambda: example_context_manager(args.host, args.port, args.collection),
        "lifecycle": lambda: example_insert_search_delete(args.host, args.port, args.collection),
        "async": lambda: asyncio.run(example_async(args.host, args.port, args.collection)),
    }

    if args.example == "all":
        for name, fn in examples.items():
            try:
                fn()
            except Exception as exc:
                print(f"\n[!] Example '{name}' failed: {exc}")
    else:
        examples[args.example]()


if __name__ == "__main__":
    main()
