"""Milvus Indexer – KFP Component.

Loads embeddings produced by the embedding-generator component and inserts
them into a Milvus vector database collection.  Supports connection pooling,
retry logic, batch insertion, idempotent upserts, and automatic index
rebuilding.

Outputs:
  • metrics.json – insertion statistics and timing information
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("milvus-indexer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MILVUS_HOST = "milvus-standalone.kubeflow.svc.cluster.local"
DEFAULT_MILVUS_PORT = 19530
DEFAULT_COLLECTION_NAME = "kubeflow_docs"
DEFAULT_BATCH_SIZE = 100
DEFAULT_OUTPUT_DIR = "/tmp/outputs"
DEFAULT_EMBEDDING_DIM = 384

METRICS_FILENAME = "metrics.json"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds

# Index parameters (IVF_FLAT is a good general-purpose index)
INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

# Schema field limits
MAX_TEXT_LENGTH = 65_535
MAX_CHUNK_ID_LENGTH = 512
MAX_METADATA_LENGTH = 65_535

CONNECTION_ALIAS = "milvus_indexer"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class IndexerMetrics:
    """Aggregate metrics for the indexing run."""

    total_vectors_inserted: int = 0
    total_duplicates_skipped: int = 0
    insert_rate_vectors_per_sec: float = 0.0
    total_insert_time_seconds: float = 0.0
    index_build_time_seconds: float = 0.0
    collection_total_entities: int = 0
    flush_time_seconds: float = 0.0
    compact_time_seconds: float = 0.0
    embedding_dimension: int = 0
    batch_size: int = 0
    num_batches: int = 0
    rebuild_index: bool = False


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def connect_to_milvus(
    host: str = DEFAULT_MILVUS_HOST,
    port: int = DEFAULT_MILVUS_PORT,
    alias: str = CONNECTION_ALIAS,
    timeout: float = 30.0,
) -> None:
    """Establish a connection to the Milvus server with retry logic.

    Raises ``ConnectionError`` if the server is unreachable after all retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            connections.connect(
                alias=alias,
                host=host,
                port=str(port),
                timeout=timeout,
            )
            logger.info(
                "Connected to Milvus at %s:%s (alias=%s)", host, port, alias
            )
            return
        except MilvusException as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Milvus connection attempt %d/%d failed: %s – retrying in %ds",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise ConnectionError(
                    f"Failed to connect to Milvus at {host}:{port} "
                    f"after {MAX_RETRIES} attempts: {exc}"
                ) from exc


def disconnect_from_milvus(alias: str = CONNECTION_ALIAS) -> None:
    """Gracefully close the Milvus connection."""
    try:
        connections.disconnect(alias=alias)
        logger.info("Disconnected from Milvus (alias=%s)", alias)
    except Exception:  # noqa: BLE001
        logger.debug("Disconnect warning (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------
def _build_schema(embedding_dim: int) -> CollectionSchema:
    """Return the Milvus collection schema for kubeflow_docs."""
    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=MAX_CHUNK_ID_LENGTH,
            description="Unique chunk identifier",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
            description="Dense embedding vector",
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=MAX_TEXT_LENGTH,
            description="Original chunk text",
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.VARCHAR,
            max_length=MAX_METADATA_LENGTH,
            description="JSON-encoded chunk metadata",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="Kubeflow documentation embeddings for RAG",
    )


def get_or_create_collection(
    name: str,
    embedding_dim: int,
    alias: str = CONNECTION_ALIAS,
) -> Collection:
    """Return an existing collection or create a new one.

    If the collection exists, its embedding dimension is validated against
    *embedding_dim*.

    Raises ``ValueError`` if the existing schema has a mismatched dimension.
    """
    if utility.has_collection(name, using=alias):
        collection = Collection(name=name, using=alias)
        # Validate embedding dimension
        for f in collection.schema.fields:
            if f.name == "embedding":
                existing_dim = f.params.get("dim", None)
                if existing_dim and existing_dim != embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: collection '{name}' "
                        f"has dim={existing_dim}, but input has dim={embedding_dim}."
                    )
                break
        logger.info(
            "Using existing collection '%s' (entities: %d)",
            name,
            collection.num_entities,
        )
        return collection

    schema = _build_schema(embedding_dim)
    collection = Collection(name=name, schema=schema, using=alias)
    logger.info(
        "Created new collection '%s' (dim=%d)", name, embedding_dim
    )
    return collection


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load embeddings from a ``.npy`` file.

    Returns a 2-D float32 NumPy array of shape ``(N, D)``.
    """
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(str(path))
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2-D embedding array, got shape {embeddings.shape}"
        )
    embeddings = embeddings.astype(np.float32)
    logger.info(
        "Loaded embeddings: shape=%s, dtype=%s", embeddings.shape, embeddings.dtype
    )
    return embeddings


def load_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    """Load per-chunk metadata from a JSON file.

    Accepts either a bare JSON list or ``{"metadata": [...]}``.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        meta_list = data
    elif isinstance(data, dict) and "metadata" in data:
        meta_list = data["metadata"]
    else:
        raise ValueError(
            f"Unexpected metadata format: expected list or dict with 'metadata' key, "
            f"got {type(data).__name__}."
        )

    logger.info("Loaded %d metadata records from %s", len(meta_list), metadata_path)
    return meta_list


def validate_inputs(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
) -> None:
    """Validate that embeddings and metadata are consistent."""
    n_emb = embeddings.shape[0]
    n_meta = len(metadata)
    if n_emb != n_meta:
        raise ValueError(
            f"Embeddings/metadata count mismatch: {n_emb} embeddings vs {n_meta} metadata records."
        )
    if n_emb == 0:
        raise ValueError("No embeddings to insert (empty input).")


# ---------------------------------------------------------------------------
# Existing ID lookup (for idempotent inserts)
# ---------------------------------------------------------------------------
def fetch_existing_ids(
    collection: Collection,
    chunk_ids: List[str],
    alias: str = CONNECTION_ALIAS,
) -> Set[str]:
    """Return the set of *chunk_ids* that already exist in the collection.

    Loads the collection into memory for querying, then releases.
    """
    if collection.num_entities == 0:
        return set()

    try:
        collection.load()
        existing: Set[str] = set()
        # Query in batches to avoid exceeding expression size limits
        batch_size = 500
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            expr_values = ", ".join(f'"{cid}"' for cid in batch)
            expr = f"chunk_id in [{expr_values}]"
            results = collection.query(expr=expr, output_fields=["chunk_id"])
            existing.update(r["chunk_id"] for r in results)
        return existing
    except MilvusException as exc:
        logger.warning(
            "Could not query existing IDs (will proceed without dedup): %s", exc
        )
        return set()
    finally:
        try:
            collection.release()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------
def prepare_batch(
    chunk_ids: List[str],
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    start: int,
    end: int,
) -> List[List[Any]]:
    """Prepare a single batch of data in Milvus column-oriented format.

    Returns ``[chunk_id_list, embedding_list, text_list, metadata_json_list]``.
    """
    batch_ids: List[str] = []
    batch_embeddings: List[List[float]] = []
    batch_texts: List[str] = []
    batch_meta: List[str] = []

    for idx in range(start, end):
        cid = chunk_ids[idx]
        emb = embeddings[idx].tolist()
        meta = metadata[idx]
        text = meta.get("text", meta.get("chunk_text", ""))[:MAX_TEXT_LENGTH]
        meta_json = json.dumps(
            {k: v for k, v in meta.items() if k not in ("text", "chunk_text")},
            ensure_ascii=False,
        )[:MAX_METADATA_LENGTH]

        batch_ids.append(cid)
        batch_embeddings.append(emb)
        batch_texts.append(text)
        batch_meta.append(meta_json)

    return [batch_ids, batch_embeddings, batch_texts, batch_meta]


# ---------------------------------------------------------------------------
# Core insertion logic
# ---------------------------------------------------------------------------
def insert_embeddings(
    collection: Collection,
    chunk_ids: List[str],
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    existing_ids: Optional[Set[str]] = None,
) -> tuple[int, int, float]:
    """Insert embeddings into *collection* in batches.

    Skips IDs present in *existing_ids* for idempotency.

    Returns ``(total_inserted, total_skipped, elapsed_seconds)``.
    """
    existing_ids = existing_ids or set()
    total = len(chunk_ids)

    # Filter out already-existing IDs while preserving order
    indices_to_insert = [
        i for i, cid in enumerate(chunk_ids) if cid not in existing_ids
    ]
    skipped = total - len(indices_to_insert)
    if skipped:
        logger.info("Skipping %d already-existing vectors.", skipped)

    if not indices_to_insert:
        logger.info("All vectors already exist – nothing to insert.")
        return 0, skipped, 0.0

    # Build filtered arrays
    filtered_ids = [chunk_ids[i] for i in indices_to_insert]
    filtered_embeddings = embeddings[indices_to_insert]
    filtered_metadata = [metadata[i] for i in indices_to_insert]

    total_to_insert = len(filtered_ids)
    inserted = 0
    start_time = time.time()

    for batch_start in range(0, total_to_insert, batch_size):
        batch_end = min(batch_start + batch_size, total_to_insert)
        batch_num = batch_start // batch_size + 1
        total_batches = (total_to_insert + batch_size - 1) // batch_size

        batch_data = prepare_batch(
            filtered_ids, filtered_embeddings, filtered_metadata,
            batch_start, batch_end,
        )

        # Retry insert for transient failures
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                collection.insert(batch_data)
                batch_count = batch_end - batch_start
                inserted += batch_count
                logger.info(
                    "Batch %d/%d inserted (%d vectors) – progress: %d/%d",
                    batch_num,
                    total_batches,
                    batch_count,
                    inserted,
                    total_to_insert,
                )
                break
            except MilvusException as exc:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Insert batch %d failed (attempt %d/%d): %s – retrying in %ds",
                        batch_num, attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to insert batch {batch_num} after {MAX_RETRIES} "
                        f"attempts: {exc}"
                    ) from exc

    elapsed = time.time() - start_time
    logger.info(
        "Insertion complete: %d vectors in %.2fs (%.1f vectors/sec)",
        inserted, elapsed, inserted / max(elapsed, 1e-9),
    )
    return inserted, skipped, elapsed


# ---------------------------------------------------------------------------
# Post-insert optimisation
# ---------------------------------------------------------------------------
def flush_collection(collection: Collection) -> float:
    """Flush the collection and return elapsed seconds."""
    start = time.time()
    collection.flush()
    elapsed = time.time() - start
    logger.info("Flush completed in %.2fs", elapsed)
    return elapsed


def compact_collection(collection: Collection) -> float:
    """Compact the collection and return elapsed seconds."""
    start = time.time()
    collection.compact()
    # Wait for compaction to complete
    collection.wait_for_compaction_completed()
    elapsed = time.time() - start
    logger.info("Compaction completed in %.2fs", elapsed)
    return elapsed


def build_index(
    collection: Collection,
    force_rebuild: bool = False,
) -> float:
    """Create or rebuild the vector index on the *embedding* field.

    Returns elapsed seconds for index building.
    """
    has_index = False
    try:
        indexes = collection.indexes
        has_index = any(idx.field_name == "embedding" for idx in indexes)
    except Exception:  # noqa: BLE001
        pass

    if has_index and not force_rebuild:
        logger.info("Index already exists – skipping rebuild.")
        return 0.0

    if has_index and force_rebuild:
        logger.info("Dropping existing index for rebuild…")
        collection.drop_index()

    logger.info("Building index with params: %s", INDEX_PARAMS)
    start = time.time()
    collection.create_index(
        field_name="embedding",
        index_params=INDEX_PARAMS,
    )
    elapsed = time.time() - start
    logger.info("Index built in %.2fs", elapsed)
    return elapsed


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------
def save_metrics(metrics: IndexerMetrics, output_dir: str) -> Path:
    """Write metrics JSON to *output_dir* and return the file path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / METRICS_FILENAME
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2), encoding="utf-8"
    )
    logger.info("Metrics saved to %s", metrics_path)
    return metrics_path


# ---------------------------------------------------------------------------
# Rollback helper
# ---------------------------------------------------------------------------
def rollback_inserted(
    collection: Collection,
    inserted_ids: List[str],
) -> None:
    """Best-effort deletion of partially inserted vectors."""
    if not inserted_ids:
        return
    try:
        expr_values = ", ".join(f'"{cid}"' for cid in inserted_ids)
        expr = f"chunk_id in [{expr_values}]"
        collection.delete(expr)
        collection.flush()
        logger.info("Rollback: deleted %d partially inserted vectors.", len(inserted_ids))
    except Exception as exc:  # noqa: BLE001
        logger.error("Rollback failed: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_indexer(
    embeddings_path: str,
    metadata_path: str,
    milvus_host: str = DEFAULT_MILVUS_HOST,
    milvus_port: int = DEFAULT_MILVUS_PORT,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild_index: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> IndexerMetrics:
    """Orchestrate the full indexing pipeline.

    1. Connect to Milvus
    2. Load embeddings & metadata
    3. Validate inputs
    4. Get or create collection
    5. Detect duplicates
    6. Insert in batches
    7. Flush, compact, index
    8. Output metrics
    """
    metrics = IndexerMetrics(batch_size=batch_size, rebuild_index=rebuild_index)
    inserted_ids: List[str] = []

    try:
        # -- 1. Connect -------------------------------------------------------
        connect_to_milvus(host=milvus_host, port=milvus_port)

        # -- 2. Load data -----------------------------------------------------
        embeddings = load_embeddings(embeddings_path)
        metadata = load_metadata(metadata_path)

        # -- 3. Validate -------------------------------------------------------
        validate_inputs(embeddings, metadata)
        n_vectors, embed_dim = embeddings.shape
        metrics.embedding_dimension = embed_dim
        logger.info(
            "Input: %d vectors of dimension %d", n_vectors, embed_dim
        )

        # -- 4. Collection -----------------------------------------------------
        collection = get_or_create_collection(
            name=collection_name,
            embedding_dim=embed_dim,
        )

        # -- 5. Duplicate detection --------------------------------------------
        chunk_ids = [
            m.get("chunk_id", m.get("id", f"chunk_{i}"))
            for i, m in enumerate(metadata)
        ]
        existing_ids = fetch_existing_ids(collection, chunk_ids)
        metrics.total_duplicates_skipped = len(existing_ids)

        # -- 6. Insert ---------------------------------------------------------
        total_inserted, total_skipped, insert_elapsed = insert_embeddings(
            collection=collection,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            metadata=metadata,
            batch_size=batch_size,
            existing_ids=existing_ids,
        )
        # Track which IDs were inserted for potential rollback
        inserted_ids = [
            cid for cid in chunk_ids if cid not in existing_ids
        ][:total_inserted]

        metrics.total_vectors_inserted = total_inserted
        metrics.total_insert_time_seconds = insert_elapsed
        metrics.insert_rate_vectors_per_sec = (
            total_inserted / max(insert_elapsed, 1e-9)
        )
        metrics.num_batches = (
            (total_inserted + batch_size - 1) // batch_size if total_inserted else 0
        )

        # -- 7. Post-insert optimisation ---------------------------------------
        metrics.flush_time_seconds = flush_collection(collection)
        metrics.compact_time_seconds = compact_collection(collection)
        metrics.index_build_time_seconds = build_index(
            collection, force_rebuild=rebuild_index
        )

        # Final entity count
        collection.flush()
        metrics.collection_total_entities = collection.num_entities
        logger.info(
            "Collection '%s' now has %d entities.",
            collection_name,
            metrics.collection_total_entities,
        )

        # -- 8. Save metrics ---------------------------------------------------
        save_metrics(metrics, output_dir)

    except Exception:
        logger.error("Indexing failed – attempting rollback…", exc_info=True)
        try:
            if inserted_ids:
                coll = Collection(name=collection_name, using=CONNECTION_ALIAS)
                rollback_inserted(coll, inserted_ids)
        except Exception:  # noqa: BLE001
            logger.error("Rollback could not be completed.", exc_info=True)
        raise
    finally:
        disconnect_from_milvus()

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milvus Indexer – load embeddings into a Milvus collection."
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=True,
        help="Path to embeddings.npy produced by embedding-generator.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        required=True,
        help="Path to metadata.json produced by embedding-generator.",
    )
    parser.add_argument(
        "--milvus-host",
        type=str,
        default=DEFAULT_MILVUS_HOST,
        help=f"Milvus server hostname (default: {DEFAULT_MILVUS_HOST}).",
    )
    parser.add_argument(
        "--milvus-port",
        type=int,
        default=DEFAULT_MILVUS_PORT,
        help=f"Milvus server port (default: {DEFAULT_MILVUS_PORT}).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Target Milvus collection (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Vectors per insert batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--rebuild-index",
        type=str,
        default="false",
        help="Force rebuild of the vector index after insertion ('true' or 'false').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output metrics (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting Milvus Indexer with args: %s", vars(args))

    metrics = run_indexer(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        rebuild_index=str(args.rebuild_index).lower() in ("true", "1", "yes"),
        output_dir=args.output_dir,
    )

    logger.info("=" * 60)
    logger.info("Indexing Summary")
    logger.info("=" * 60)
    logger.info("  Vectors inserted:    %d", metrics.total_vectors_inserted)
    logger.info("  Duplicates skipped:  %d", metrics.total_duplicates_skipped)
    logger.info("  Insert rate:         %.1f vectors/sec", metrics.insert_rate_vectors_per_sec)
    logger.info("  Insert time:         %.2fs", metrics.total_insert_time_seconds)
    logger.info("  Index build time:    %.2fs", metrics.index_build_time_seconds)
    logger.info("  Collection entities: %d", metrics.collection_total_entities)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
