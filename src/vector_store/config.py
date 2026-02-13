"""Configuration dataclasses for the Milvus vector store SDK.

Centralises all tuneable parameters – connection details, retry behaviour,
search defaults, embedding model, and collection schema constraints – so that
they can be overridden in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Connection configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MilvusConnectionConfig:
    """Parameters for connecting to a Milvus instance.

    Attributes:
        host: Hostname or IP address of the Milvus server.
        port: gRPC port of the Milvus server (default ``19530``).
        alias: Pymilvus connection alias.  Using a unique alias allows
            multiple concurrent connections.
        timeout: Connection timeout in seconds.
        secure: Whether to use TLS for the connection.
        token: Optional authentication token (Milvus ≥ 2.2).
    """

    host: str = "localhost"
    port: int = 19530
    alias: str = "vector_store_sdk"
    timeout: float = 30.0
    secure: bool = False
    token: Optional[str] = None


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryConfig:
    """Retry parameters with exponential back-off.

    Attributes:
        max_retries: Maximum number of attempts before giving up.
        backoff_base: Base duration (seconds) for exponential back-off.
            Wait time for attempt *n* = ``backoff_base ** n``.
        backoff_max: Upper-bound on the wait time between retries.
    """

    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 30.0


# ---------------------------------------------------------------------------
# Search defaults
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchConfig:
    """Default parameters for similarity search.

    Attributes:
        metric_type: Distance metric used by the Milvus index
            (``"IP"`` for inner product, ``"L2"`` for Euclidean).
        top_k: Number of nearest neighbours to return.
        search_params: Extra parameters forwarded to ``collection.search``.
        output_fields: Fields to include in search results.
    """

    metric_type: str = "IP"
    top_k: int = 5
    search_params: Dict[str, Any] = field(
        default_factory=lambda: {"nprobe": 16},
    )
    output_fields: list[str] = field(
        default_factory=lambda: ["chunk_id", "text", "metadata"],
    )


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the sentence-transformer used by ``search_by_text``.

    Attributes:
        model_name: Hugging Face model identifier.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        batch_size: Batch size when encoding multiple texts at once.
        normalize: Whether to L2-normalise embeddings (recommended for IP
            metric).
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


# ---------------------------------------------------------------------------
# Collection schema limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaConfig:
    """Size limits mirroring the collection schema in the milvus-indexer.

    Attributes:
        max_chunk_id_length: Maximum VARCHAR length for ``chunk_id``.
        max_text_length: Maximum VARCHAR length for ``text``.
        max_metadata_length: Maximum VARCHAR length for ``metadata`` JSON.
        default_embedding_dim: Default embedding dimensionality.
    """

    max_chunk_id_length: int = 512
    max_text_length: int = 65_535
    max_metadata_length: int = 65_535
    default_embedding_dim: int = 384


# ---------------------------------------------------------------------------
# Index configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexConfig:
    """Parameters for the Milvus vector index.

    Attributes:
        index_type: Index algorithm (e.g. ``"IVF_FLAT"``, ``"HNSW"``).
        metric_type: Distance metric.
        params: Algorithm-specific parameters.
    """

    index_type: str = "IVF_FLAT"
    metric_type: str = "IP"
    params: Dict[str, Any] = field(
        default_factory=lambda: {"nlist": 128},
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the format expected by ``collection.create_index``."""
        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": dict(self.params),
        }


# ---------------------------------------------------------------------------
# Aggregate SDK configuration
# ---------------------------------------------------------------------------
@dataclass
class VectorStoreConfig:
    """Top-level configuration that bundles every sub-config.

    Instantiate with defaults::

        cfg = VectorStoreConfig()

    Or override selectively::

        cfg = VectorStoreConfig(
            connection=MilvusConnectionConfig(host="milvus.example.com"),
            retry=RetryConfig(max_retries=5),
        )
    """

    connection: MilvusConnectionConfig = field(
        default_factory=MilvusConnectionConfig,
    )
    retry: RetryConfig = field(default_factory=RetryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    collection_name: str = "kubeflow_docs"
    batch_size: int = 100
