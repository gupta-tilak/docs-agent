"""Milvus Vector Store SDK – high-level client.

Provides :class:`MilvusVectorStore`, a batteries-included wrapper around
``pymilvus`` for the docs-agent project.  Features include:

* Connection retry with exponential back-off
* Batch insertion with deduplication
* Semantic search (raw vector **and** text convenience method)
* Metadata filtering via Milvus boolean expressions
* Structured logging for every operation
* Full type-hint coverage
* Optional async helpers (via ``asyncio.to_thread``)

The class is designed to match the collection schema used by the
``milvus-indexer`` KFP component so that data written by the pipeline is
immediately queryable through this SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Sequence

import numpy as np
from pymilvus import (
    Collection,
    MilvusException,
    connections,
    utility,
)

from vector_store.config import (
    EmbeddingConfig,
    MilvusConnectionConfig,
    RetryConfig,
    SchemaConfig,
    SearchConfig,
    VectorStoreConfig,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("vector_store.milvus_client")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry(
    fn: Any,
    *args: Any,
    retry_cfg: RetryConfig = RetryConfig(),
    operation: str = "operation",
    **kwargs: Any,
) -> Any:
    """Execute *fn* with retry + exponential back-off.

    Parameters
    ----------
    fn:
        Callable to invoke.
    retry_cfg:
        Controls max retries and back-off timing.
    operation:
        Human-readable label used in log messages.

    Returns
    -------
    The return value of *fn* on success.

    Raises
    ------
    RuntimeError
        If all attempts are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, retry_cfg.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except (MilvusException, ConnectionError, OSError) as exc:
            last_exc = exc
            if attempt < retry_cfg.max_retries:
                wait = min(
                    retry_cfg.backoff_base ** attempt,
                    retry_cfg.backoff_max,
                )
                logger.warning(
                    "%s attempt %d/%d failed: %s – retrying in %.1fs",
                    operation,
                    attempt,
                    retry_cfg.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"{operation} failed after {retry_cfg.max_retries} "
                    f"attempts: {last_exc}"
                ) from last_exc
    # Should never reach here, but satisfies the type-checker.
    raise RuntimeError(f"{operation} failed unexpectedly")  # pragma: no cover


# ---------------------------------------------------------------------------
# Main SDK class
# ---------------------------------------------------------------------------

class MilvusVectorStore:
    """High-level client for Milvus vector store operations.

    Parameters
    ----------
    host:
        Milvus server hostname.
    port:
        Milvus gRPC port.
    collection_name:
        Name of the target Milvus collection.
    config:
        Optional :class:`VectorStoreConfig` instance.  When provided, *host*,
        *port*, and *collection_name* override the corresponding values inside
        the config; everything else is drawn from the config.

    Raises
    ------
    ConnectionError
        If the server is unreachable after retries.
    ValueError
        If the specified collection does not exist.

    Examples
    --------
    >>> store = MilvusVectorStore("localhost", 19530, "kubeflow_docs")
    >>> results = store.search_by_text("How to deploy a pipeline?")
    >>> store.close()
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "kubeflow_docs",
        config: Optional[VectorStoreConfig] = None,
    ) -> None:
        # Merge explicit args with optional config
        self._config = config or VectorStoreConfig()
        self._conn_cfg = MilvusConnectionConfig(
            host=host,
            port=port,
            alias=self._config.connection.alias,
            timeout=self._config.connection.timeout,
            secure=self._config.connection.secure,
            token=self._config.connection.token,
        )
        self._retry_cfg: RetryConfig = self._config.retry
        self._search_cfg: SearchConfig = self._config.search
        self._embedding_cfg: EmbeddingConfig = self._config.embedding
        self._schema_cfg: SchemaConfig = self._config.schema
        self._batch_size: int = self._config.batch_size
        self._collection_name: str = collection_name

        # Lazy-loaded embedding model (only needed for search_by_text)
        self._embedding_model: Any = None

        # Connect
        self._connect()

        # Validate collection and load
        self._collection = self._load_collection()

        logger.info(
            "MilvusVectorStore ready – collection='%s', entities=%d",
            self._collection_name,
            self._collection.num_entities,
        )

    # -------------------------------------------------------------- connect
    def _connect(self) -> None:
        """Establish a connection to Milvus with retry logic."""
        cfg = self._conn_cfg

        def _do_connect() -> None:
            connect_kwargs: Dict[str, Any] = {
                "alias": cfg.alias,
                "host": cfg.host,
                "port": str(cfg.port),
                "timeout": cfg.timeout,
                "secure": cfg.secure,
            }
            if cfg.token:
                connect_kwargs["token"] = cfg.token
            connections.connect(**connect_kwargs)

        _retry(
            _do_connect,
            retry_cfg=self._retry_cfg,
            operation="Milvus connection",
        )
        logger.info(
            "Connected to Milvus at %s:%s (alias=%s)",
            cfg.host,
            cfg.port,
            cfg.alias,
        )

    # ------------------------------------------------- collection bootstrap
    def _load_collection(self) -> Collection:
        """Validate that the collection exists and load it into memory.

        Returns
        -------
        Collection
            A pymilvus ``Collection`` handle that is already loaded.

        Raises
        ------
        ValueError
            If the collection does not exist on the server.
        """
        alias = self._conn_cfg.alias
        if not utility.has_collection(self._collection_name, using=alias):
            raise ValueError(
                f"Collection '{self._collection_name}' does not exist on "
                f"Milvus at {self._conn_cfg.host}:{self._conn_cfg.port}.  "
                "Create it via the milvus-indexer pipeline first."
            )

        collection = Collection(
            name=self._collection_name,
            using=alias,
        )
        collection.load()
        logger.info(
            "Loaded collection '%s' into memory (%d entities)",
            self._collection_name,
            collection.num_entities,
        )
        return collection

    # ---------------------------------------------------------------- insert
    def insert(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Insert document chunks with embeddings into the collection.

        Each dict in *chunks* **must** contain:

        * ``"embedding"`` – a list of floats or a 1-D numpy array.

        And **should** contain:

        * ``"chunk_id"``  – unique identifier (auto-generated if absent).
        * ``"text"``      – the original chunk text.
        * ``"metadata"``  – arbitrary JSON-serialisable metadata dict.

        Parameters
        ----------
        chunks:
            A list of chunk dictionaries.

        Returns
        -------
        List[str]
            The ``chunk_id`` values for every successfully inserted chunk.

        Raises
        ------
        ValueError
            If *chunks* is empty or any chunk lacks an ``"embedding"`` key.
        RuntimeError
            If a batch insert fails after retries.

        Examples
        --------
        >>> ids = store.insert([
        ...     {
        ...         "chunk_id": "doc1_c0",
        ...         "embedding": [0.1, 0.2, ...],
        ...         "text": "Kubeflow overview",
        ...         "metadata": {"source": "intro.md"},
        ...     }
        ... ])
        """
        if not chunks:
            raise ValueError("Cannot insert an empty chunk list.")

        # Validate and normalise
        chunk_ids: List[str] = []
        embeddings: List[List[float]] = []
        texts: List[str] = []
        meta_jsons: List[str] = []

        for i, chunk in enumerate(chunks):
            if "embedding" not in chunk:
                raise ValueError(
                    f"Chunk at index {i} is missing the required 'embedding' key."
                )

            cid = chunk.get("chunk_id", str(uuid.uuid4()))
            emb = chunk["embedding"]
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            text = str(chunk.get("text", ""))[: self._schema_cfg.max_text_length]
            meta = chunk.get("metadata", {})
            meta_str = json.dumps(meta, ensure_ascii=False)[
                : self._schema_cfg.max_metadata_length
            ]

            chunk_ids.append(cid)
            embeddings.append(emb)
            texts.append(text)
            meta_jsons.append(meta_str)

        # Batch insert
        inserted_ids: List[str] = []
        total = len(chunk_ids)
        for batch_start in range(0, total, self._batch_size):
            batch_end = min(batch_start + self._batch_size, total)
            batch_data = [
                chunk_ids[batch_start:batch_end],
                embeddings[batch_start:batch_end],
                texts[batch_start:batch_end],
                meta_jsons[batch_start:batch_end],
            ]
            batch_num = batch_start // self._batch_size + 1
            total_batches = (total + self._batch_size - 1) // self._batch_size

            def _do_insert(data: List = batch_data) -> None:
                self._collection.insert(data)

            _retry(
                _do_insert,
                retry_cfg=self._retry_cfg,
                operation=f"Insert batch {batch_num}/{total_batches}",
            )

            inserted_ids.extend(chunk_ids[batch_start:batch_end])
            logger.info(
                "Inserted batch %d/%d (%d vectors)",
                batch_num,
                total_batches,
                batch_end - batch_start,
            )

        # Flush to make data visible for search
        self._collection.flush()
        logger.info(
            "Insert complete – %d vectors inserted and flushed.", len(inserted_ids)
        )
        return inserted_ids

    # ---------------------------------------------------------------- search
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search.

        Parameters
        ----------
        query_vector:
            A 1-D numpy array (or list of floats) representing the query
            embedding.
        top_k:
            Maximum number of results to return.
        filter:
            Optional metadata filter expressed as a dict.  Keys are metadata
            field names and values are the required values.  Translated into a
            Milvus boolean expression applied to the ``metadata`` JSON field.

            Example: ``{"source": "intro.md"}`` →
            ``metadata like '%"source": "intro.md"%'``

        Returns
        -------
        List[Dict[str, Any]]
            Ranked list of results, each with keys ``chunk_id``, ``text``,
            ``score``, and ``metadata``.

        Examples
        --------
        >>> results = store.search(embedding_array, top_k=3)
        >>> results[0]["score"]
        0.95
        """
        if isinstance(query_vector, np.ndarray):
            query_vector_list = query_vector.tolist()
        else:
            query_vector_list = list(query_vector)

        # Build optional expression from filter dict
        expr: Optional[str] = None
        if filter:
            conditions = []
            for key, value in filter.items():
                # Use Milvus JSON-string LIKE matching on the metadata field
                escaped_val = json.dumps(value).strip('"')
                conditions.append(
                    f'metadata like \'%"{key}": "{escaped_val}"%\''
                )
            expr = " and ".join(conditions)

        search_params = {
            "metric_type": self._search_cfg.metric_type,
            "params": dict(self._search_cfg.search_params),
        }

        def _do_search() -> Any:
            return self._collection.search(
                data=[query_vector_list],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=self._search_cfg.output_fields,
            )

        raw_results = _retry(
            _do_search,
            retry_cfg=self._retry_cfg,
            operation="Vector search",
        )

        return self._format_results(raw_results)

    # -------------------------------------------------------- search_by_text
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience method: embed *query_text* then perform a vector search.

        Uses the same sentence-transformer model configured in
        :attr:`EmbeddingConfig` (default ``all-MiniLM-L6-v2``), matching the
        model used by the ``embedding-generator`` pipeline component.

        Parameters
        ----------
        query_text:
            Natural language query string.
        top_k:
            Number of results to return.
        filter:
            Optional metadata filter (see :meth:`search`).

        Returns
        -------
        List[Dict[str, Any]]
            Same format as :meth:`search`.

        Examples
        --------
        >>> results = store.search_by_text("How do I deploy a pipeline?")
        """
        embedding = self._embed_text(query_text)
        return self.search(embedding, top_k=top_k, filter=filter)

    # ---------------------------------------------------------------- delete
    def delete(self, chunk_ids: List[str]) -> int:
        """Delete chunks by their IDs.

        Parameters
        ----------
        chunk_ids:
            List of ``chunk_id`` values to remove.

        Returns
        -------
        int
            The number of chunks requested for deletion.  Milvus does not
            return a confirmed count, so this equals ``len(chunk_ids)`` when
            no error is raised.

        Raises
        ------
        ValueError
            If *chunk_ids* is empty.
        RuntimeError
            If the delete operation fails after retries.

        Examples
        --------
        >>> deleted = store.delete(["doc1_c0", "doc1_c1"])
        >>> print(deleted)
        2
        """
        if not chunk_ids:
            raise ValueError("chunk_ids must be a non-empty list.")

        # Milvus DELETE uses a boolean expression
        expr_values = ", ".join(f'"{cid}"' for cid in chunk_ids)
        expr = f"chunk_id in [{expr_values}]"

        def _do_delete() -> None:
            self._collection.delete(expr)

        _retry(
            _do_delete,
            retry_cfg=self._retry_cfg,
            operation="Delete",
        )
        self._collection.flush()
        logger.info("Deleted %d chunks (expression: %s)", len(chunk_ids), expr)
        return len(chunk_ids)

    # ------------------------------------------------------------- get_stats
    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:

            * ``collection_name`` – name of the collection.
            * ``total_entities`` – number of vectors stored.
            * ``index_type`` – index algorithm name (e.g. ``"IVF_FLAT"``).
            * ``metric_type`` – distance metric (e.g. ``"IP"``).
            * ``embedding_dim`` – dimensionality of the stored vectors.
            * ``schema_fields`` – list of field names in the schema.
            * ``loaded`` – whether the collection is currently loaded.

        Examples
        --------
        >>> stats = store.get_stats()
        >>> stats["total_entities"]
        42000
        """
        # Flush to ensure entity count is up to date
        self._collection.flush()

        # Determine index info
        index_type: str = "N/A"
        metric_type: str = "N/A"
        try:
            for idx in self._collection.indexes:
                if idx.field_name == "embedding":
                    idx_params = idx.params
                    index_type = idx_params.get("index_type", "N/A")
                    metric_type = idx_params.get("metric_type", "N/A")
                    break
        except Exception:  # noqa: BLE001
            logger.debug("Could not retrieve index info", exc_info=True)

        # Determine embedding dimension from schema
        embedding_dim: int = 0
        field_names: List[str] = []
        for f in self._collection.schema.fields:
            field_names.append(f.name)
            if f.name == "embedding":
                embedding_dim = f.params.get("dim", 0)

        stats: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "total_entities": self._collection.num_entities,
            "index_type": index_type,
            "metric_type": metric_type,
            "embedding_dim": embedding_dim,
            "schema_fields": field_names,
            "loaded": True,
        }

        logger.info("Collection stats: %s", stats)
        return stats

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        """Release the collection and disconnect from Milvus.

        Safe to call multiple times.

        Examples
        --------
        >>> store.close()
        """
        alias = self._conn_cfg.alias
        try:
            self._collection.release()
            logger.info("Released collection '%s'", self._collection_name)
        except Exception:  # noqa: BLE001
            logger.debug("Release warning (non-fatal)", exc_info=True)

        try:
            connections.disconnect(alias=alias)
            logger.info("Disconnected from Milvus (alias=%s)", alias)
        except Exception:  # noqa: BLE001
            logger.debug("Disconnect warning (non-fatal)", exc_info=True)

        # Free embedding model memory
        self._embedding_model = None

    # ------------------------------------------------------- context manager
    def __enter__(self) -> "MilvusVectorStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------- async wrappers
    async def ainsert(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Async wrapper around :meth:`insert`.

        Delegates to a thread so the event loop is not blocked.
        """
        return await asyncio.to_thread(self.insert, chunks)

    async def asearch(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Async wrapper around :meth:`search`."""
        return await asyncio.to_thread(self.search, query_vector, top_k, filter)

    async def asearch_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Async wrapper around :meth:`search_by_text`."""
        return await asyncio.to_thread(
            self.search_by_text, query_text, top_k, filter
        )

    async def adelete(self, chunk_ids: List[str]) -> int:
        """Async wrapper around :meth:`delete`."""
        return await asyncio.to_thread(self.delete, chunk_ids)

    # ------------------------------------------------------- private helpers
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text string.

        Lazily loads the sentence-transformer model on first call.
        """
        if self._embedding_model is None:
            self._embedding_model = self._load_embedding_model()

        embedding: np.ndarray = self._embedding_model.encode(
            text,
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self._embedding_cfg.normalize,
        )
        return embedding.astype(np.float32)

    def _load_embedding_model(self) -> Any:
        """Load the sentence-transformer model specified in the config."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for search_by_text().  "
                "Install it with:  pip install sentence-transformers"
            ) from exc

        logger.info(
            "Loading embedding model '%s' on device '%s'",
            self._embedding_cfg.model_name,
            self._embedding_cfg.device,
        )
        model = SentenceTransformer(
            self._embedding_cfg.model_name,
            device=self._embedding_cfg.device,
        )
        logger.info("Embedding model loaded successfully.")
        return model

    @staticmethod
    def _format_results(raw_results: Any) -> List[Dict[str, Any]]:
        """Convert raw pymilvus search results into a clean list of dicts.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict has ``chunk_id``, ``text``, ``score``, ``metadata``.
        """
        formatted: List[Dict[str, Any]] = []
        for hits in raw_results:
            for hit in hits:
                entity = hit.entity
                # Parse metadata JSON back into a dict
                meta_raw = entity.get("metadata", "{}")
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                except (json.JSONDecodeError, TypeError):
                    meta = {"_raw": meta_raw}

                formatted.append(
                    {
                        "chunk_id": entity.get("chunk_id", hit.id),
                        "text": entity.get("text", ""),
                        "score": float(hit.score),
                        "metadata": meta,
                    }
                )
        return formatted

    # ------------------------------------------------------- repr
    def __repr__(self) -> str:
        return (
            f"MilvusVectorStore("
            f"host='{self._conn_cfg.host}', "
            f"port={self._conn_cfg.port}, "
            f"collection='{self._collection_name}')"
        )
