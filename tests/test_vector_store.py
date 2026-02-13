"""Unit and integration tests for the Milvus vector store SDK.

Unit tests mock ``pymilvus`` entirely so they can run without a live server.
Integration tests (marked ``@pytest.mark.integration``) require a running
Milvus instance and are skipped by default – enable them with::

    pytest -m integration --milvus-host localhost --milvus-port 19530
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the SDK importable from the repo root
# ---------------------------------------------------------------------------
SDK_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SDK_ROOT))

from vector_store.config import (
    EmbeddingConfig,
    IndexConfig,
    MilvusConnectionConfig,
    RetryConfig,
    SchemaConfig,
    SearchConfig,
    VectorStoreConfig,
)
from vector_store.milvus_client import MilvusVectorStore, _retry


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def default_config() -> VectorStoreConfig:
    """A VectorStoreConfig with test-friendly defaults."""
    return VectorStoreConfig(
        connection=MilvusConnectionConfig(host="test-host", port=19530),
        retry=RetryConfig(max_retries=2, backoff_base=0.01),
        batch_size=2,
    )


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Three sample chunks with embeddings of dimension 4."""
    rng = np.random.default_rng(42)
    return [
        {
            "chunk_id": "doc1_c0",
            "embedding": rng.random(4, dtype=np.float32).tolist(),
            "text": "Kubeflow is an ML toolkit for Kubernetes.",
            "metadata": {"source": "intro.md", "position": 0},
        },
        {
            "chunk_id": "doc1_c1",
            "embedding": rng.random(4, dtype=np.float32).tolist(),
            "text": "Pipelines orchestrate ML workflows.",
            "metadata": {"source": "intro.md", "position": 1},
        },
        {
            "chunk_id": "doc2_c0",
            "embedding": rng.random(4, dtype=np.float32).tolist(),
            "text": "Training operators support TensorFlow.",
            "metadata": {"source": "training.md", "position": 0},
        },
    ]


@pytest.fixture
def sample_query_vector() -> np.ndarray:
    return np.random.default_rng(99).random(4).astype(np.float32)


def _make_mock_collection(num_entities: int = 100) -> MagicMock:
    """Return a mock pymilvus Collection with reasonable defaults."""
    mock_col = MagicMock(spec=["insert", "delete", "search", "flush",
                                "release", "load", "schema", "indexes",
                                "num_entities", "compact",
                                "wait_for_compaction_completed", "query"])
    type(mock_col).num_entities = PropertyMock(return_value=num_entities)

    # Schema with an embedding field
    emb_field = MagicMock()
    emb_field.name = "embedding"
    emb_field.params = {"dim": 4}
    text_field = MagicMock()
    text_field.name = "text"
    text_field.params = {}
    meta_field = MagicMock()
    meta_field.name = "metadata"
    meta_field.params = {}
    id_field = MagicMock()
    id_field.name = "chunk_id"
    id_field.params = {}
    mock_col.schema.fields = [id_field, emb_field, text_field, meta_field]

    # Default index
    idx = MagicMock()
    idx.field_name = "embedding"
    idx.params = {"index_type": "IVF_FLAT", "metric_type": "IP"}
    mock_col.indexes = [idx]

    return mock_col


def _build_store_with_mocks(
    config: VectorStoreConfig | None = None,
    num_entities: int = 100,
) -> tuple[MilvusVectorStore, MagicMock]:
    """Construct a ``MilvusVectorStore`` with all pymilvus calls mocked."""
    cfg = config or VectorStoreConfig(
        connection=MilvusConnectionConfig(host="mock-host"),
        retry=RetryConfig(max_retries=2, backoff_base=0.01),
        batch_size=2,
    )
    mock_col = _make_mock_collection(num_entities)

    with (
        patch("vector_store.milvus_client.connections") as mock_conns,
        patch("vector_store.milvus_client.utility") as mock_util,
        patch("vector_store.milvus_client.Collection", return_value=mock_col),
    ):
        mock_util.has_collection.return_value = True
        store = MilvusVectorStore(
            host=cfg.connection.host,
            port=cfg.connection.port,
            collection_name=cfg.collection_name,
            config=cfg,
        )
    return store, mock_col


# =========================================================================
# Unit tests – _retry helper
# =========================================================================

class TestRetry:
    """Tests for the standalone ``_retry`` helper."""

    def test_succeeds_first_try(self) -> None:
        fn = MagicMock(return_value="ok")
        result = _retry(fn, retry_cfg=RetryConfig(max_retries=3, backoff_base=0.01), operation="test")
        assert result == "ok"
        fn.assert_called_once()

    def test_retries_then_succeeds(self) -> None:
        from pymilvus import MilvusException

        fn = MagicMock(side_effect=[MilvusException(message="fail"), "ok"])
        result = _retry(fn, retry_cfg=RetryConfig(max_retries=2, backoff_base=0.01), operation="test")
        assert result == "ok"
        assert fn.call_count == 2

    def test_exhausts_retries(self) -> None:
        from pymilvus import MilvusException

        fn = MagicMock(side_effect=MilvusException(message="persistent failure"))
        with pytest.raises(RuntimeError, match="persistent failure"):
            _retry(fn, retry_cfg=RetryConfig(max_retries=2, backoff_base=0.01), operation="test")
        assert fn.call_count == 2


# =========================================================================
# Unit tests – MilvusVectorStore.__init__
# =========================================================================

class TestInit:
    """Tests for initialisation and connection handling."""

    def test_successful_init(self) -> None:
        store, mock_col = _build_store_with_mocks()
        assert store._collection_name == "kubeflow_docs"
        mock_col.load.assert_called_once()

    def test_collection_not_found_raises(self) -> None:
        with (
            patch("vector_store.milvus_client.connections"),
            patch("vector_store.milvus_client.utility") as mock_util,
        ):
            mock_util.has_collection.return_value = False
            with pytest.raises(ValueError, match="does not exist"):
                MilvusVectorStore(
                    host="mock-host",
                    port=19530,
                    collection_name="nonexistent",
                    config=VectorStoreConfig(
                        retry=RetryConfig(max_retries=1, backoff_base=0.01),
                    ),
                )

    def test_connection_failure_raises_after_retries(self) -> None:
        from pymilvus import MilvusException

        with patch(
            "vector_store.milvus_client.connections"
        ) as mock_conns:
            mock_conns.connect.side_effect = MilvusException(message="unreachable")
            with pytest.raises(RuntimeError, match="unreachable"):
                MilvusVectorStore(
                    host="bad-host",
                    config=VectorStoreConfig(
                        retry=RetryConfig(max_retries=2, backoff_base=0.01),
                    ),
                )

    def test_context_manager(self) -> None:
        store, mock_col = _build_store_with_mocks()
        with store as s:
            assert s is store
        mock_col.release.assert_called_once()


# =========================================================================
# Unit tests – insert
# =========================================================================

class TestInsert:
    """Tests for the insert method."""

    def test_insert_single_batch(self, sample_chunks: List[Dict]) -> None:
        store, mock_col = _build_store_with_mocks()
        store._batch_size = 10  # single batch
        ids = store.insert(sample_chunks)
        assert ids == ["doc1_c0", "doc1_c1", "doc2_c0"]
        mock_col.insert.assert_called_once()
        mock_col.flush.assert_called()

    def test_insert_multiple_batches(self, sample_chunks: List[Dict]) -> None:
        store, mock_col = _build_store_with_mocks()
        store._batch_size = 2  # forces 2 batches for 3 chunks
        ids = store.insert(sample_chunks)
        assert len(ids) == 3
        assert mock_col.insert.call_count == 2

    def test_insert_auto_generates_chunk_ids(self) -> None:
        store, mock_col = _build_store_with_mocks()
        chunks = [{"embedding": [0.1, 0.2, 0.3, 0.4]}]
        ids = store.insert(chunks)
        assert len(ids) == 1
        # Auto-generated ID should be a valid UUID
        import uuid
        uuid.UUID(ids[0])  # raises if invalid

    def test_insert_empty_raises(self) -> None:
        store, _ = _build_store_with_mocks()
        with pytest.raises(ValueError, match="empty"):
            store.insert([])

    def test_insert_missing_embedding_raises(self) -> None:
        store, _ = _build_store_with_mocks()
        with pytest.raises(ValueError, match="embedding"):
            store.insert([{"text": "no embedding"}])

    def test_insert_numpy_embedding(self) -> None:
        store, mock_col = _build_store_with_mocks()
        chunks = [
            {
                "chunk_id": "np_chunk",
                "embedding": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "text": "numpy test",
            }
        ]
        ids = store.insert(chunks)
        assert ids == ["np_chunk"]
        # Verify embedding was converted to list
        inserted_data = mock_col.insert.call_args[0][0]
        assert isinstance(inserted_data[1][0], list)


# =========================================================================
# Unit tests – search
# =========================================================================

class TestSearch:
    """Tests for the search and search_by_text methods."""

    def _mock_search_results(self) -> MagicMock:
        """Create mock search results mimicking pymilvus output."""
        hit = MagicMock()
        hit.id = "doc1_c0"
        hit.score = 0.95
        hit.entity.get = lambda field, default="": {
            "chunk_id": "doc1_c0",
            "text": "Kubeflow overview",
            "metadata": json.dumps({"source": "intro.md"}),
        }.get(field, default)

        hits = MagicMock()
        hits.__iter__ = MagicMock(return_value=iter([hit]))

        results = MagicMock()
        results.__iter__ = MagicMock(return_value=iter([hits]))
        return results

    def test_search_basic(self, sample_query_vector: np.ndarray) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_col.search.return_value = self._mock_search_results()

        results = store.search(sample_query_vector, top_k=3)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "doc1_c0"
        assert results[0]["score"] == 0.95
        assert results[0]["text"] == "Kubeflow overview"
        assert results[0]["metadata"] == {"source": "intro.md"}

    def test_search_with_filter(self, sample_query_vector: np.ndarray) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_col.search.return_value = self._mock_search_results()

        store.search(sample_query_vector, filter={"source": "intro.md"})

        call_kwargs = mock_col.search.call_args[1]
        assert "intro.md" in call_kwargs["expr"]

    def test_search_passes_top_k(self, sample_query_vector: np.ndarray) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_col.search.return_value = self._mock_search_results()

        store.search(sample_query_vector, top_k=10)

        call_kwargs = mock_col.search.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_search_by_text(self) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_col.search.return_value = self._mock_search_results()

        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        )
        store._embedding_model = mock_model

        results = store.search_by_text("How to deploy?")

        assert len(results) == 1
        mock_model.encode.assert_called_once()
        mock_col.search.assert_called_once()

    def test_search_by_text_lazy_loads_model(self) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_col.search.return_value = self._mock_search_results()

        assert store._embedding_model is None

        with patch(
            "vector_store.milvus_client.MilvusVectorStore._load_embedding_model"
        ) as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array(
                [0.1, 0.2, 0.3, 0.4], dtype=np.float32
            )
            mock_load.return_value = mock_model

            store.search_by_text("test query")

            mock_load.assert_called_once()


# =========================================================================
# Unit tests – delete
# =========================================================================

class TestDelete:
    """Tests for the delete method."""

    def test_delete_basic(self) -> None:
        store, mock_col = _build_store_with_mocks()
        count = store.delete(["doc1_c0", "doc1_c1"])
        assert count == 2
        mock_col.delete.assert_called_once()
        # Expression should contain both IDs
        expr = mock_col.delete.call_args[0][0]
        assert "doc1_c0" in expr
        assert "doc1_c1" in expr

    def test_delete_empty_raises(self) -> None:
        store, _ = _build_store_with_mocks()
        with pytest.raises(ValueError, match="non-empty"):
            store.delete([])


# =========================================================================
# Unit tests – get_stats
# =========================================================================

class TestGetStats:
    """Tests for the get_stats method."""

    def test_get_stats_returns_expected_keys(self) -> None:
        store, mock_col = _build_store_with_mocks(num_entities=42)
        stats = store.get_stats()

        assert stats["collection_name"] == "kubeflow_docs"
        assert stats["total_entities"] == 42
        assert stats["index_type"] == "IVF_FLAT"
        assert stats["metric_type"] == "IP"
        assert stats["embedding_dim"] == 4
        assert "chunk_id" in stats["schema_fields"]
        assert "embedding" in stats["schema_fields"]
        assert stats["loaded"] is True


# =========================================================================
# Unit tests – close
# =========================================================================

class TestClose:
    """Tests for the close method."""

    def test_close_releases_and_disconnects(self) -> None:
        store, mock_col = _build_store_with_mocks()

        with patch("vector_store.milvus_client.connections") as mock_conns:
            store.close()

        mock_col.release.assert_called_once()
        # Embedding model should be cleared
        assert store._embedding_model is None

    def test_close_tolerates_exceptions(self) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_col.release.side_effect = Exception("release failed")

        with patch("vector_store.milvus_client.connections") as mock_conns:
            mock_conns.disconnect.side_effect = Exception("disconnect failed")
            # Should not raise
            store.close()


# =========================================================================
# Unit tests – config dataclasses
# =========================================================================

class TestConfig:
    """Tests for configuration dataclasses."""

    def test_default_vector_store_config(self) -> None:
        cfg = VectorStoreConfig()
        assert cfg.connection.host == "localhost"
        assert cfg.connection.port == 19530
        assert cfg.retry.max_retries == 3
        assert cfg.search.metric_type == "IP"
        assert cfg.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.collection_name == "kubeflow_docs"

    def test_custom_config(self) -> None:
        cfg = VectorStoreConfig(
            connection=MilvusConnectionConfig(host="custom", port=9999),
            retry=RetryConfig(max_retries=5),
            collection_name="my_collection",
        )
        assert cfg.connection.host == "custom"
        assert cfg.connection.port == 9999
        assert cfg.retry.max_retries == 5
        assert cfg.collection_name == "my_collection"

    def test_index_config_to_dict(self) -> None:
        idx = IndexConfig()
        d = idx.to_dict()
        assert d["index_type"] == "IVF_FLAT"
        assert d["metric_type"] == "IP"
        assert "nlist" in d["params"]

    def test_frozen_connection_config(self) -> None:
        cfg = MilvusConnectionConfig()
        with pytest.raises(AttributeError):
            cfg.host = "changed"  # type: ignore[misc]


# =========================================================================
# Unit tests – async wrappers
# =========================================================================

class TestAsync:
    """Tests for async wrapper methods."""

    @pytest.mark.asyncio
    async def test_ainsert(self, sample_chunks: List[Dict]) -> None:
        store, mock_col = _build_store_with_mocks()
        store._batch_size = 10
        ids = await store.ainsert(sample_chunks)
        assert len(ids) == 3

    @pytest.mark.asyncio
    async def test_asearch(self, sample_query_vector: np.ndarray) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))
        mock_col.search.return_value = mock_results

        results = await store.asearch(sample_query_vector, top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_adelete(self) -> None:
        store, mock_col = _build_store_with_mocks()
        count = await store.adelete(["id1"])
        assert count == 1

    @pytest.mark.asyncio
    async def test_asearch_by_text(self) -> None:
        store, mock_col = _build_store_with_mocks()
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))
        mock_col.search.return_value = mock_results

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        )
        store._embedding_model = mock_model

        results = await store.asearch_by_text("test query")
        assert results == []


# =========================================================================
# Unit tests – _format_results
# =========================================================================

class TestFormatResults:
    """Tests for the static _format_results helper."""

    def test_empty_results(self) -> None:
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))
        assert MilvusVectorStore._format_results(mock_results) == []

    def test_malformed_metadata_handled(self) -> None:
        hit = MagicMock()
        hit.id = "bad_meta"
        hit.score = 0.5
        hit.entity.get = lambda field, default="": {
            "chunk_id": "bad_meta",
            "text": "text",
            "metadata": "not valid json{{{",
        }.get(field, default)

        hits = MagicMock()
        hits.__iter__ = MagicMock(return_value=iter([hit]))
        results = MagicMock()
        results.__iter__ = MagicMock(return_value=iter([hits]))

        formatted = MilvusVectorStore._format_results(results)
        assert len(formatted) == 1
        assert "_raw" in formatted[0]["metadata"]

    def test_multiple_hits(self) -> None:
        def _make_hit(cid: str, score: float) -> MagicMock:
            hit = MagicMock()
            hit.id = cid
            hit.score = score
            hit.entity.get = lambda field, default="", _cid=cid: {
                "chunk_id": _cid,
                "text": f"text for {_cid}",
                "metadata": "{}",
            }.get(field, default)
            return hit

        hits = MagicMock()
        hits.__iter__ = MagicMock(
            return_value=iter([_make_hit("a", 0.9), _make_hit("b", 0.8)])
        )
        results = MagicMock()
        results.__iter__ = MagicMock(return_value=iter([hits]))

        formatted = MilvusVectorStore._format_results(results)
        assert len(formatted) == 2
        assert formatted[0]["score"] == 0.9
        assert formatted[1]["chunk_id"] == "b"


# =========================================================================
# Unit tests – repr
# =========================================================================

class TestRepr:
    def test_repr(self) -> None:
        store, _ = _build_store_with_mocks()
        r = repr(store)
        assert "mock-host" in r
        assert "kubeflow_docs" in r


# =========================================================================
# Integration tests (require a live Milvus)
# =========================================================================

def pytest_addoption(parser: Any) -> None:
    """Register custom CLI options for integration tests."""
    try:
        parser.addoption("--milvus-host", default="localhost")
        parser.addoption("--milvus-port", default="19530", type=int)
    except Exception:
        pass  # Already registered


@pytest.fixture
def milvus_host(request: Any) -> str:
    return request.config.getoption("--milvus-host", default="localhost")


@pytest.fixture
def milvus_port(request: Any) -> int:
    return int(request.config.getoption("--milvus-port", default=19530))


@pytest.mark.integration
class TestIntegration:
    """Integration tests that communicate with a real Milvus server.

    Run with: ``pytest -m integration --milvus-host localhost``
    """

    def test_connect_and_stats(self, milvus_host: str, milvus_port: int) -> None:
        store = MilvusVectorStore(
            host=milvus_host,
            port=milvus_port,
            collection_name="kubeflow_docs",
        )
        stats = store.get_stats()
        assert "total_entities" in stats
        store.close()

    def test_insert_search_delete(
        self, milvus_host: str, milvus_port: int
    ) -> None:
        store = MilvusVectorStore(
            host=milvus_host,
            port=milvus_port,
            collection_name="kubeflow_docs",
        )
        dim = store.get_stats()["embedding_dim"]
        rng = np.random.default_rng(123)

        # Insert
        chunks = [
            {
                "chunk_id": f"integration_test_{i}",
                "embedding": rng.random(dim, dtype=np.float32).tolist(),
                "text": f"Integration test chunk {i}",
                "metadata": {"test": True},
            }
            for i in range(3)
        ]
        ids = store.insert(chunks)
        assert len(ids) == 3

        # Search
        query = rng.random(dim, dtype=np.float32)
        results = store.search(query, top_k=3)
        assert len(results) <= 3
        assert all("score" in r for r in results)

        # Delete
        deleted = store.delete(ids)
        assert deleted == 3

        store.close()
