"""Unit tests for the milvus-indexer component."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# Ensure the component module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from component import (
    CONNECTION_ALIAS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MILVUS_HOST,
    DEFAULT_MILVUS_PORT,
    INDEX_PARAMS,
    MAX_RETRIES,
    METRICS_FILENAME,
    IndexerMetrics,
    _build_schema,
    build_index,
    compact_collection,
    connect_to_milvus,
    disconnect_from_milvus,
    fetch_existing_ids,
    flush_collection,
    get_or_create_collection,
    insert_embeddings,
    load_embeddings,
    load_metadata,
    prepare_batch,
    rollback_inserted,
    run_indexer,
    save_metrics,
    validate_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """3 embeddings of dimension 4."""
    rng = np.random.default_rng(42)
    return rng.random((3, 4), dtype=np.float32)


@pytest.fixture
def sample_metadata() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "doc1_chunk0",
            "text": "Kubeflow is the ML toolkit for Kubernetes.",
            "source_doc": "intro.md",
            "position": 0,
        },
        {
            "chunk_id": "doc1_chunk1",
            "text": "Kubeflow Pipelines orchestrates ML workflows.",
            "source_doc": "intro.md",
            "position": 1,
        },
        {
            "chunk_id": "doc2_chunk0",
            "text": "Training operators support TensorFlow and PyTorch.",
            "source_doc": "training.md",
            "position": 0,
        },
    ]


@pytest.fixture
def embeddings_file(sample_embeddings: np.ndarray, tmp_path: Path) -> Path:
    fp = tmp_path / "embeddings.npy"
    np.save(str(fp), sample_embeddings)
    return fp


@pytest.fixture
def metadata_file(sample_metadata: List[Dict[str, Any]], tmp_path: Path) -> Path:
    fp = tmp_path / "metadata.json"
    fp.write_text(json.dumps(sample_metadata, indent=2))
    return fp


@pytest.fixture
def metadata_file_dict(sample_metadata: List[Dict[str, Any]], tmp_path: Path) -> Path:
    """Metadata wrapped in a dict with a 'metadata' key."""
    fp = tmp_path / "metadata_dict.json"
    fp.write_text(json.dumps({"metadata": sample_metadata}, indent=2))
    return fp


@pytest.fixture
def large_embeddings(tmp_path: Path) -> tuple[Path, Path, int]:
    """Generate 250 embeddings with metadata for batch tests."""
    n, dim = 250, 8
    rng = np.random.default_rng(123)
    emb = rng.random((n, dim), dtype=np.float32)
    meta = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"Sentence number {i} about ML.",
            "source_doc": f"doc_{i // 50}.md",
            "position": i,
        }
        for i in range(n)
    ]
    emb_path = tmp_path / "large_embeddings.npy"
    meta_path = tmp_path / "large_metadata.json"
    np.save(str(emb_path), emb)
    meta_path.write_text(json.dumps(meta, indent=2))
    return emb_path, meta_path, n


# ---------------------------------------------------------------------------
# Tests: load_embeddings
# ---------------------------------------------------------------------------


class TestLoadEmbeddings:
    def test_load_success(self, embeddings_file: Path, sample_embeddings: np.ndarray):
        result = load_embeddings(str(embeddings_file))
        np.testing.assert_array_almost_equal(result, sample_embeddings)
        assert result.dtype == np.float32

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_embeddings(str(tmp_path / "missing.npy"))

    def test_invalid_dimensions(self, tmp_path: Path):
        one_d = np.array([1.0, 2.0, 3.0])
        fp = tmp_path / "1d.npy"
        np.save(str(fp), one_d)
        with pytest.raises(ValueError, match="2-D"):
            load_embeddings(str(fp))


# ---------------------------------------------------------------------------
# Tests: load_metadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_load_list_format(self, metadata_file: Path, sample_metadata):
        result = load_metadata(str(metadata_file))
        assert len(result) == len(sample_metadata)
        assert result[0]["chunk_id"] == "doc1_chunk0"

    def test_load_dict_format(self, metadata_file_dict: Path, sample_metadata):
        result = load_metadata(str(metadata_file_dict))
        assert len(result) == len(sample_metadata)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_metadata(str(tmp_path / "missing.json"))

    def test_invalid_format(self, tmp_path: Path):
        fp = tmp_path / "bad.json"
        fp.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError, match="Unexpected metadata format"):
            load_metadata(str(fp))


# ---------------------------------------------------------------------------
# Tests: validate_inputs
# ---------------------------------------------------------------------------


class TestValidateInputs:
    def test_valid(self, sample_embeddings, sample_metadata):
        validate_inputs(sample_embeddings, sample_metadata)  # should not raise

    def test_count_mismatch(self, sample_embeddings, sample_metadata):
        with pytest.raises(ValueError, match="mismatch"):
            validate_inputs(sample_embeddings, sample_metadata[:2])

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_inputs(np.empty((0, 4), dtype=np.float32), [])


# ---------------------------------------------------------------------------
# Tests: prepare_batch
# ---------------------------------------------------------------------------


class TestPrepareBatch:
    def test_basic_batch(self, sample_embeddings, sample_metadata):
        chunk_ids = [m["chunk_id"] for m in sample_metadata]
        batch = prepare_batch(chunk_ids, sample_embeddings, sample_metadata, 0, 2)
        assert len(batch) == 4  # ids, embeddings, texts, metadata
        assert len(batch[0]) == 2  # 2 items in this batch
        assert batch[0] == ["doc1_chunk0", "doc1_chunk1"]
        # embeddings should be lists of floats
        assert isinstance(batch[1][0], list)
        assert len(batch[1][0]) == 4  # dim=4

    def test_text_extraction(self, sample_embeddings, sample_metadata):
        chunk_ids = [m["chunk_id"] for m in sample_metadata]
        batch = prepare_batch(chunk_ids, sample_embeddings, sample_metadata, 0, 1)
        assert "Kubeflow" in batch[2][0]  # text field

    def test_metadata_excludes_text(self, sample_embeddings, sample_metadata):
        chunk_ids = [m["chunk_id"] for m in sample_metadata]
        batch = prepare_batch(chunk_ids, sample_embeddings, sample_metadata, 0, 1)
        meta_dict = json.loads(batch[3][0])
        assert "text" not in meta_dict
        assert "source_doc" in meta_dict


# ---------------------------------------------------------------------------
# Tests: connect / disconnect
# ---------------------------------------------------------------------------


class TestConnection:
    @patch("component.connections")
    def test_connect_success(self, mock_connections):
        connect_to_milvus("localhost", 19530)
        mock_connections.connect.assert_called_once()

    @patch("component.connections")
    def test_connect_retry_then_success(self, mock_connections):
        from pymilvus import MilvusException

        mock_connections.connect.side_effect = [
            MilvusException(message="timeout"),
            None,
        ]
        connect_to_milvus("localhost", 19530)
        assert mock_connections.connect.call_count == 2

    @patch("component.time.sleep")  # skip actual sleep
    @patch("component.connections")
    def test_connect_all_retries_fail(self, mock_connections, mock_sleep):
        from pymilvus import MilvusException

        mock_connections.connect.side_effect = MilvusException(message="down")
        with pytest.raises(ConnectionError, match="Failed to connect"):
            connect_to_milvus("localhost", 19530)
        assert mock_connections.connect.call_count == MAX_RETRIES

    @patch("component.connections")
    def test_disconnect(self, mock_connections):
        disconnect_from_milvus()
        mock_connections.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _build_schema
# ---------------------------------------------------------------------------


class TestBuildSchema:
    def test_schema_fields(self):
        schema = _build_schema(384)
        field_names = [f.name for f in schema.fields]
        assert "chunk_id" in field_names
        assert "embedding" in field_names
        assert "text" in field_names
        assert "metadata" in field_names

    def test_embedding_dimension(self):
        schema = _build_schema(768)
        emb_field = [f for f in schema.fields if f.name == "embedding"][0]
        assert emb_field.params["dim"] == 768

    def test_primary_key(self):
        schema = _build_schema(384)
        pk_field = [f for f in schema.fields if f.name == "chunk_id"][0]
        assert pk_field.is_primary is True


# ---------------------------------------------------------------------------
# Tests: get_or_create_collection
# ---------------------------------------------------------------------------


class TestGetOrCreateCollection:
    @patch("component.Collection")
    @patch("component.utility")
    def test_create_new(self, mock_utility, mock_coll_cls):
        mock_utility.has_collection.return_value = False
        mock_coll = MagicMock()
        mock_coll_cls.return_value = mock_coll

        result = get_or_create_collection("test_coll", 384)
        mock_coll_cls.assert_called_once()
        assert result is mock_coll

    @patch("component.Collection")
    @patch("component.utility")
    def test_use_existing_matching_dim(self, mock_utility, mock_coll_cls):
        mock_utility.has_collection.return_value = True

        emb_field = MagicMock()
        emb_field.name = "embedding"
        emb_field.params = {"dim": 384}

        mock_coll = MagicMock()
        mock_coll.schema.fields = [emb_field]
        mock_coll.num_entities = 100
        mock_coll_cls.return_value = mock_coll

        result = get_or_create_collection("test_coll", 384)
        assert result is mock_coll

    @patch("component.Collection")
    @patch("component.utility")
    def test_dimension_mismatch(self, mock_utility, mock_coll_cls):
        mock_utility.has_collection.return_value = True

        emb_field = MagicMock()
        emb_field.name = "embedding"
        emb_field.params = {"dim": 768}

        mock_coll = MagicMock()
        mock_coll.schema.fields = [emb_field]
        mock_coll_cls.return_value = mock_coll

        with pytest.raises(ValueError, match="dimension mismatch"):
            get_or_create_collection("test_coll", 384)


# ---------------------------------------------------------------------------
# Tests: fetch_existing_ids
# ---------------------------------------------------------------------------


class TestFetchExistingIds:
    def test_empty_collection(self):
        mock_coll = MagicMock()
        mock_coll.num_entities = 0
        result = fetch_existing_ids(mock_coll, ["id1", "id2"])
        assert result == set()

    def test_some_existing(self):
        mock_coll = MagicMock()
        mock_coll.num_entities = 10
        mock_coll.query.return_value = [{"chunk_id": "id1"}]
        result = fetch_existing_ids(mock_coll, ["id1", "id2"])
        assert result == {"id1"}
        mock_coll.load.assert_called_once()
        mock_coll.release.assert_called_once()

    def test_query_failure_returns_empty(self):
        from pymilvus import MilvusException

        mock_coll = MagicMock()
        mock_coll.num_entities = 10
        mock_coll.load.side_effect = MilvusException(message="load error")
        result = fetch_existing_ids(mock_coll, ["id1"])
        assert result == set()


# ---------------------------------------------------------------------------
# Tests: insert_embeddings
# ---------------------------------------------------------------------------


class TestInsertEmbeddings:
    def test_insert_all_new(self, sample_embeddings, sample_metadata):
        mock_coll = MagicMock()
        chunk_ids = [m["chunk_id"] for m in sample_metadata]

        inserted, skipped, elapsed = insert_embeddings(
            mock_coll, chunk_ids, sample_embeddings, sample_metadata, batch_size=2
        )
        assert inserted == 3
        assert skipped == 0
        assert elapsed > 0
        assert mock_coll.insert.call_count == 2  # ceil(3/2) = 2 batches

    def test_insert_with_existing_skipped(self, sample_embeddings, sample_metadata):
        mock_coll = MagicMock()
        chunk_ids = [m["chunk_id"] for m in sample_metadata]
        existing = {"doc1_chunk0"}

        inserted, skipped, elapsed = insert_embeddings(
            mock_coll, chunk_ids, sample_embeddings, sample_metadata,
            batch_size=10, existing_ids=existing,
        )
        assert inserted == 2
        assert skipped == 1

    def test_insert_all_existing(self, sample_embeddings, sample_metadata):
        mock_coll = MagicMock()
        chunk_ids = [m["chunk_id"] for m in sample_metadata]
        existing = set(chunk_ids)

        inserted, skipped, elapsed = insert_embeddings(
            mock_coll, chunk_ids, sample_embeddings, sample_metadata,
            existing_ids=existing,
        )
        assert inserted == 0
        assert skipped == 3
        mock_coll.insert.assert_not_called()

    @patch("component.time.sleep")
    def test_insert_retry_on_failure(self, mock_sleep, sample_embeddings, sample_metadata):
        from pymilvus import MilvusException

        mock_coll = MagicMock()
        # Fail once, then succeed
        mock_coll.insert.side_effect = [
            MilvusException(message="transient"),
            MagicMock(),
        ]
        chunk_ids = [m["chunk_id"] for m in sample_metadata]

        inserted, skipped, elapsed = insert_embeddings(
            mock_coll, chunk_ids, sample_embeddings, sample_metadata,
            batch_size=10,
        )
        assert inserted == 3
        assert mock_coll.insert.call_count == 2

    @patch("component.time.sleep")
    def test_insert_all_retries_fail(self, mock_sleep, sample_embeddings, sample_metadata):
        from pymilvus import MilvusException

        mock_coll = MagicMock()
        mock_coll.insert.side_effect = MilvusException(message="persistent error")
        chunk_ids = [m["chunk_id"] for m in sample_metadata]

        with pytest.raises(RuntimeError, match="Failed to insert batch"):
            insert_embeddings(
                mock_coll, chunk_ids, sample_embeddings, sample_metadata,
                batch_size=10,
            )

    def test_large_batch_insertion(self, large_embeddings):
        emb_path, meta_path, n = large_embeddings
        embeddings = np.load(str(emb_path))
        metadata = load_metadata(str(meta_path))
        chunk_ids = [m["chunk_id"] for m in metadata]

        mock_coll = MagicMock()
        inserted, skipped, elapsed = insert_embeddings(
            mock_coll, chunk_ids, embeddings, metadata, batch_size=50,
        )
        assert inserted == n
        assert mock_coll.insert.call_count == 5  # 250 / 50


# ---------------------------------------------------------------------------
# Tests: flush / compact / build_index
# ---------------------------------------------------------------------------


class TestPostInsertOps:
    def test_flush(self):
        mock_coll = MagicMock()
        elapsed = flush_collection(mock_coll)
        mock_coll.flush.assert_called_once()
        assert elapsed >= 0

    def test_compact(self):
        mock_coll = MagicMock()
        elapsed = compact_collection(mock_coll)
        mock_coll.compact.assert_called_once()
        mock_coll.wait_for_compaction_completed.assert_called_once()

    def test_build_index_no_existing(self):
        mock_coll = MagicMock()
        mock_coll.indexes = []
        elapsed = build_index(mock_coll, force_rebuild=False)
        mock_coll.create_index.assert_called_once_with(
            field_name="embedding", index_params=INDEX_PARAMS,
        )
        assert elapsed >= 0

    def test_skip_existing_index(self):
        mock_idx = MagicMock()
        mock_idx.field_name = "embedding"
        mock_coll = MagicMock()
        mock_coll.indexes = [mock_idx]
        elapsed = build_index(mock_coll, force_rebuild=False)
        mock_coll.create_index.assert_not_called()
        assert elapsed == 0.0

    def test_force_rebuild_index(self):
        mock_idx = MagicMock()
        mock_idx.field_name = "embedding"
        mock_coll = MagicMock()
        mock_coll.indexes = [mock_idx]
        elapsed = build_index(mock_coll, force_rebuild=True)
        mock_coll.drop_index.assert_called_once()
        mock_coll.create_index.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: save_metrics
# ---------------------------------------------------------------------------


class TestSaveMetrics:
    def test_save_and_load(self, tmp_path: Path):
        metrics = IndexerMetrics(
            total_vectors_inserted=100,
            insert_rate_vectors_per_sec=500.0,
            embedding_dimension=384,
        )
        path = save_metrics(metrics, str(tmp_path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_vectors_inserted"] == 100
        assert data["insert_rate_vectors_per_sec"] == 500.0

    def test_creates_output_dir(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        metrics = IndexerMetrics()
        path = save_metrics(metrics, str(nested))
        assert path.exists()


# ---------------------------------------------------------------------------
# Tests: rollback_inserted
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_deletes(self):
        mock_coll = MagicMock()
        rollback_inserted(mock_coll, ["id1", "id2"])
        mock_coll.delete.assert_called_once()
        mock_coll.flush.assert_called_once()

    def test_rollback_empty_noop(self):
        mock_coll = MagicMock()
        rollback_inserted(mock_coll, [])
        mock_coll.delete.assert_not_called()

    def test_rollback_failure_logged(self):
        mock_coll = MagicMock()
        mock_coll.delete.side_effect = Exception("delete failed")
        # Should not raise
        rollback_inserted(mock_coll, ["id1"])


# ---------------------------------------------------------------------------
# Tests: run_indexer (integration-level with mocks)
# ---------------------------------------------------------------------------


class TestRunIndexer:
    @patch("component.disconnect_from_milvus")
    @patch("component.build_index", return_value=1.0)
    @patch("component.compact_collection", return_value=0.5)
    @patch("component.flush_collection", return_value=0.3)
    @patch("component.insert_embeddings", return_value=(3, 0, 0.1))
    @patch("component.fetch_existing_ids", return_value=set())
    @patch("component.get_or_create_collection")
    @patch("component.connect_to_milvus")
    def test_full_pipeline(
        self,
        mock_connect,
        mock_get_coll,
        mock_fetch_ids,
        mock_insert,
        mock_flush,
        mock_compact,
        mock_build_idx,
        mock_disconnect,
        embeddings_file,
        metadata_file,
        tmp_path,
    ):
        mock_coll = MagicMock()
        mock_coll.num_entities = 3
        mock_get_coll.return_value = mock_coll

        metrics = run_indexer(
            embeddings_path=str(embeddings_file),
            metadata_path=str(metadata_file),
            milvus_host="localhost",
            milvus_port=19530,
            collection_name="test",
            batch_size=10,
            rebuild_index=False,
            output_dir=str(tmp_path / "out"),
        )

        mock_connect.assert_called_once()
        mock_get_coll.assert_called_once()
        mock_insert.assert_called_once()
        mock_flush.assert_called()
        mock_compact.assert_called_once()
        mock_build_idx.assert_called_once()
        mock_disconnect.assert_called_once()

        assert metrics.total_vectors_inserted == 3
        assert metrics.embedding_dimension == 4  # from sample embeddings

        # Verify metrics file was written
        metrics_path = tmp_path / "out" / METRICS_FILENAME
        assert metrics_path.exists()

    @patch("component.disconnect_from_milvus")
    @patch("component.connect_to_milvus")
    def test_rollback_on_failure(
        self,
        mock_connect,
        mock_disconnect,
        embeddings_file,
        metadata_file,
        tmp_path,
    ):
        """Verify rollback is triggered when insertion fails."""
        with patch("component.get_or_create_collection") as mock_get_coll, \
             patch("component.fetch_existing_ids", return_value=set()), \
             patch("component.insert_embeddings") as mock_insert, \
             patch("component.rollback_inserted") as mock_rollback:

            mock_coll = MagicMock()
            mock_coll.num_entities = 0
            mock_get_coll.return_value = mock_coll
            mock_insert.side_effect = RuntimeError("insert failed")

            with pytest.raises(RuntimeError, match="insert failed"):
                run_indexer(
                    embeddings_path=str(embeddings_file),
                    metadata_path=str(metadata_file),
                    milvus_host="localhost",
                    output_dir=str(tmp_path / "out"),
                )

            mock_disconnect.assert_called_once()

    @patch("component.disconnect_from_milvus")
    @patch("component.connect_to_milvus")
    def test_connection_failure(
        self,
        mock_connect,
        mock_disconnect,
        embeddings_file,
        metadata_file,
        tmp_path,
    ):
        mock_connect.side_effect = ConnectionError("unreachable")
        with pytest.raises(ConnectionError):
            run_indexer(
                embeddings_path=str(embeddings_file),
                metadata_path=str(metadata_file),
                output_dir=str(tmp_path / "out"),
            )


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_args(self):
        from component import parse_args

        with patch("sys.argv", [
            "component.py",
            "--embeddings-path", "/data/emb.npy",
            "--metadata-path", "/data/meta.json",
        ]):
            args = parse_args()
            assert args.embeddings_path == "/data/emb.npy"
            assert args.metadata_path == "/data/meta.json"
            assert args.milvus_host == DEFAULT_MILVUS_HOST
            assert args.milvus_port == DEFAULT_MILVUS_PORT
            assert args.collection_name == DEFAULT_COLLECTION_NAME
            assert args.batch_size == DEFAULT_BATCH_SIZE
            assert args.rebuild_index == "false"

    def test_all_args(self):
        from component import parse_args

        with patch("sys.argv", [
            "component.py",
            "--embeddings-path", "/data/emb.npy",
            "--metadata-path", "/data/meta.json",
            "--milvus-host", "my-host",
            "--milvus-port", "19531",
            "--collection-name", "my_coll",
            "--batch-size", "200",
            "--rebuild-index", "true",
            "--output-dir", "/custom/out",
        ]):
            args = parse_args()
            assert args.milvus_host == "my-host"
            assert args.milvus_port == 19531
            assert args.collection_name == "my_coll"
            assert args.batch_size == 200
            assert args.rebuild_index == "true"
            assert args.output_dir == "/custom/out"
