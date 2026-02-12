"""Unit tests for the embedding-generator component."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the component module is importable
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)

from component import (
    CheckpointManager,
    EmbeddingMetrics,
    EmbeddingRecord,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    EMBEDDINGS_FILENAME,
    EMBEDDINGS_JSON_FILENAME,
    METADATA_FILENAME,
    METRICS_FILENAME,
    generate_embeddings,
    load_chunks,
    resolve_device,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Minimal chunks matching the document-chunker output schema."""
    return [
        {
            "chunk_id": "doc1_chunk0",
            "text": "Kubeflow is the machine learning toolkit for Kubernetes.",
            "tokens": 10,
            "metadata": {
                "source_doc": "website/docs/intro.md",
                "position": 0,
                "parent_title": "Introduction",
                "section": "docs",
            },
        },
        {
            "chunk_id": "doc1_chunk1",
            "text": "Kubeflow Pipelines is a platform for building ML workflows.",
            "tokens": 11,
            "metadata": {
                "source_doc": "website/docs/intro.md",
                "position": 1,
                "parent_title": "Pipelines",
                "section": "docs",
            },
        },
        {
            "chunk_id": "doc2_chunk0",
            "text": "Training operators support TensorFlow, PyTorch, and MXNet.",
            "tokens": 9,
            "metadata": {
                "source_doc": "website/docs/training.md",
                "position": 0,
                "parent_title": "Training",
                "section": "docs",
            },
        },
    ]


@pytest.fixture
def chunks_file(sample_chunks: List[Dict[str, Any]], tmp_path: Path) -> Path:
    """Write sample chunks to a JSON file and return the path."""
    fp = tmp_path / "chunks.json"
    fp.write_text(json.dumps({"chunks": sample_chunks}, indent=2))
    return fp


@pytest.fixture
def chunks_file_list_format(sample_chunks: List[Dict[str, Any]], tmp_path: Path) -> Path:
    """Write chunks as a bare JSON list (alternative format)."""
    fp = tmp_path / "chunks_list.json"
    fp.write_text(json.dumps(sample_chunks, indent=2))
    return fp


@pytest.fixture
def large_chunks() -> List[Dict[str, Any]]:
    """Generate a larger set of chunks for batch tests."""
    return [
        {
            "chunk_id": f"doc_{i}_chunk_{j}",
            "text": f"This is sentence number {i * 10 + j} about machine learning topic {j}.",
            "tokens": 12,
            "metadata": {
                "source_doc": f"docs/file_{i}.md",
                "position": j,
            },
        }
        for i in range(10)
        for j in range(5)
    ]


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384

    def _encode(texts, **kwargs):
        return np.random.default_rng(42).standard_normal(
            (len(texts), 384)
        ).astype(np.float32)

    model.encode.side_effect = _encode
    return model


# ---------------------------------------------------------------------------
# Tests: load_chunks
# ---------------------------------------------------------------------------

class TestLoadChunks:
    """Tests for the chunk-loading function."""

    def test_load_dict_format(self, chunks_file: Path) -> None:
        chunks = load_chunks(str(chunks_file))
        assert len(chunks) == 3
        assert chunks[0]["chunk_id"] == "doc1_chunk0"

    def test_load_list_format(self, chunks_file_list_format: Path) -> None:
        chunks = load_chunks(str(chunks_file_list_format))
        assert len(chunks) == 3

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_chunks(str(tmp_path / "nonexistent.json"))

    def test_invalid_format(self, tmp_path: Path) -> None:
        fp = tmp_path / "bad.json"
        fp.write_text('"just a string"')
        with pytest.raises(ValueError, match="Unexpected chunks format"):
            load_chunks(str(fp))


# ---------------------------------------------------------------------------
# Tests: resolve_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    """Tests for device resolution."""

    def test_cpu_requested(self) -> None:
        assert resolve_device("cpu") == "cpu"

    @patch("component.torch")
    def test_cuda_available(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla T4"
        assert resolve_device("cuda") == "cuda"

    @patch("component.torch")
    def test_cuda_fallback_to_cpu(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = False
        assert resolve_device("cuda") == "cpu"


# ---------------------------------------------------------------------------
# Tests: CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    """Tests for checkpoint save/load/remove."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path)
        emb = np.random.randn(5, 384).astype(np.float32)
        ids = ["id_0", "id_1", "id_2", "id_3", "id_4"]

        mgr.save(emb, ids)
        assert mgr.exists

        loaded_emb, loaded_ids = mgr.load()
        np.testing.assert_array_almost_equal(emb, loaded_emb)
        assert loaded_ids == ids

    def test_remove(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path)
        mgr.save(np.zeros((1, 384)), ["id_0"])
        assert mgr.exists

        mgr.remove()
        assert not mgr.exists

    def test_not_exists_initially(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path)
        assert not mgr.exists


# ---------------------------------------------------------------------------
# Tests: generate_embeddings
# ---------------------------------------------------------------------------

class TestGenerateEmbeddings:
    """Tests for the core embedding generation logic."""

    def test_basic_generation(
        self,
        sample_chunks: List[Dict[str, Any]],
        mock_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        emb, meta, metrics = generate_embeddings(
            chunks=sample_chunks,
            model=mock_model,
            model_name=DEFAULT_MODEL_NAME,
            batch_size=2,
            device="cpu",
            output_dir=str(tmp_path),
        )

        assert emb.shape == (3, 384)
        assert len(meta) == 3
        assert metrics.total_embeddings == 3
        assert metrics.embedding_dimensions == 384

    def test_empty_chunks(self, mock_model: MagicMock, tmp_path: Path) -> None:
        emb, meta, metrics = generate_embeddings(
            chunks=[],
            model=mock_model,
            batch_size=32,
            device="cpu",
            output_dir=str(tmp_path),
        )

        assert emb.shape == (0, 384)
        assert len(meta) == 0
        assert metrics.total_embeddings == 0

    def test_batch_processing(
        self,
        large_chunks: List[Dict[str, Any]],
        mock_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        emb, meta, metrics = generate_embeddings(
            chunks=large_chunks,
            model=mock_model,
            batch_size=8,
            device="cpu",
            output_dir=str(tmp_path),
        )

        assert emb.shape == (50, 384)
        assert metrics.total_embeddings == 50
        assert metrics.batch_size == 8

    def test_metrics_recorded(
        self,
        sample_chunks: List[Dict[str, Any]],
        mock_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        _, _, metrics = generate_embeddings(
            chunks=sample_chunks,
            model=mock_model,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
            output_dir=str(tmp_path),
        )

        assert metrics.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert metrics.device == "cpu"
        assert metrics.total_time_seconds >= 0
        assert metrics.embeddings_per_second >= 0

    def test_checkpoint_resume(
        self,
        sample_chunks: List[Dict[str, Any]],
        mock_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Simulate a checkpoint from a prior partial run."""
        mgr = CheckpointManager(tmp_path)
        prior_emb = np.random.randn(1, 384).astype(np.float32)
        mgr.save(prior_emb, ["doc1_chunk0"])

        emb, meta, metrics = generate_embeddings(
            chunks=sample_chunks,
            model=mock_model,
            batch_size=32,
            device="cpu",
            output_dir=str(tmp_path),
        )

        # Should include the 1 checkpointed + 2 newly generated
        assert emb.shape == (3, 384)
        assert metrics.resumed_from_checkpoint is True
        assert metrics.checkpoint_chunks_skipped == 1

    def test_metadata_contains_model_short_name(
        self,
        sample_chunks: List[Dict[str, Any]],
        mock_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        _, meta, _ = generate_embeddings(
            chunks=sample_chunks,
            model=mock_model,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
            output_dir=str(tmp_path),
        )

        for rec in meta:
            assert rec["model"] == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Tests: save_outputs
# ---------------------------------------------------------------------------

class TestSaveOutputs:
    """Tests for the output serialisation."""

    def test_all_files_written(self, tmp_path: Path) -> None:
        emb = np.random.randn(3, 384).astype(np.float32)
        meta = [
            {
                "chunk_id": f"c{i}",
                "model": "all-MiniLM-L6-v2",
                "text": f"text {i}",
                "metadata": {},
                "embedding_index": i,
            }
            for i in range(3)
        ]
        metrics = EmbeddingMetrics(
            total_embeddings=3,
            embedding_dimensions=384,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        paths = save_outputs(emb, meta, metrics, str(tmp_path))

        assert (tmp_path / EMBEDDINGS_FILENAME).exists()
        assert (tmp_path / METADATA_FILENAME).exists()
        assert (tmp_path / METRICS_FILENAME).exists()
        assert (tmp_path / EMBEDDINGS_JSON_FILENAME).exists()

    def test_npy_shape_and_dtype(self, tmp_path: Path) -> None:
        emb = np.random.randn(5, 384).astype(np.float64)
        meta = [
            {"chunk_id": f"c{i}", "model": "m", "text": "", "metadata": {}, "embedding_index": i}
            for i in range(5)
        ]
        metrics = EmbeddingMetrics(total_embeddings=5, embedding_dimensions=384)

        save_outputs(emb, meta, metrics, str(tmp_path))

        loaded = np.load(tmp_path / EMBEDDINGS_FILENAME)
        assert loaded.shape == (5, 384)
        assert loaded.dtype == np.float32

    def test_combined_json_structure(self, tmp_path: Path) -> None:
        emb = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        meta = [
            {
                "chunk_id": "c0",
                "model": "all-MiniLM-L6-v2",
                "text": "hello",
                "metadata": {"section": "docs"},
                "embedding_index": 0,
            }
        ]
        metrics = EmbeddingMetrics(total_embeddings=1, embedding_dimensions=3)

        save_outputs(emb, meta, metrics, str(tmp_path))

        data = json.loads((tmp_path / EMBEDDINGS_JSON_FILENAME).read_text())
        assert "embeddings" in data
        assert "metrics" in data
        assert len(data["embeddings"]) == 1

        rec = data["embeddings"][0]
        assert rec["chunk_id"] == "c0"
        assert rec["model"] == "all-MiniLM-L6-v2"
        assert rec["text"] == "hello"
        assert len(rec["embedding"]) == 3
        assert rec["metadata"]["section"] == "docs"

    def test_skip_combined_json(self, tmp_path: Path) -> None:
        emb = np.random.randn(2, 384).astype(np.float32)
        meta = [
            {"chunk_id": f"c{i}", "model": "m", "text": "", "metadata": {}, "embedding_index": i}
            for i in range(2)
        ]
        metrics = EmbeddingMetrics(total_embeddings=2, embedding_dimensions=384)

        paths = save_outputs(emb, meta, metrics, str(tmp_path), save_combined_json=False)

        assert (tmp_path / EMBEDDINGS_FILENAME).exists()
        assert (tmp_path / METADATA_FILENAME).exists()
        assert not (tmp_path / EMBEDDINGS_JSON_FILENAME).exists()
        assert "embeddings_json" not in paths

    def test_metrics_json_content(self, tmp_path: Path) -> None:
        emb = np.random.randn(1, 384).astype(np.float32)
        meta = [{"chunk_id": "c0", "model": "m", "text": "", "metadata": {}, "embedding_index": 0}]
        metrics = EmbeddingMetrics(
            total_embeddings=1,
            embedding_dimensions=384,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            batch_size=32,
            embeddings_per_second=100.0,
            total_time_seconds=0.01,
        )

        save_outputs(emb, meta, metrics, str(tmp_path))

        data = json.loads((tmp_path / METRICS_FILENAME).read_text())
        assert data["total_embeddings"] == 1
        assert data["embedding_dimensions"] == 384
        assert data["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert data["embeddings_per_second"] == 100.0


# ---------------------------------------------------------------------------
# Tests: EmbeddingRecord dataclass
# ---------------------------------------------------------------------------

class TestEmbeddingRecord:
    """Basic tests for the EmbeddingRecord dataclass."""

    def test_defaults(self) -> None:
        rec = EmbeddingRecord(
            chunk_id="c0",
            embedding=[0.1, 0.2],
            model="all-MiniLM-L6-v2",
            text="hello",
        )
        assert rec.chunk_id == "c0"
        assert rec.metadata == {}

    def test_with_metadata(self) -> None:
        rec = EmbeddingRecord(
            chunk_id="c1",
            embedding=[0.1],
            model="m",
            text="world",
            metadata={"section": "docs"},
        )
        assert rec.metadata["section"] == "docs"
