"""Unit tests for the document-chunker component."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure the component module is importable
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)

from component import (
    Chunk,
    ChunkMetadata,
    ChunkingMetrics,
    TokenCounter,
    _find_span,
    _fixed_size_chunks,
    _semantic_chunks,
    _sentence_chunks,
    chunk_document,
    chunk_documents,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    VALID_STRATEGIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def token_counter() -> TokenCounter:
    return TokenCounter()


@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """A minimal document matching the crawler output schema."""
    return {
        "id": "doc1",
        "source": "website/docs/intro.md",
        "title": "Introduction to Kubeflow",
        "content": (
            "# Introduction to Kubeflow\n\n"
            "Kubeflow is the machine learning toolkit for Kubernetes. "
            "It provides a straightforward way to deploy ML workflows "
            "on Kubernetes.\n\n"
            "## Getting Started\n\n"
            "To get started with Kubeflow, you need a Kubernetes cluster. "
            "You can use a local cluster for development or a cloud-based "
            "cluster for production workloads.\n\n"
            "### Prerequisites\n\n"
            "Before installing Kubeflow, make sure you have the following:\n"
            "- kubectl installed and configured\n"
            "- A Kubernetes cluster (v1.25+)\n"
            "- kustomize v5.0+\n\n"
            "## Pipelines\n\n"
            "Kubeflow Pipelines is a platform for building and deploying "
            "portable, scalable machine learning workflows based on Docker "
            "containers. It provides a way to orchestrate complex ML "
            "pipelines with ease.\n\n"
            "## Training\n\n"
            "Kubeflow provides several operators for distributed training, "
            "including TFJob for TensorFlow, PyTorchJob for PyTorch, and "
            "MXJob for Apache MXNet."
        ),
        "url": "https://www.kubeflow.org/docs/started/introduction/",
        "metadata": {
            "section": "docs",
            "version": "1.8",
        },
    }


@pytest.fixture
def long_document() -> Dict[str, Any]:
    """A document with enough content to produce multiple chunks."""
    paragraph = (
        "Machine learning is a branch of artificial intelligence that "
        "focuses on building systems that learn from data. These systems "
        "improve their performance on a specific task over time without "
        "being explicitly programmed. Modern ML encompasses supervised, "
        "unsupervised, and reinforcement learning paradigms. "
    )
    # Repeat to ensure we exceed 512 tokens
    content = "# Machine Learning Overview\n\n" + (paragraph * 30)
    return {
        "id": "longdoc",
        "source": "website/docs/ml-overview.md",
        "title": "Machine Learning Overview",
        "content": content,
        "url": "https://example.com/docs/ml-overview",
        "metadata": {"section": "general", "version": "latest"},
    }


@pytest.fixture
def empty_document() -> Dict[str, Any]:
    return {
        "id": "emptydoc",
        "content": "",
        "title": "Empty",
        "metadata": {},
    }


@pytest.fixture
def documents_file(sample_document, long_document, tmp_path) -> Path:
    """Write sample documents to a temp JSON file."""
    path = tmp_path / "documents.json"
    payload = {"documents": [sample_document, long_document]}
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# TokenCounter tests
# ---------------------------------------------------------------------------

class TestTokenCounter:
    def test_count_nonempty(self, token_counter):
        count = token_counter.count("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_count_empty(self, token_counter):
        assert token_counter.count("") == 0

    def test_length_function_callable(self, token_counter):
        fn = token_counter.token_length_function()
        assert callable(fn)
        assert fn("test") > 0

    def test_length_function_matches_count(self, token_counter):
        text = "Kubeflow pipelines are great for ML workflows."
        assert token_counter.count(text) == token_counter.token_length_function()(text)


# ---------------------------------------------------------------------------
# _find_span tests
# ---------------------------------------------------------------------------

class TestFindSpan:
    def test_exact_match(self):
        haystack = "The quick brown fox jumps over the lazy dog."
        start, end = _find_span(haystack, "brown fox")
        assert start == 10
        assert end == 19

    def test_not_found(self):
        start, end = _find_span("hello world", "foobar")
        assert start == -1
        assert end == -1

    def test_search_start_offset(self):
        haystack = "abc abc abc"
        s1, _ = _find_span(haystack, "abc", search_start=0)
        s2, _ = _find_span(haystack, "abc", search_start=1)
        assert s1 == 0
        assert s2 == 4


# ---------------------------------------------------------------------------
# Fixed-size chunking tests
# ---------------------------------------------------------------------------

class TestFixedSizeChunking:
    def test_short_text_single_chunk(self, token_counter):
        text = "Short text."
        chunks = _fixed_size_chunks(text, 512, 128, token_counter)
        assert len(chunks) == 1
        assert chunks[0].strip() == text

    def test_long_text_multiple_chunks(self, token_counter):
        text = "Word " * 2000  # many tokens
        chunks = _fixed_size_chunks(text, 100, 20, token_counter)
        assert len(chunks) > 1

    def test_all_content_preserved(self, token_counter):
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)
        chunks = _fixed_size_chunks(text, 50, 10, token_counter)
        # Every original word should appear in at least one chunk
        joined = " ".join(chunks)
        for w in words[:20]:  # spot-check first 20
            assert w in joined


# ---------------------------------------------------------------------------
# Semantic chunking tests
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    def test_splits_on_headers(self, token_counter):
        text = (
            "# Title\n\nIntro paragraph.\n\n"
            "## Section A\n\nContent A.\n\n"
            "## Section B\n\nContent B.\n"
        )
        chunks = _semantic_chunks(text, 512, 128, token_counter)
        assert len(chunks) >= 2

    def test_falls_back_without_headers(self, token_counter):
        text = "No headers here. " * 200
        chunks = _semantic_chunks(text, 50, 10, token_counter)
        assert len(chunks) > 1  # should fall back to fixed-size

    def test_large_section_is_subsplit(self, token_counter):
        text = "# Big Section\n\n" + ("This is filler content. " * 500)
        chunks = _semantic_chunks(text, 100, 20, token_counter)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Sentence chunking tests
# ---------------------------------------------------------------------------

class TestSentenceChunking:
    def test_respects_sentence_boundaries(self, token_counter):
        text = "First sentence. Second sentence. Third sentence."
        chunks = _sentence_chunks(text, 512, 0, token_counter)
        # Should stay as one chunk (small text)
        assert len(chunks) == 1

    def test_splits_long_text(self, token_counter):
        sentences = [f"Sentence number {i} has some words in it." for i in range(100)]
        text = " ".join(sentences)
        chunks = _sentence_chunks(text, 50, 10, token_counter)
        assert len(chunks) > 1

    def test_overlap_present(self, token_counter):
        sentences = [f"Sentence {i} is unique." for i in range(50)]
        text = " ".join(sentences)
        chunks = _sentence_chunks(text, 30, 10, token_counter)
        if len(chunks) >= 2:
            # Last words of chunk N should appear at start of chunk N+1
            tail_words = set(chunks[0].split()[-3:])
            head_words = set(chunks[1].split()[:10])
            assert tail_words & head_words, "Expected overlap between chunks"


# ---------------------------------------------------------------------------
# chunk_document tests
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_basic_chunking(self, sample_document, token_counter):
        chunks = chunk_document(
            sample_document,
            chunk_size=512,
            overlap=128,
            strategy="semantic",
            token_counter=token_counter,
        )
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id.startswith("doc1_chunk")
            assert c.tokens > 0
            assert c.metadata.source_doc == "doc1"

    def test_empty_document_skipped(self, empty_document, token_counter):
        chunks = chunk_document(
            empty_document,
            chunk_size=512,
            overlap=128,
            strategy="fixed",
            token_counter=token_counter,
        )
        assert chunks == []

    def test_all_strategies(self, sample_document, token_counter):
        for strategy in VALID_STRATEGIES:
            chunks = chunk_document(
                sample_document,
                chunk_size=256,
                overlap=64,
                strategy=strategy,
                token_counter=token_counter,
            )
            assert len(chunks) >= 1, f"Strategy '{strategy}' produced no chunks"

    def test_invalid_strategy_raises(self, sample_document, token_counter):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_document(
                sample_document,
                chunk_size=512,
                overlap=128,
                strategy="nonexistent",
                token_counter=token_counter,
            )

    def test_metadata_preserved(self, sample_document, token_counter):
        chunks = chunk_document(
            sample_document,
            chunk_size=512,
            overlap=128,
            strategy="fixed",
            token_counter=token_counter,
        )
        for c in chunks:
            assert c.metadata.parent_title == "Introduction to Kubeflow"
            assert c.metadata.section == "docs"
            assert "version" in c.metadata.extra

    def test_chunk_ids_unique(self, long_document, token_counter):
        chunks = chunk_document(
            long_document,
            chunk_size=100,
            overlap=20,
            strategy="fixed",
            token_counter=token_counter,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_position_monotonic(self, long_document, token_counter):
        chunks = chunk_document(
            long_document,
            chunk_size=100,
            overlap=20,
            strategy="fixed",
            token_counter=token_counter,
        )
        positions = [c.metadata.position for c in chunks]
        assert positions == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# chunk_documents (full pipeline) tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_end_to_end(self, documents_file, tmp_path):
        out = tmp_path / "chunks.json"
        result_path = chunk_documents(
            documents_path=str(documents_file),
            output_path=str(out),
            chunk_size=256,
            overlap=64,
            strategy="fixed",
        )
        assert Path(result_path).is_file()
        data = json.loads(out.read_text())
        assert "chunks" in data
        assert "metrics" in data
        assert len(data["chunks"]) > 0
        # Validate chunk schema
        for chunk in data["chunks"]:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "tokens" in chunk
            assert "metadata" in chunk
            meta = chunk["metadata"]
            assert "source_doc" in meta
            assert "position" in meta
            assert "parent_title" in meta
            assert "section" in meta

    def test_metrics_written(self, documents_file, tmp_path):
        out = tmp_path / "chunks.json"
        chunk_documents(
            documents_path=str(documents_file),
            output_path=str(out),
            chunk_size=256,
            overlap=64,
            strategy="semantic",
        )
        metrics_file = tmp_path / "metrics.json"
        assert metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        assert metrics["total_chunks"] > 0
        assert metrics["total_tokens"] > 0
        assert metrics["average_chunk_size"] > 0
        assert metrics["documents_processed"] == 2
        assert metrics["strategy"] == "semantic"
        assert metrics["elapsed_seconds"] >= 0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            chunk_documents(
                documents_path=str(tmp_path / "nonexistent.json"),
                output_path=str(tmp_path / "out.json"),
            )

    def test_invalid_strategy_rejected(self, documents_file, tmp_path):
        with pytest.raises(ValueError, match="Invalid strategy"):
            chunk_documents(
                documents_path=str(documents_file),
                output_path=str(tmp_path / "out.json"),
                strategy="unknown",
            )

    def test_overlap_exceeds_chunk_size_rejected(self, documents_file, tmp_path):
        with pytest.raises(ValueError, match="overlap must be less"):
            chunk_documents(
                documents_path=str(documents_file),
                output_path=str(tmp_path / "out.json"),
                chunk_size=100,
                overlap=200,
            )

    def test_all_strategies_produce_output(self, documents_file, tmp_path):
        for strategy in VALID_STRATEGIES:
            out = tmp_path / f"chunks_{strategy}.json"
            chunk_documents(
                documents_path=str(documents_file),
                output_path=str(out),
                chunk_size=256,
                overlap=64,
                strategy=strategy,
            )
            data = json.loads(out.read_text())
            assert len(data["chunks"]) > 0, (
                f"Strategy '{strategy}' produced no chunks"
            )

    def test_empty_documents_list(self, tmp_path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(json.dumps({"documents": []}))
        out = tmp_path / "out.json"
        chunk_documents(
            documents_path=str(empty_file),
            output_path=str(out),
        )
        data = json.loads(out.read_text())
        assert data["chunks"] == []
        assert data["metrics"]["total_chunks"] == 0
