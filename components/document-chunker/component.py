"""Document Chunker - KFP Component.

Loads documents produced by the crawler component and splits them into
overlapping, token-counted chunks using one of several strategies:
  • fixed   – fixed-size windows with token overlap
  • semantic – split on markdown headers / sections
  • sentence – split on sentence boundaries

Outputs a JSON artifact of chunks with metadata and pipeline metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("document-chunker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 512       # tokens
DEFAULT_OVERLAP = 128          # tokens
DEFAULT_STRATEGY = "semantic"
DEFAULT_ENCODING = "cl100k_base"  # GPT-4 / text-embedding-ada-002

VALID_STRATEGIES = ("fixed", "semantic", "sentence")

# Sentence-ending regex used by the sentence strategy
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Markdown header hierarchy for the semantic splitter
_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ChunkMetadata:
    """Metadata attached to every chunk."""

    source_doc: str
    position: int  # chunk index within the document
    start_char: int
    end_char: int
    parent_title: str = ""
    section: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A single text chunk."""

    chunk_id: str
    text: str
    tokens: int
    metadata: ChunkMetadata


@dataclass
class ChunkingMetrics:
    """Aggregate metrics for the chunking run."""

    total_chunks: int = 0
    average_chunk_size: float = 0.0
    total_tokens: int = 0
    documents_processed: int = 0
    strategy: str = ""
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
class TokenCounter:
    """Thin wrapper around *tiktoken* for token counting and length fn."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self.encoding.encode(text))

    def token_length_function(self):
        """Return a callable suitable for langchain's length_function."""
        return lambda text: len(self.encoding.encode(text))


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _fixed_size_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    token_counter: TokenCounter,
) -> List[str]:
    """Split *text* into fixed-size token windows with overlap.

    Uses ``RecursiveCharacterTextSplitter`` configured with a tiktoken
    length function so that splits respect token boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=token_counter.token_length_function(),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _semantic_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    token_counter: TokenCounter,
) -> List[str]:
    """Split on markdown headers first, then sub-split large sections.

    1. ``MarkdownHeaderTextSplitter`` splits by ``#``-level headings.
    2. Any section that exceeds *chunk_size* tokens is further split with
       ``RecursiveCharacterTextSplitter``.
    3. Short consecutive sections are **not** merged to preserve semantic
       boundaries.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    md_docs = md_splitter.split_text(text)

    length_fn = token_counter.token_length_function()
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[str] = []
    for doc in md_docs:
        section_text = doc.page_content
        # Prepend header metadata as context for the chunk
        header_parts = []
        for level in ("h1", "h2", "h3", "h4"):
            if level in doc.metadata:
                header_parts.append(doc.metadata[level])
        if header_parts:
            section_text = " > ".join(header_parts) + "\n\n" + section_text

        if length_fn(section_text) > chunk_size:
            chunks.extend(sub_splitter.split_text(section_text))
        else:
            chunks.append(section_text)

    # Fallback: if the markdown splitter produced nothing (e.g. no headers),
    # fall back to fixed-size chunking.
    if not chunks:
        return _fixed_size_chunks(text, chunk_size, overlap, token_counter)

    return chunks


def _sentence_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    token_counter: TokenCounter,
) -> List[str]:
    """Split on sentence boundaries then group sentences into chunks.

    Sentences are detected with a regex on ``[.!?]`` followed by whitespace.
    Groups of sentences are accumulated until *chunk_size* tokens is reached,
    then a new chunk is started with *overlap* tokens of carry-over.
    """
    sentences = _SENTENCE_RE.split(text)
    if not sentences:
        return [text] if text.strip() else []

    length_fn = token_counter.token_length_function()
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = length_fn(sentence)
        if current_tokens + sent_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Build overlap from the tail of the current chunk
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tok = length_fn(s)
                if overlap_tokens + s_tok > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tok
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


_STRATEGY_MAP = {
    "fixed": _fixed_size_chunks,
    "semantic": _semantic_chunks,
    "sentence": _sentence_chunks,
}


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _find_span(haystack: str, needle: str, search_start: int = 0) -> tuple[int, int]:
    """Return (start, end) char offsets of *needle* inside *haystack*.

    Falls back to (-1, -1) if the chunk isn't found verbatim (can happen
    when splitters normalise whitespace).
    """
    idx = haystack.find(needle, search_start)
    if idx == -1:
        # Try a whitespace-normalised search
        needle_norm = " ".join(needle.split())
        haystack_norm = " ".join(haystack.split())
        idx = haystack_norm.find(needle_norm)
        if idx == -1:
            return -1, -1
        return idx, idx + len(needle_norm)
    return idx, idx + len(needle)


def chunk_document(
    doc: Dict[str, Any],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = DEFAULT_STRATEGY,
    token_counter: TokenCounter,
) -> List[Chunk]:
    """Split a single document dict into ``Chunk`` objects."""
    doc_id: str = doc.get("id", "unknown")
    content: str = doc.get("content", "")
    title: str = doc.get("title", "")
    doc_meta: Dict[str, Any] = doc.get("metadata", {})

    if not content.strip():
        logger.warning("Document %s has no content – skipping.", doc_id)
        return []

    chunker_fn = _STRATEGY_MAP.get(strategy)
    if chunker_fn is None:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Choose from {VALID_STRATEGIES}."
        )

    raw_chunks = chunker_fn(content, chunk_size, overlap, token_counter)

    chunks: List[Chunk] = []
    search_pos = 0
    for idx, text in enumerate(raw_chunks):
        start, end = _find_span(content, text, search_pos)
        if start != -1:
            search_pos = start + 1  # advance for next search

        tokens = token_counter.count(text)
        meta = ChunkMetadata(
            source_doc=doc_id,
            position=idx,
            start_char=max(start, 0),
            end_char=max(end, 0),
            parent_title=title,
            section=doc_meta.get("section", ""),
            extra={k: v for k, v in doc_meta.items() if k != "section"},
        )
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_chunk{idx}",
                text=text,
                tokens=tokens,
                metadata=meta,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def chunk_documents(
    documents_path: str,
    output_path: str = "/tmp/outputs/chunks.json",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = DEFAULT_STRATEGY,
) -> str:
    """Load documents JSON, chunk them, and write the output artifact.

    Parameters
    ----------
    documents_path:
        Path to the JSON file produced by the crawler component.  Expected
        schema: ``{"documents": [{"id", "content", "title", "metadata", …}]}``.
    output_path:
        Destination path for the chunked output JSON.
    chunk_size:
        Target number of tokens per chunk.
    overlap:
        Number of overlapping tokens between consecutive chunks.
    strategy:
        One of ``fixed``, ``semantic``, or ``sentence``.

    Returns
    -------
    str
        The path to the written output file.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of {VALID_STRATEGIES}."
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size.")

    # ------------------------------------------------------------------
    # 2. Load documents
    # ------------------------------------------------------------------
    docs_file = Path(documents_path)
    if not docs_file.is_file():
        raise FileNotFoundError(
            f"Documents file not found: {documents_path}"
        )

    logger.info("Loading documents from %s", documents_path)
    with open(docs_file, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    documents: List[Dict[str, Any]] = payload.get("documents", [])
    if not documents:
        logger.warning("No documents found in %s", documents_path)

    logger.info("Loaded %d documents.", len(documents))

    # ------------------------------------------------------------------
    # 3. Chunk
    # ------------------------------------------------------------------
    token_counter = TokenCounter()
    all_chunks: List[Chunk] = []

    for doc in documents:
        doc_chunks = chunk_document(
            doc,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
            token_counter=token_counter,
        )
        all_chunks.extend(doc_chunks)

    logger.info(
        "Created %d chunks from %d documents using '%s' strategy.",
        len(all_chunks),
        len(documents),
        strategy,
    )

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    total_tokens = sum(c.tokens for c in all_chunks)
    avg_size = total_tokens / len(all_chunks) if all_chunks else 0.0
    elapsed = time.time() - start_time

    metrics = ChunkingMetrics(
        total_chunks=len(all_chunks),
        average_chunk_size=round(avg_size, 2),
        total_tokens=total_tokens,
        documents_processed=len(documents),
        strategy=strategy,
        elapsed_seconds=round(elapsed, 3),
    )

    logger.info(
        "Metrics – chunks: %d | avg tokens: %.1f | total tokens: %d | time: %.2fs",
        metrics.total_chunks,
        metrics.average_chunk_size,
        metrics.total_tokens,
        metrics.elapsed_seconds,
    )

    # ------------------------------------------------------------------
    # 5. Serialise and write output
    # ------------------------------------------------------------------
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "tokens": c.tokens,
                "metadata": {
                    "source_doc": c.metadata.source_doc,
                    "position": c.metadata.position,
                    "parent_title": c.metadata.parent_title,
                    "section": c.metadata.section,
                    "start_char": c.metadata.start_char,
                    "end_char": c.metadata.end_char,
                    **c.metadata.extra,
                },
            }
            for c in all_chunks
        ],
        "metrics": asdict(metrics),
    }

    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Wrote %d chunks to %s", len(all_chunks), output)

    # ------------------------------------------------------------------
    # 6. Write KFP metrics (optional sidecar file)
    # ------------------------------------------------------------------
    metrics_path = output.parent / "metrics.json"
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2))
    logger.info("Wrote metrics to %s", metrics_path)

    return str(output)


# ---------------------------------------------------------------------------
# KFP v2 component wrapper
# ---------------------------------------------------------------------------
def document_chunker_component(
    documents_path: str,
    output_path: str = "/tmp/outputs/chunks.json",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = DEFAULT_STRATEGY,
) -> str:
    """Thin wrapper expected by the KFP component spec."""
    return chunk_documents(
        documents_path=documents_path,
        output_path=output_path,
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# CLI entry-point for local testing / container execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk documents for the RAG pipeline."
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        required=True,
        help="Path to the crawler output JSON (documents.json).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/tmp/outputs/chunks.json",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target chunk size in tokens (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Overlap in tokens between chunks (default: {DEFAULT_OVERLAP}).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        choices=VALID_STRATEGIES,
        help=f"Chunking strategy (default: {DEFAULT_STRATEGY}).",
    )
    args = parser.parse_args()

    path = document_chunker_component(
        documents_path=args.documents_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
    )
    print(f"Output written to {path}")
