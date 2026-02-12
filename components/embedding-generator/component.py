"""Embedding Generator - KFP Component.

Loads text chunks produced by the document-chunker component and generates
dense vector embeddings using a sentence-transformer model.  Supports
batch processing, GPU acceleration, checkpoint saving, and memory-efficient
streaming for large datasets.

Outputs:
  • embeddings.npy  – NumPy array of shape (N, D)
  • metadata.json   – per-chunk metadata aligned with the array rows
  • embeddings.json – full combined artifact (optional, smaller datasets)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("embedding-generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cpu"
DEFAULT_OUTPUT_DIR = "/tmp/outputs"

CHECKPOINT_FILENAME = "checkpoint.npz"
EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"
EMBEDDINGS_JSON_FILENAME = "embeddings.json"
METRICS_FILENAME = "metrics.json"

# Cache directory for Hugging Face model downloads
MODEL_CACHE_DIR = os.environ.get(
    "TRANSFORMERS_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingRecord:
    """A single embedding with its associated metadata."""

    chunk_id: str
    embedding: List[float]
    model: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingMetrics:
    """Aggregate metrics for the embedding run."""

    total_embeddings: int = 0
    embedding_dimensions: int = 0
    model_name: str = ""
    device: str = ""
    batch_size: int = 0
    embeddings_per_second: float = 0.0
    total_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    resumed_from_checkpoint: bool = False
    checkpoint_chunks_skipped: int = 0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Manages checkpoint save/load for resumable embedding generation."""

    def __init__(self, output_dir: Path) -> None:
        self.checkpoint_path = output_dir / CHECKPOINT_FILENAME
        self._output_dir = output_dir

    @property
    def exists(self) -> bool:
        return self.checkpoint_path.exists()

    def save(
        self,
        embeddings: np.ndarray,
        processed_ids: List[str],
    ) -> None:
        """Persist current progress to disk."""
        np.savez(
            self.checkpoint_path,
            embeddings=embeddings,
            processed_ids=np.array(processed_ids, dtype=object),
        )
        logger.info(
            "Checkpoint saved – %d embeddings written to %s",
            len(processed_ids),
            self.checkpoint_path,
        )

    def load(self) -> tuple[np.ndarray, List[str]]:
        """Restore embeddings and processed IDs from the last checkpoint."""
        data = np.load(self.checkpoint_path, allow_pickle=True)
        embeddings = data["embeddings"]
        processed_ids = data["processed_ids"].tolist()
        logger.info(
            "Checkpoint restored – %d embeddings loaded from %s",
            len(processed_ids),
            self.checkpoint_path,
        )
        return embeddings, processed_ids

    def remove(self) -> None:
        """Delete the checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint file removed.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    cache_dir: Optional[str] = None,
) -> SentenceTransformer:
    """Load (or download) a sentence-transformer model.

    Model weights are cached in *cache_dir* (defaults to the HuggingFace
    cache directory) so subsequent runs skip the download.
    """
    effective_cache = cache_dir or MODEL_CACHE_DIR
    logger.info(
        "Loading model '%s' on device '%s' (cache: %s)",
        model_name,
        device,
        effective_cache,
    )

    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=effective_cache,
    )

    dim = model.get_sentence_embedding_dimension()
    logger.info(
        "Model loaded – embedding dimension: %d, device: %s",
        dim,
        device,
    )
    return model


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> str:
    """Return the best available device, falling back to CPU."""
    if requested == "cuda" and torch.cuda.is_available():
        dev = "cuda"
        logger.info("CUDA device selected: %s", torch.cuda.get_device_name(0))
    elif requested == "cuda":
        logger.warning(
            "CUDA requested but not available – falling back to CPU."
        )
        dev = "cpu"
    else:
        dev = "cpu"
    return dev


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------
def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    """Load chunks produced by the document-chunker component.

    Expected input schema (from document-chunker):
    {
      "chunks": [
        {
          "chunk_id": "...",
          "text": "...",
          "tokens": 123,
          "metadata": { ... }
        }
      ]
    }
    """
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    chunks: List[Dict[str, Any]]
    if isinstance(data, dict) and "chunks" in data:
        chunks = data["chunks"]
    elif isinstance(data, list):
        chunks = data
    else:
        raise ValueError(
            f"Unexpected chunks format. Expected dict with 'chunks' key "
            f"or a list, got {type(data).__name__}."
        )

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------
def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    checkpoint_interval: int = 5,
) -> tuple[np.ndarray, List[Dict[str, Any]], EmbeddingMetrics]:
    """Generate embeddings for *chunks* with checkpointing and progress tracking.

    Parameters
    ----------
    chunks:
        List of chunk dicts (must contain ``chunk_id`` and ``text``).
    model:
        A loaded ``SentenceTransformer`` instance.
    model_name:
        Name of the model (stored in metadata).
    batch_size:
        Number of texts encoded per forward pass.
    device:
        Compute device (``cpu`` or ``cuda``).
    output_dir:
        Directory for checkpoints and outputs.
    checkpoint_interval:
        Save a checkpoint every *N* batches.

    Returns
    -------
    (embeddings_array, metadata_records, metrics)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ckpt = CheckpointManager(out_path)
    dim = model.get_sentence_embedding_dimension()

    # ---- Resume from checkpoint if available ----------------------------
    processed_ids: List[str] = []
    existing_embeddings: Optional[np.ndarray] = None
    skipped = 0

    if ckpt.exists:
        existing_embeddings, processed_ids = ckpt.load()
        skipped = len(processed_ids)
        logger.info("Resuming – skipping %d already-processed chunks.", skipped)

    processed_set = set(processed_ids)
    remaining = [c for c in chunks if c["chunk_id"] not in processed_set]
    logger.info(
        "Chunks to process: %d (skipped: %d, total: %d)",
        len(remaining),
        skipped,
        len(chunks),
    )

    # ---- Batch generation -----------------------------------------------
    start_time = time.time()
    batch_embeddings_list: List[np.ndarray] = []
    batch_ids: List[str] = []
    total_batches = (len(remaining) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(remaining), batch_size):
        batch = remaining[batch_idx : batch_idx + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]

        batch_num = batch_idx // batch_size + 1
        logger.info(
            "Batch %d/%d – encoding %d texts …",
            batch_num,
            total_batches,
            len(texts),
        )

        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=device,
            normalize_embeddings=True,
        )

        batch_embeddings_list.append(emb)
        batch_ids.extend(ids)
        processed_ids.extend(ids)

        # Periodic checkpoint
        if checkpoint_interval and batch_num % checkpoint_interval == 0:
            _combined = np.vstack(
                [e for e in ([existing_embeddings] if existing_embeddings is not None else []) + batch_embeddings_list]
            )
            ckpt.save(_combined, processed_ids)

        # Free GPU memory every few batches
        if device == "cuda" and batch_num % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # ---- Assemble final arrays ------------------------------------------
    parts = []
    if existing_embeddings is not None:
        parts.append(existing_embeddings)
    parts.extend(batch_embeddings_list)

    if parts:
        all_embeddings = np.vstack(parts)
    else:
        all_embeddings = np.empty((0, dim), dtype=np.float32)

    elapsed = time.time() - start_time
    new_count = len(remaining)
    eps = new_count / elapsed if elapsed > 0 else 0.0

    # ---- Build metadata records -----------------------------------------
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name

    metadata_records: List[Dict[str, Any]] = []
    for idx, cid in enumerate(processed_ids):
        chunk = chunk_lookup.get(cid, {})
        metadata_records.append(
            {
                "chunk_id": cid,
                "model": short_model_name,
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "embedding_index": idx,
            }
        )

    # ---- Compute peak memory -------------------------------------------
    peak_mem = 0.0
    if device == "cuda" and torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    metrics = EmbeddingMetrics(
        total_embeddings=len(all_embeddings),
        embedding_dimensions=dim,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        embeddings_per_second=round(eps, 2),
        total_time_seconds=round(elapsed, 3),
        peak_memory_mb=round(peak_mem, 2),
        resumed_from_checkpoint=skipped > 0,
        checkpoint_chunks_skipped=skipped,
    )

    # Clean up checkpoint on success
    ckpt.remove()

    logger.info(
        "Embedding generation complete – %d embeddings (dim=%d) in %.2fs (%.1f emb/s)",
        metrics.total_embeddings,
        metrics.embedding_dimensions,
        metrics.total_time_seconds,
        metrics.embeddings_per_second,
    )

    return all_embeddings, metadata_records, metrics


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def save_outputs(
    embeddings: np.ndarray,
    metadata_records: List[Dict[str, Any]],
    metrics: EmbeddingMetrics,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_combined_json: bool = True,
) -> Dict[str, str]:
    """Persist embeddings, metadata, and metrics to *output_dir*.

    Files written:
      • embeddings.npy   – dense float32 array  (N, D)
      • metadata.json    – list of per-chunk metadata
      • metrics.json     – run metrics
      • embeddings.json  – combined JSON (optional, for small datasets)

    Returns a dict mapping logical name → file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # 1. NumPy embeddings
    emb_path = out / EMBEDDINGS_FILENAME
    np.save(emb_path, embeddings.astype(np.float32))
    paths["embeddings_npy"] = str(emb_path)
    logger.info("Saved embeddings array %s to %s", embeddings.shape, emb_path)

    # 2. Metadata JSON
    meta_path = out / METADATA_FILENAME
    meta_path.write_text(
        json.dumps(metadata_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["metadata"] = str(meta_path)
    logger.info("Saved metadata (%d records) to %s", len(metadata_records), meta_path)

    # 3. Metrics JSON
    metrics_path = out / METRICS_FILENAME
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["metrics"] = str(metrics_path)
    logger.info("Saved metrics to %s", metrics_path)

    # 4. Combined JSON (optional – useful for inspection / small datasets)
    if save_combined_json:
        combined: Dict[str, Any] = {
            "embeddings": [
                {
                    "chunk_id": rec["chunk_id"],
                    "embedding": embeddings[rec["embedding_index"]].tolist(),
                    "model": rec["model"],
                    "text": rec["text"],
                    "metadata": rec["metadata"],
                }
                for rec in metadata_records
            ],
            "metrics": asdict(metrics),
        }
        json_path = out / EMBEDDINGS_JSON_FILENAME
        json_path.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["embeddings_json"] = str(json_path)
        logger.info("Saved combined JSON to %s", json_path)

    return paths


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def generate_and_save_embeddings(
    chunks_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """End-to-end: load chunks → generate embeddings → write outputs.

    This is the function called by both the KFP component wrapper and the
    CLI entry-point.
    """
    logger.info("=" * 60)
    logger.info("Embedding Generator – starting")
    logger.info("=" * 60)

    # 1. Resolve device
    device = resolve_device(device)

    # 2. Load chunks
    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.warning("No chunks to process – writing empty outputs.")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / EMBEDDINGS_FILENAME, np.empty((0, 384), dtype=np.float32))
        (out / METADATA_FILENAME).write_text("[]")
        (out / METRICS_FILENAME).write_text(
            json.dumps(asdict(EmbeddingMetrics(model_name=model_name, device=device)), indent=2)
        )
        return str(out)

    # 3. Load model
    model = load_model(model_name=model_name, device=device)

    # 4. Generate embeddings
    embeddings, metadata_records, metrics = generate_embeddings(
        chunks=chunks,
        model=model,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        output_dir=output_dir,
    )

    # 5. Save outputs
    save_combined = len(chunks) <= 50_000  # skip full JSON for very large sets
    paths = save_outputs(
        embeddings=embeddings,
        metadata_records=metadata_records,
        metrics=metrics,
        output_dir=output_dir,
        save_combined_json=save_combined,
    )

    logger.info("All outputs written to %s", output_dir)
    logger.info("Files: %s", json.dumps(paths, indent=2))

    return output_dir


# ---------------------------------------------------------------------------
# KFP v2 component wrapper
# ---------------------------------------------------------------------------
def embedding_generator_component(
    chunks_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Thin wrapper expected by the KFP component spec."""
    return generate_and_save_embeddings(
        chunks_path=chunks_path,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# CLI entry-point for local testing / container execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings for text chunks (RAG pipeline)."
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        required=True,
        help="Path to the chunks JSON produced by the document-chunker.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Sentence-transformer model to use (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for encoding (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help=f"Compute device (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    result_dir = embedding_generator_component(
        chunks_path=args.chunks_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(f"Outputs written to {result_dir}")
