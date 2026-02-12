"""Kubeflow Docs Ingestion Pipeline – KFP v2.

Orchestrates the full document-ingestion workflow:

    CrawlDocs ─► ChunkText ─► GenerateEmbeddings ─► ValidateOutput

Each task maps to an existing container component image and communicates
via file-based artifacts on a shared PVC / object store.

Usage:
    # Compile to YAML
    python ingestion_pipeline.py --compile

    # Or import in run_pipeline.py to submit directly
"""

from __future__ import annotations

import argparse
import json
from typing import List

from kfp import compiler, dsl
from kfp.dsl import Artifact, Input, Metrics, Output

# ---------------------------------------------------------------------------
# Container image tags – override via pipeline_config.yaml or env vars
# ---------------------------------------------------------------------------
CRAWLER_IMAGE = "kubeflow-docs-crawler:latest"
CHUNKER_IMAGE = "document-chunker:latest"
EMBEDDER_IMAGE = "embedding-generator:latest"

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
DEFAULT_REPOS = json.dumps([
    "https://github.com/kubeflow/website",
    "https://github.com/kubeflow/manifests",
    "https://github.com/kubeflow/pipelines",
])
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_CHUNK_STRATEGY = "semantic"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Component 1 – CrawlDocs
# ═══════════════════════════════════════════════════════════════════════════
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "requests>=2.31",
        "beautifulsoup4>=4.12",
        "gitpython>=3.1",
    ],
)
def crawl_docs(
    repos_json: str,
    include_examples: bool,
    documents: Output[Artifact],
    crawl_metrics: Output[Metrics],
) -> None:
    """Clone Kubeflow repos and extract markdown documentation.

    Produces ``documents.json`` – a structured list of documents with
    metadata such as source, title, URL, section, and version.
    """
    import hashlib
    import json as _json
    import logging
    import os
    import re
    import shutil
    import tempfile
    import time
    from pathlib import Path
    from typing import Any, Optional
    from urllib.parse import urljoin

    import git
    import requests
    from bs4 import BeautifulSoup

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("crawl-docs")

    DEFAULT_REPO_LIST = [
        "https://github.com/kubeflow/website",
        "https://github.com/kubeflow/manifests",
        "https://github.com/kubeflow/pipelines",
    ]
    REPO_SECTION_MAP = {"website": "docs", "manifests": "architecture", "pipelines": "examples"}
    DOCS_BASE_URL = "https://www.kubeflow.org/docs/"
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2

    # -- helpers --
    def _retry(func, *a, retries=MAX_RETRIES, **kw) -> Any:
        last: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                return func(*a, **kw)
            except Exception as exc:
                last = exc
                time.sleep(RETRY_BACKOFF ** attempt)
        raise RuntimeError(f"All {retries} attempts failed") from last

    def _gen_id(source: str, path: str) -> str:
        return hashlib.sha256(f"{source}:{path}".encode()).hexdigest()[:16]

    def _extract_title(content: str) -> str:
        for line in content.splitlines():
            s = line.strip()
            if s.startswith("#"):
                return s.lstrip("#").strip()
        return "Untitled"

    def _infer_section(repo_name: str, rel: str) -> str:
        section = REPO_SECTION_MAP.get(repo_name, "general")
        parts = rel.lower().split(os.sep)
        for kw in ("pipelines", "notebooks", "katib", "training", "serving"):
            if kw in parts:
                return kw
        return section

    def _canonical_url(repo_name: str, rel: str) -> str:
        if repo_name == "website":
            marker = "content/en/docs/"
            idx = rel.find(marker)
            if idx != -1:
                slug = rel[idx + len(marker):]
                slug = re.sub(r"(_index)?\.md$", "/", slug)
                return urljoin(DOCS_BASE_URL, slug)
        return f"https://github.com/kubeflow/{repo_name}/blob/master/{rel}"

    # -- main logic --
    repo_urls = _json.loads(repos_json) if repos_json else DEFAULT_REPO_LIST
    work_dir = Path(tempfile.mkdtemp(prefix="crawl_"))
    all_documents: list = []
    start = time.time()

    for url in repo_urls:
        repo_name = url.rstrip("/").split("/")[-1]
        clone_path = work_dir / repo_name
        logger.info("Cloning %s", url)
        _retry(git.Repo.clone_from, url, str(clone_path), depth=1, single_branch=True)

        for root_dir in ["docs", "content", "examples", "doc"]:
            candidate = clone_path / root_dir
            if not candidate.is_dir():
                continue
            for md in candidate.rglob("*.md"):
                try:
                    content = md.read_text(errors="replace")
                except OSError:
                    continue
                rel = str(md.relative_to(clone_path))
                all_documents.append({
                    "id": _gen_id(repo_name, rel),
                    "source": f"{repo_name}/{rel}",
                    "title": _extract_title(content),
                    "content": content,
                    "url": _canonical_url(repo_name, rel),
                    "metadata": {"section": _infer_section(repo_name, rel)},
                })

    # de-duplicate
    seen: set = set()
    unique = []
    for d in all_documents:
        if d["id"] not in seen:
            seen.add(d["id"])
            unique.append(d)
    all_documents = unique
    elapsed = time.time() - start

    # write artifact
    out = Path(documents.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps({"documents": all_documents}, indent=2, ensure_ascii=False))
    logger.info("Wrote %d documents to %s", len(all_documents), out)

    # metrics
    crawl_metrics.log_metric("documents_crawled", len(all_documents))
    crawl_metrics.log_metric("repos_processed", len(repo_urls))
    crawl_metrics.log_metric("elapsed_seconds", round(elapsed, 2))

    shutil.rmtree(work_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Component 2 – ChunkText
# ═══════════════════════════════════════════════════════════════════════════
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "tiktoken>=0.5",
        "langchain>=0.1,<0.3",
        "langchain-text-splitters>=0.0.1",
    ],
)
def chunk_text(
    documents: Input[Artifact],
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    chunks: Output[Artifact],
    chunk_metrics: Output[Metrics],
) -> None:
    """Split crawled documents into overlapping, token-counted chunks.

    Strategies:
      - ``fixed``    – fixed-size token windows
      - ``semantic`` – split on markdown headers, then sub-split
      - ``sentence`` – split on sentence boundaries
    """
    import json as _json
    import logging
    import re
    import time
    from pathlib import Path

    import tiktoken
    from langchain.text_splitter import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("chunk-text")

    ENCODING = "cl100k_base"
    enc = tiktoken.get_encoding(ENCODING)
    length_fn = lambda t: len(enc.encode(t))  # noqa: E731

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
    _MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

    def _fixed(text: str) -> list[str]:
        sp = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=length_fn, separators=["\n\n", "\n", ". ", " ", ""],
        )
        return sp.split_text(text)

    def _semantic(text: str) -> list[str]:
        md_sp = MarkdownHeaderTextSplitter(headers_to_split_on=_MD_HEADERS, strip_headers=False)
        md_docs = md_sp.split_text(text)
        sub_sp = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=length_fn, separators=["\n\n", "\n", ". ", " ", ""],
        )
        result: list[str] = []
        for doc in md_docs:
            sec = doc.page_content
            headers = [doc.metadata.get(h) for h in ("h1", "h2", "h3", "h4") if doc.metadata.get(h)]
            if headers:
                sec = " > ".join(headers) + "\n\n" + sec
            if length_fn(sec) > chunk_size:
                result.extend(sub_sp.split_text(sec))
            else:
                result.append(sec)
        return result or _fixed(text)

    def _sentence(text: str) -> list[str]:
        sentences = _SENTENCE_RE.split(text)
        chunks_out: list[str] = []
        cur: list[str] = []
        cur_tok = 0
        for s in sentences:
            st = length_fn(s)
            if cur_tok + st > chunk_size and cur:
                chunks_out.append(" ".join(cur))
                overlap_s: list[str] = []
                ot = 0
                for x in reversed(cur):
                    xt = length_fn(x)
                    if ot + xt > chunk_overlap:
                        break
                    overlap_s.insert(0, x)
                    ot += xt
                cur, cur_tok = overlap_s, ot
            cur.append(s)
            cur_tok += st
        if cur:
            chunks_out.append(" ".join(cur))
        return chunks_out

    STRAT = {"fixed": _fixed, "semantic": _semantic, "sentence": _sentence}

    # -- load documents --
    data = _json.loads(Path(documents.path).read_text())
    docs = data.get("documents", data) if isinstance(data, dict) else data

    start = time.time()
    all_chunks: list[dict] = []
    for doc in docs:
        content = doc.get("content", "")
        if not content.strip():
            continue
        fn = STRAT.get(strategy, _semantic)
        parts = fn(content)
        for idx, text in enumerate(parts):
            cid = f"{doc['id']}_{idx:04d}"
            all_chunks.append({
                "chunk_id": cid,
                "text": text,
                "tokens": length_fn(text),
                "metadata": {
                    "source_doc": doc.get("source", ""),
                    "position": idx,
                    "parent_title": doc.get("title", ""),
                    "section": doc.get("metadata", {}).get("section", ""),
                },
            })
    elapsed = time.time() - start

    # -- write output --
    total_tokens = sum(c["tokens"] for c in all_chunks)
    avg_size = total_tokens / len(all_chunks) if all_chunks else 0
    result = {
        "chunks": all_chunks,
        "metrics": {
            "total_chunks": len(all_chunks),
            "average_chunk_size": round(avg_size, 2),
            "total_tokens": total_tokens,
            "documents_processed": len(docs),
            "strategy": strategy,
            "elapsed_seconds": round(elapsed, 2),
        },
    }
    out = Path(chunks.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Wrote %d chunks to %s", len(all_chunks), out)

    chunk_metrics.log_metric("total_chunks", len(all_chunks))
    chunk_metrics.log_metric("total_tokens", total_tokens)
    chunk_metrics.log_metric("average_chunk_size", round(avg_size, 2))
    chunk_metrics.log_metric("documents_processed", len(docs))
    chunk_metrics.log_metric("elapsed_seconds", round(elapsed, 2))


# ═══════════════════════════════════════════════════════════════════════════
# Component 3 – GenerateEmbeddings
# ═══════════════════════════════════════════════════════════════════════════
@dsl.component(
    base_image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=[
        "sentence-transformers>=2.6",
        "numpy>=1.26",
    ],
)
def generate_embeddings(
    chunks: Input[Artifact],
    embedding_model: str,
    batch_size: int,
    device: str,
    embeddings_artifact: Output[Artifact],
    metadata_artifact: Output[Artifact],
    embed_metrics: Output[Metrics],
) -> None:
    """Generate dense vector embeddings from text chunks.

    Outputs ``embeddings.npy`` and ``metadata.json`` into a shared
    output directory.
    """
    import gc
    import json as _json
    import logging
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("generate-embeddings")

    # -- resolve device --
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable – falling back to CPU")
        device = "cpu"

    # -- load chunks --
    raw = _json.loads(Path(chunks.path).read_text())
    chunk_list = raw.get("chunks", raw) if isinstance(raw, dict) else raw
    if not chunk_list:
        logger.warning("No chunks – writing empty outputs")
        out_e = Path(embeddings_artifact.path)
        out_e.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_e, np.empty((0, 384), dtype=np.float32))
        out_m = Path(metadata_artifact.path)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text("[]")
        embed_metrics.log_metric("total_embeddings", 0)
        return

    # -- load model --
    logger.info("Loading model %s on %s", embedding_model, device)
    model = SentenceTransformer(embedding_model, device=device)
    dim = model.get_sentence_embedding_dimension()

    texts = [c["text"] for c in chunk_list]
    meta = [
        {"chunk_id": c["chunk_id"], "tokens": c.get("tokens", 0), **c.get("metadata", {})}
        for c in chunk_list
    ]

    # -- batch encode --
    start = time.time()
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(embs)
        if (i // batch_size) % 10 == 0:
            logger.info("Encoded %d / %d", min(i + batch_size, len(texts)), len(texts))
    embeddings = np.vstack(all_embs).astype(np.float32)
    elapsed = time.time() - start

    # -- save embeddings --
    out_e = Path(embeddings_artifact.path)
    out_e.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_e, embeddings)

    # -- save metadata --
    out_m = Path(metadata_artifact.path)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(_json.dumps(meta, indent=2, ensure_ascii=False))

    logger.info(
        "Generated %d embeddings (dim=%d) in %.1fs",
        embeddings.shape[0], dim, elapsed,
    )

    # -- metrics --
    embed_metrics.log_metric("total_embeddings", int(embeddings.shape[0]))
    embed_metrics.log_metric("embedding_dimensions", int(dim))
    embed_metrics.log_metric("model_name", embedding_model)
    embed_metrics.log_metric("elapsed_seconds", round(elapsed, 2))
    embed_metrics.log_metric("embeddings_per_second", round(embeddings.shape[0] / elapsed, 2) if elapsed > 0 else 0)

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# Component 4 – ValidateOutput
# ═══════════════════════════════════════════════════════════════════════════
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["numpy>=1.26"],
)
def validate_output(
    documents: Input[Artifact],
    chunks: Input[Artifact],
    embeddings_artifact: Input[Artifact],
    metadata_artifact: Input[Artifact],
    validation_metrics: Output[Metrics],
) -> str:
    """Validate all pipeline outputs and print summary statistics.

    Checks:
      - All expected files exist and are non-empty.
      - Embedding array shape is (N, D) with D > 0.
      - Metadata row count matches embedding count.
      - Document / chunk / embedding counts are consistent.

    Returns a JSON summary string.
    """
    import json as _json
    import logging
    from pathlib import Path

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("validate-output")

    errors: list[str] = []
    summary: dict = {}

    # --- documents ---
    doc_path = Path(documents.path)
    if not doc_path.exists() or doc_path.stat().st_size == 0:
        errors.append("documents.json is missing or empty")
        num_docs = 0
    else:
        doc_data = _json.loads(doc_path.read_text())
        num_docs = len(doc_data.get("documents", doc_data) if isinstance(doc_data, dict) else doc_data)
    summary["total_documents"] = num_docs

    # --- chunks ---
    chunk_path = Path(chunks.path)
    if not chunk_path.exists() or chunk_path.stat().st_size == 0:
        errors.append("chunks.json is missing or empty")
        num_chunks = 0
    else:
        chunk_data = _json.loads(chunk_path.read_text())
        chunk_list = chunk_data.get("chunks", chunk_data) if isinstance(chunk_data, dict) else chunk_data
        num_chunks = len(chunk_list)
    summary["total_chunks"] = num_chunks

    # --- embeddings ---
    emb_path = Path(embeddings_artifact.path)
    if not emb_path.exists() or emb_path.stat().st_size == 0:
        errors.append("embeddings.npy is missing or empty")
        num_embs, emb_dim = 0, 0
    else:
        emb = np.load(emb_path)
        num_embs = emb.shape[0]
        emb_dim = emb.shape[1] if emb.ndim == 2 else 0
        if emb_dim == 0:
            errors.append(f"Unexpected embedding shape: {emb.shape}")
    summary["total_embeddings"] = num_embs
    summary["embedding_dimensions"] = emb_dim

    # --- metadata ---
    meta_path = Path(metadata_artifact.path)
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        errors.append("metadata.json is missing or empty")
        num_meta = 0
    else:
        meta_list = _json.loads(meta_path.read_text())
        num_meta = len(meta_list)
    summary["metadata_records"] = num_meta

    # --- cross-validation ---
    if num_embs != num_meta:
        errors.append(f"Embedding count ({num_embs}) != metadata count ({num_meta})")
    if num_chunks > 0 and num_embs != num_chunks:
        errors.append(f"Chunk count ({num_chunks}) != embedding count ({num_embs})")

    summary["errors"] = errors
    summary["status"] = "PASSED" if not errors else "FAILED"

    # --- log metrics ---
    validation_metrics.log_metric("total_documents", num_docs)
    validation_metrics.log_metric("total_chunks", num_chunks)
    validation_metrics.log_metric("total_embeddings", num_embs)
    validation_metrics.log_metric("embedding_dimensions", emb_dim)
    validation_metrics.log_metric("validation_status", 1 if not errors else 0)

    # --- print summary ---
    border = "=" * 60
    print(f"\n{border}")
    print("  PIPELINE VALIDATION SUMMARY")
    print(border)
    print(f"  Status           : {summary['status']}")
    print(f"  Documents crawled: {num_docs}")
    print(f"  Chunks created   : {num_chunks}")
    print(f"  Embeddings       : {num_embs}  (dim={emb_dim})")
    print(f"  Metadata records : {num_meta}")
    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print(f"\n  ✓ All checks passed.")
    print(f"{border}\n")

    return _json.dumps(summary, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline definition
# ═══════════════════════════════════════════════════════════════════════════
@dsl.pipeline(
    name="kubeflow-docs-ingestion-pipeline",
    description=(
        "End-to-end document ingestion pipeline for the Kubeflow docs RAG "
        "system.  Crawls repositories, chunks text, generates embeddings, "
        "and validates all outputs."
    ),
)
def kubeflow_docs_ingestion_pipeline(
    repos: str = DEFAULT_REPOS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    include_examples: bool = True,
) -> None:
    """Orchestrate crawl → chunk → embed → validate."""

    # ── Step 1: Crawl documentation ──────────────────────────────────
    crawl_task = crawl_docs(
        repos_json=repos,
        include_examples=include_examples,
    )
    crawl_task.set_display_name("CrawlDocs")
    crawl_task.set_caching_options(enable_caching=True)
    crawl_task.set_retry(
        num_retries=3,
        backoff_duration="30s",
        backoff_factor=2.0,
        backoff_max_duration="600s",
    )
    crawl_task.set_cpu_request("1")
    crawl_task.set_memory_request("2Gi")
    crawl_task.set_cpu_limit("2")
    crawl_task.set_memory_limit("4Gi")

    # ── Step 2: Chunk text ───────────────────────────────────────────
    chunk_task = chunk_text(
        documents=crawl_task.outputs["documents"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy,
    )
    chunk_task.set_display_name("ChunkText")
    chunk_task.set_caching_options(enable_caching=True)
    chunk_task.set_cpu_request("1")
    chunk_task.set_memory_request("2Gi")
    chunk_task.set_cpu_limit("2")
    chunk_task.set_memory_limit("4Gi")

    # ── Step 3: Generate embeddings ──────────────────────────────────
    embed_task = generate_embeddings(
        chunks=chunk_task.outputs["chunks"],
        embedding_model=embedding_model,
        batch_size=batch_size,
        device=device,
    )
    embed_task.set_display_name("GenerateEmbeddings")
    embed_task.set_caching_options(enable_caching=True)
    embed_task.set_retry(
        num_retries=2,
        backoff_duration="60s",
        backoff_factor=2.0,
        backoff_max_duration="300s",
    )
    embed_task.set_cpu_request("2")
    embed_task.set_memory_request("4Gi")
    embed_task.set_cpu_limit("4")
    embed_task.set_memory_limit("8Gi")

    # ── Step 4: Validate all outputs ─────────────────────────────────
    validate_task = validate_output(
        documents=crawl_task.outputs["documents"],
        chunks=chunk_task.outputs["chunks"],
        embeddings_artifact=embed_task.outputs["embeddings_artifact"],
        metadata_artifact=embed_task.outputs["metadata_artifact"],
    )
    validate_task.set_display_name("ValidateOutput")
    validate_task.set_cpu_request("0.5")
    validate_task.set_memory_request("512Mi")
    validate_task.set_cpu_limit("1")
    validate_task.set_memory_limit("1Gi")


# ═══════════════════════════════════════════════════════════════════════════
# CLI – compile pipeline to YAML
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile the docs-ingestion pipeline.")
    parser.add_argument(
        "--output",
        type=str,
        default="pipelines/kubeflow_docs_ingestion_pipeline.yaml",
        help="Path for the compiled pipeline YAML.",
    )
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=kubeflow_docs_ingestion_pipeline,
        package_path=args.output,
    )
    print(f"Pipeline compiled → {args.output}")
