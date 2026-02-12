# Kubeflow Docs Ingestion Pipeline – Usage Guide

## Overview

The **kubeflow-docs-ingestion-pipeline** orchestrates the full document
ingestion workflow for the Kubeflow documentation RAG system.  It runs four
sequential tasks:

```
CrawlDocs ──► ChunkText ──► GenerateEmbeddings ──► ValidateOutput
```

| Task | Description | Key I/O |
|------|-------------|---------|
| **CrawlDocs** | Clones Kubeflow GitHub repos and extracts markdown files | → `documents.json` |
| **ChunkText** | Splits documents into overlapping, token-counted chunks | → `chunks.json` |
| **GenerateEmbeddings** | Generates dense vectors with a sentence-transformer model | → `embeddings.npy`, `metadata.json` |
| **ValidateOutput** | Verifies file existence, shapes, and prints summary stats | → validation report |

---

## Prerequisites

1. A running **Kubeflow Pipelines** cluster (v2 / KFP SDK ≥ 2.0).
2. Python 3.10+ with the KFP SDK installed:

```bash
pip install kfp>=2.0 pyyaml
```

3. Network access from the cluster to GitHub (for repo cloning) and
   HuggingFace (for model download).

---

## Quick Start

### 1. Compile the pipeline

```bash
cd pipelines/
python ingestion_pipeline.py --output kubeflow_docs_ingestion_pipeline.yaml
```

This produces a self-contained KFP YAML that can be uploaded to the
Pipelines UI.

### 2. Submit via the CLI

```bash
# Port-forward the KFP UI (if running in a Kind / local cluster)
kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow

# Submit with defaults from pipeline_config.yaml
python run_pipeline.py --host http://localhost:8080

# Or override parameters inline
python run_pipeline.py \
    --host http://localhost:8080 \
    --experiment my-experiment \
    --repos https://github.com/kubeflow/website \
    --chunk-size 256 \
    --embedding-model sentence-transformers/all-mpnet-base-v2
```

### 3. Upload via the Pipelines UI

1. Open `http://localhost:8080` in a browser.
2. Go to **Pipelines → Upload Pipeline**.
3. Select the compiled YAML file.
4. Click **Create Run**, fill in parameters, and start the run.

---

## Pipeline Parameters

All parameters have sensible defaults defined in
`pipelines/pipeline_config.yaml`.  Override them via CLI flags or the
Pipelines UI when starting a run.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repos` | `str` (JSON list) | 3 Kubeflow repos | GitHub repository URLs to crawl |
| `chunk_size` | `int` | `512` | Target chunk size in tokens |
| `chunk_overlap` | `int` | `128` | Overlap tokens between chunks |
| `chunk_strategy` | `str` | `"semantic"` | `fixed`, `semantic`, or `sentence` |
| `embedding_model` | `str` | `all-MiniLM-L6-v2` | HuggingFace model name |
| `batch_size` | `int` | `32` | Encoding batch size |
| `device` | `str` | `"cpu"` | `cpu` or `cuda` |
| `include_examples` | `bool` | `true` | Include pipeline example notebooks |

---

## Configuration File

Edit `pipelines/pipeline_config.yaml` to change defaults without touching
Python code:

```yaml
pipeline_parameters:
  repos:
    - "https://github.com/kubeflow/website"
  chunk_size: 256
  chunk_strategy: "fixed"
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  device: "cuda"

runtime:
  kfp_host: "https://pipelines.example.com"
  experiment_name: "production-ingestion"
```

Use a custom config file:

```bash
python run_pipeline.py --config /path/to/custom_config.yaml
```

---

## Resource Allocation

Resource requests/limits are set per task inside `ingestion_pipeline.py`:

| Task | CPU Request | Memory Request | CPU Limit | Memory Limit |
|------|------------|----------------|-----------|--------------|
| CrawlDocs | 1 | 2 Gi | 2 | 4 Gi |
| ChunkText | 1 | 2 Gi | 2 | 4 Gi |
| GenerateEmbeddings | 2 | 4 Gi | 4 | 8 Gi |
| ValidateOutput | 0.5 | 512 Mi | 1 | 1 Gi |

To change these, edit the `set_cpu_request()` / `set_memory_request()`
calls in `ingestion_pipeline.py`.

---

## Pipeline Features

### Caching

Expensive tasks (CrawlDocs, ChunkText, GenerateEmbeddings) have caching
enabled.  If you re-run the pipeline with identical inputs, KFP will skip
already-completed tasks and reuse cached artifacts.

### Retry Logic

Network-dependent tasks include automatic retries with exponential backoff:

- **CrawlDocs**: 3 retries, 30 s initial backoff, 10 min max
- **GenerateEmbeddings**: 2 retries, 60 s initial backoff, 5 min max

### Visualization

The pipeline graph is fully renderable in the KFP UI.  Each task reports
`Metrics` artifacts so you can inspect:

- Documents crawled, repos processed, and elapsed time
- Chunk counts, average size, total tokens, strategy used
- Embedding count, dimensions, throughput (embeddings/sec)
- Validation pass/fail status

---

## Example Invocation & Expected Output

### Command

```bash
python run_pipeline.py \
    --host http://localhost:8080 \
    --experiment docs-ingestion \
    --repos https://github.com/kubeflow/website \
    --chunk-size 512 \
    --chunk-strategy semantic \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

### CLI Output

```
[info] Submitting run 'docs-ingestion-20260212-143022' to experiment 'docs-ingestion'
[info] KFP host: http://localhost:8080
[info] Parameters:
         repos: ["https://github.com/kubeflow/website"]
         chunk_size: 512
         chunk_overlap: 128
         chunk_strategy: semantic
         embedding_model: sentence-transformers/all-MiniLM-L6-v2
         batch_size: 32
         device: cpu
         include_examples: True

[ok] Run created successfully!
     Run ID   : a1b2c3d4-e5f6-7890-abcd-ef1234567890
     Run URL  : http://localhost:8080/#/runs/details/a1b2c3d4-e5f6-7890-abcd-ef1234567890

Monitor progress in the Kubeflow Pipelines UI.
```

### ValidateOutput Task Log (visible in the KFP UI)

```
============================================================
  PIPELINE VALIDATION SUMMARY
============================================================
  Status           : PASSED
  Documents crawled: 347
  Chunks created   : 2184
  Embeddings       : 2184  (dim=384)
  Metadata records : 2184

  ✓ All checks passed.
============================================================
```

### Metrics Captured

| Metric | Example Value |
|--------|---------------|
| `documents_crawled` | 347 |
| `repos_processed` | 1 |
| `total_chunks` | 2 184 |
| `average_chunk_size` | 387.4 |
| `total_tokens` | 846 081 |
| `total_embeddings` | 2 184 |
| `embedding_dimensions` | 384 |
| `embeddings_per_second` | 142.6 |
| `validation_status` | 1 (pass) |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ConnectionRefusedError` on submit | KFP API not reachable | Run `kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow` |
| CrawlDocs timeout | GitHub rate limiting | Add a GitHub token or reduce repos |
| GenerateEmbeddings OOM | Model + data exceed memory | Lower `batch_size`, or increase memory limit |
| Validation FAILED: count mismatch | Partial embedding failure | Check GenerateEmbeddings logs; retry the run |
| Caching not working | Parameter values changed subtly | Ensure JSON-serialized `repos` is identical between runs |

---

## File Reference

| File | Purpose |
|------|---------|
| `pipelines/ingestion_pipeline.py` | Pipeline definition (4 components + wiring) |
| `pipelines/run_pipeline.py` | CLI script to compile or submit the pipeline |
| `pipelines/pipeline_config.yaml` | Default parameter values and runtime settings |
| `pipelines/requirements.txt` | Python dependencies for the pipeline environment |
