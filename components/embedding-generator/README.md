# embedding-generator

A Kubeflow Pipelines (KFP) component that generates dense vector embeddings from text chunks using [sentence-transformers](https://www.sbert.net/).

## What it does

1. **Loads** text chunks produced by the `document-chunker` component.
2. **Initialises** a sentence-transformer model (default: `all-MiniLM-L6-v2`, 384 dimensions).
3. **Generates** normalised embeddings with batch processing and optional GPU acceleration.
4. **Checkpoints** progress so interrupted runs can be resumed.
5. **Writes** outputs in multiple formats for downstream flexibility.

## Output files

| File | Format | Description |
|------|--------|-------------|
| `embeddings.npy` | NumPy `float32` array `(N, 384)` | Dense embedding vectors |
| `metadata.json` | JSON list | Per-chunk metadata aligned with array rows |
| `embeddings.json` | JSON | Combined embeddings + metadata (datasets ≤ 50 k chunks) |
| `metrics.json` | JSON | Run metrics |

### Combined JSON schema (`embeddings.json`)

```json
{
  "embeddings": [
    {
      "chunk_id": "doc1_chunk0",
      "embedding": [0.1, 0.2, ...],
      "model": "all-MiniLM-L6-v2",
      "text": "original chunk text",
      "metadata": {
        "source_doc": "website/docs/intro.md",
        "position": 0
      }
    }
  ],
  "metrics": { ... }
}
```

## Input parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunks_path` | `str` | *(required)* | Path to the chunks JSON from `document-chunker` |
| `model_name` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model identifier |
| `batch_size` | `int` | `32` | Texts per forward pass |
| `device` | `str` | `cpu` | `cpu` or `cuda` (falls back to CPU if CUDA unavailable) |
| `output_dir` | `str` | `/tmp/outputs` | Directory for output files |

## Metrics

| Metric | Description |
|--------|-------------|
| `total_embeddings` | Number of embeddings generated |
| `embedding_dimensions` | Vector dimensionality (e.g. 384) |
| `embeddings_per_second` | Throughput |
| `total_time_seconds` | Wall-clock time |
| `peak_memory_mb` | Peak GPU memory (CUDA only) |
| `resumed_from_checkpoint` | Whether the run was resumed |

## Usage

### In a KFP pipeline (v1 component YAML)

```python
import kfp
from kfp.components import load_component_from_file

chunker_op = load_component_from_file("components/document-chunker/component.yaml")
embed_op   = load_component_from_file("components/embedding-generator/component.yaml")

@kfp.dsl.pipeline(name="docs-embedding-pipeline")
def pipeline():
    chunks = chunker_op(
        documents_path="/data/documents.json",
    )
    embeddings = embed_op(
        chunks_path=chunks.outputs["chunks"],
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size="32",
        device="cpu",
    )
```

### Local / CLI

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python component.py \
  --chunks-path /data/chunks.json \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 32 \
  --device cpu \
  --output-dir ./output
```

### Docker

```bash
docker build -t embedding-generator .
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/tmp/outputs" \
  embedding-generator \
    --chunks-path /data/chunks.json \
    --batch-size 64 \
    --device cpu
```

## Features

- **Model caching** — downloads are cached in `~/.cache/huggingface` (configurable via `TRANSFORMERS_CACHE`).
- **Checkpointing** — a `checkpoint.npz` file is saved periodically; if the process restarts it picks up where it left off.
- **Memory-efficient** — processes in configurable batches; periodically frees GPU memory with `torch.cuda.empty_cache()`.
- **Dual output** — NumPy binary for efficient downstream loading; JSON for inspection.

## Testing

```bash
cd components/embedding-generator
pip install -r requirements.txt pytest
pytest tests/ -v
```
