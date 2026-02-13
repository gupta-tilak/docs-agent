# milvus-indexer

A Kubeflow Pipelines (KFP) component that loads dense vector embeddings into a [Milvus](https://milvus.io/) vector database for similarity search in a RAG pipeline.

## What it does

1. **Connects** to a Milvus server with retry logic and connection pooling.
2. **Loads** embeddings (`.npy`) and metadata (`.json`) from the `embedding-generator` component.
3. **Validates** embedding dimensions against the collection schema.
4. **Detects duplicates** and skips already-indexed vectors (idempotent).
5. **Inserts** vectors in configurable batches with retry on transient failures.
6. **Optimises** the collection (flush, compact, index rebuild).
7. **Rolls back** partially inserted data on failure.

## Output files

| File | Format | Description |
|------|--------|-------------|
| `metrics.json` | JSON | Insertion statistics and timing information |

### Metrics schema (`metrics.json`)

```json
{
  "total_vectors_inserted": 1500,
  "total_duplicates_skipped": 0,
  "insert_rate_vectors_per_sec": 2500.0,
  "total_insert_time_seconds": 0.6,
  "index_build_time_seconds": 1.2,
  "collection_total_entities": 1500,
  "flush_time_seconds": 0.3,
  "compact_time_seconds": 0.5,
  "embedding_dimension": 384,
  "batch_size": 100,
  "num_batches": 15,
  "rebuild_index": false
}
```

## Input parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embeddings_path` | `str` | *(required)* | Path to `embeddings.npy` from `embedding-generator` |
| `metadata_path` | `str` | *(required)* | Path to `metadata.json` from `embedding-generator` |
| `milvus_host` | `str` | `milvus-standalone.kubeflow.svc.cluster.local` | Milvus server hostname |
| `milvus_port` | `int` | `19530` | Milvus gRPC port |
| `collection_name` | `str` | `kubeflow_docs` | Target collection name |
| `batch_size` | `int` | `100` | Vectors per insert batch |
| `rebuild_index` | `bool` | `false` | Force rebuild the vector index |
| `output_dir` | `str` | `/tmp/outputs` | Directory for output metrics |

## Collection schema

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `VARCHAR(512)` | Primary key – unique chunk identifier |
| `embedding` | `FLOAT_VECTOR(D)` | Dense embedding vector |
| `text` | `VARCHAR(65535)` | Original chunk text |
| `metadata` | `VARCHAR(65535)` | JSON-encoded chunk metadata |

## Metrics

| Metric | Description |
|--------|-------------|
| `total_vectors_inserted` | Number of new vectors inserted |
| `total_duplicates_skipped` | Vectors skipped (already in collection) |
| `insert_rate_vectors_per_sec` | Insertion throughput |
| `total_insert_time_seconds` | Wall-clock time for all inserts |
| `index_build_time_seconds` | Time to build/rebuild index |
| `collection_total_entities` | Final entity count in collection |

## Usage

### In a KFP pipeline (v1 component YAML)

```python
import kfp
from kfp.components import load_component_from_file

embed_op   = load_component_from_file("components/embedding-generator/component.yaml")
indexer_op = load_component_from_file("components/milvus-indexer/component.yaml")

@kfp.dsl.pipeline(name="docs-indexing-pipeline")
def pipeline():
    embeddings = embed_op(
        chunks_path="/data/chunks.json",
    )
    indexer = indexer_op(
        embeddings_path=embeddings.outputs["embeddings"],
        metadata_path=embeddings.outputs["metadata"],
        milvus_host="milvus-standalone.kubeflow.svc.cluster.local",
        milvus_port=19530,
        collection_name="kubeflow_docs",
        batch_size=100,
        rebuild_index="false",
    )
```

### Standalone (Docker)

```bash
docker build -t milvus-indexer:latest .

docker run --rm \
  -v /path/to/data:/data \
  milvus-indexer:latest \
    --embeddings-path /data/embeddings.npy \
    --metadata-path /data/metadata.json \
    --milvus-host localhost \
    --milvus-port 19530 \
    --collection-name kubeflow_docs \
    --batch-size 100
```

### Local development

```bash
pip install -r requirements.txt

python component.py \
  --embeddings-path /tmp/outputs/embeddings.npy \
  --metadata-path /tmp/outputs/metadata.json \
  --milvus-host localhost \
  --milvus-port 19530
```

## Error handling

- **Connection failures** – retried up to 3 times with exponential backoff.
- **Insert failures** – individual batches are retried; on total failure, partially inserted vectors are rolled back.
- **Dimension mismatch** – detected before insertion begins; raises a clear error.
- **Duplicate vectors** – detected via primary key query; existing IDs are skipped.

## Running tests

```bash
pip install pytest
pytest components/milvus-indexer/tests/ -v
```
