# Document Chunker – KFP Component

Kubeflow Pipeline component that splits documents from the crawler into
overlapping, token-counted chunks for downstream embedding and retrieval.

## Chunking Strategies

| Strategy   | Description |
|------------|-------------|
| `fixed`    | Fixed-size token windows with configurable overlap. |
| `semantic` | Split on markdown headers/sections first, then sub-split large sections. |
| `sentence` | Split on sentence boundaries, grouping sentences up to the token limit. |

## Inputs

| Parameter        | Type   | Default                    | Description |
|------------------|--------|----------------------------|-------------|
| `documents_path` | `str`  | *(required)*               | Path to the crawler output JSON |
| `chunk_size`     | `int`  | `512`                      | Target tokens per chunk |
| `overlap`        | `int`  | `128`                      | Overlap tokens between chunks |
| `strategy`       | `str`  | `semantic`                 | Chunking strategy |
| `output_path`    | `str`  | `/tmp/outputs/chunks.json` | Output file path |

## Output Schema

```json
{
  "chunks": [
    {
      "chunk_id": "doc1_chunk0",
      "text": "chunk content...",
      "tokens": 512,
      "metadata": {
        "source_doc": "doc1",
        "position": 0,
        "parent_title": "...",
        "section": "...",
        "start_char": 0,
        "end_char": 1024
      }
    }
  ],
  "metrics": {
    "total_chunks": 42,
    "average_chunk_size": 487.3,
    "total_tokens": 20467,
    "documents_processed": 5,
    "strategy": "semantic",
    "elapsed_seconds": 1.234
  }
}
```

## Local Usage

```bash
python component.py \
  --documents-path /path/to/documents.json \
  --output-path /tmp/outputs/chunks.json \
  --chunk-size 512 \
  --overlap 128 \
  --strategy semantic
```

## Docker

```bash
docker build -t document-chunker:latest .
docker run -v /data:/data document-chunker:latest \
  --documents-path /data/documents.json \
  --output-path /data/chunks.json
```

## Tests

```bash
pip install pytest tiktoken langchain langchain-text-splitters
pytest tests/ -v
```
