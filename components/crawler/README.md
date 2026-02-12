# kubeflow-docs-crawler

A Kubeflow Pipelines (KFP) component that crawls Kubeflow documentation from GitHub repositories and [kubeflow.org](https://www.kubeflow.org/docs/).

## What it does

1. **Clones** specific Kubeflow GitHub repositories (shallow clone for speed):
   - `kubeflow/website` — official documentation
   - `kubeflow/manifests` — architecture / deployment manifests
   - `kubeflow/pipelines` — pipeline examples and SDK docs
2. **Extracts** every Markdown file from each repo's docs directories.
3. **Scrapes** additional pages from `kubeflow.org/docs` following internal links.
4. **De-duplicates** documents by a deterministic content ID.
5. **Writes** a single `documents.json` artifact with the schema below.

## Output schema

```json
{
  "documents": [
    {
      "id": "a1b2c3d4e5f67890",
      "source": "website/content/en/docs/pipelines/overview.md",
      "title": "Pipelines Overview",
      "content": "# Pipelines Overview\n...",
      "url": "https://www.kubeflow.org/docs/pipelines/overview/",
      "metadata": {
        "section": "pipelines",
        "version": "latest"
      }
    }
  ]
}
```

## Input parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repos` | `List[str]` (JSON string) | All three default repos | GitHub repository URLs to clone |
| `output_path` | `str` | `/tmp/outputs/documents.json` | File path for the output artifact |
| `include_examples` | `bool` | `true` | Include example notebooks/scripts from the pipelines repo |

## Usage

### In a KFP pipeline (v1 component YAML)

```python
import kfp
from kfp.components import load_component_from_file

crawler_op = load_component_from_file("components/crawler/component.yaml")

@kfp.dsl.pipeline(name="kubeflow-docs-ingestion")
def docs_pipeline():
    crawl_task = crawler_op(
        repos='["https://github.com/kubeflow/website"]',
        output_path="/tmp/outputs/documents.json",
        include_examples="true",
    )
```

### Local execution (CLI)

```bash
# Install dependencies
pip install -r components/crawler/requirements.txt

# Run with defaults (all repos)
python components/crawler/component.py

# Run with custom repos
python components/crawler/component.py \
  --repos '["https://github.com/kubeflow/website"]' \
  --output-path ./output/documents.json \
  --include-examples true
```

### Docker

```bash
# Build
docker build -t kubeflow-docs-crawler:latest components/crawler/

# Run
docker run --rm \
  -v "$(pwd)/output:/tmp/outputs" \
  kubeflow-docs-crawler:latest \
  --output-path /tmp/outputs/documents.json
```

## Error handling

- **Network failures** — all HTTP requests and `git clone` operations use exponential-backoff retry (3 attempts, 2/4/8 s delays).
- **Unreadable files** — skipped with a warning; the rest of the repo is still processed.
- **Failed repos** — logged and skipped; other repos continue normally.
- **Web scraping failures** — individual pages that fail are skipped; the overall scrape still completes.

## Dependencies

| Package | Purpose |
|---------|---------|
| `gitpython` | Clone GitHub repositories |
| `requests` | HTTP requests to kubeflow.org |
| `beautifulsoup4` | Parse HTML pages |

## Project structure

```
components/crawler/
├── component.py        # Main crawler logic + CLI entry-point
├── component.yaml      # KFP component specification
├── Dockerfile          # Container image definition (python:3.11-slim)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
