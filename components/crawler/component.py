"""Kubeflow Docs Crawler - KFP Component.

Clones Kubeflow repositories, extracts markdown documentation,
downloads additional resources from kubeflow.org, and produces
a structured JSON artifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urljoin

import git
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("kubeflow-docs-crawler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_REPOS: List[str] = [
    "https://github.com/kubeflow/website",
    "https://github.com/kubeflow/manifests",
    "https://github.com/kubeflow/pipelines",
]

REPO_SECTION_MAP = {
    "website": "docs",
    "manifests": "architecture",
    "pipelines": "examples",
}

DOCS_BASE_URL = "https://www.kubeflow.org/docs/"

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds (exponential base)
REQUEST_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _retry(func, *args, retries: int = MAX_RETRIES, **kwargs) -> Any:
    """Execute *func* with exponential-backoff retry on failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = RETRY_BACKOFF ** attempt
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %ds …",
                attempt,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {retries} attempts failed for {func.__name__}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def clone_repo(url: str, dest: Path) -> Path:
    """Clone a repository with shallow depth for speed."""
    repo_name = url.rstrip("/").split("/")[-1]
    clone_path = dest / repo_name
    if clone_path.exists():
        shutil.rmtree(clone_path)
    logger.info("Cloning %s → %s", url, clone_path)
    _retry(
        git.Repo.clone_from,
        url,
        str(clone_path),
        depth=1,
        single_branch=True,
    )
    return clone_path


# ---------------------------------------------------------------------------
# Markdown extraction
# ---------------------------------------------------------------------------
def _generate_id(source: str, path: str) -> str:
    """Deterministic document id from source + path."""
    return hashlib.sha256(f"{source}:{path}".encode()).hexdigest()[:16]


def _extract_title(content: str) -> str:
    """Pull the first markdown heading from *content*."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return "Untitled"


def _infer_section(repo_name: str, rel_path: str) -> str:
    """Heuristic section label based on repo and path."""
    section = REPO_SECTION_MAP.get(repo_name, "general")
    # Refine from path components
    parts = rel_path.lower().split(os.sep)
    for keyword in ("pipelines", "notebooks", "katib", "training", "serving"):
        if keyword in parts:
            return keyword
    return section


def _infer_version(repo_path: Path) -> str:
    """Try to read a VERSION file or fall back to 'latest'."""
    for name in ("VERSION", "version.txt"):
        vfile = repo_path / name
        if vfile.is_file():
            return vfile.read_text().strip()
    return "latest"


def _canonical_url(repo_name: str, rel_path: str) -> str:
    """Best-effort canonical URL for a doc file."""
    if repo_name == "website":
        # website repo stores docs under content/en/docs/…
        marker = "content/en/docs/"
        idx = rel_path.find(marker)
        if idx != -1:
            slug = rel_path[idx + len(marker) :]
            slug = re.sub(r"(_index)?\.md$", "/", slug)
            return urljoin(DOCS_BASE_URL, slug)
    return f"https://github.com/kubeflow/{repo_name}/blob/master/{rel_path}"


def extract_markdown_files(
    repo_path: Path,
    repo_name: str,
    version: str,
) -> List[dict]:
    """Walk *repo_path* and return document dicts for every .md file."""
    documents: List[dict] = []
    docs_dirs = ["docs", "content", "examples", "doc"]
    search_roots: List[Path] = []
    for d in docs_dirs:
        candidate = repo_path / d
        if candidate.is_dir():
            search_roots.append(candidate)
    # Fallback: scan entire repo if no known docs dir found
    if not search_roots:
        search_roots = [repo_path]

    for root in search_roots:
        for md_file in root.rglob("*.md"):
            try:
                content = md_file.read_text(errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", md_file, exc)
                continue
            rel = str(md_file.relative_to(repo_path))
            doc = {
                "id": _generate_id(repo_name, rel),
                "source": f"{repo_name}/{rel}",
                "title": _extract_title(content),
                "content": content,
                "url": _canonical_url(repo_name, rel),
                "metadata": {
                    "section": _infer_section(repo_name, rel),
                    "version": version,
                },
            }
            documents.append(doc)
    logger.info(
        "Extracted %d documents from %s", len(documents), repo_name
    )
    return documents


# ---------------------------------------------------------------------------
# Web scraping (kubeflow.org)
# ---------------------------------------------------------------------------
def _fetch_page(url: str) -> str:
    """GET a page with retry + timeout."""
    resp = _retry(requests.get, url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def download_kubeflow_org_docs(
    base_url: str = DOCS_BASE_URL,
    max_pages: int = 200,
) -> List[dict]:
    """Scrape markdown-equivalent content from kubeflow.org/docs."""
    documents: List[dict] = []
    visited: set[str] = set()
    to_visit: list[str] = [base_url]

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            html = _fetch_page(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue

        soup = BeautifulSoup(html, "html.parser")

        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.body
        if main is None:
            continue

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"
        text = main.get_text(separator="\n", strip=True)

        if not text.strip():
            continue

        doc_id = _generate_id("kubeflow.org", url)
        documents.append(
            {
                "id": doc_id,
                "source": f"kubeflow.org:{url}",
                "title": title,
                "content": text,
                "url": url,
                "metadata": {
                    "section": _section_from_url(url),
                    "version": "latest",
                },
            }
        )

        # Discover links under the docs tree
        for link in main.find_all("a", href=True):
            href = urljoin(url, link["href"]).split("#")[0].split("?")[0]
            if href.startswith(base_url) and href not in visited:
                to_visit.append(href)

    logger.info(
        "Downloaded %d pages from %s", len(documents), base_url
    )
    return documents


def _section_from_url(url: str) -> str:
    """Derive a section label from a kubeflow.org URL."""
    path = url.replace(DOCS_BASE_URL, "").strip("/")
    parts = path.split("/")
    return parts[0] if parts and parts[0] else "general"


# ---------------------------------------------------------------------------
# Main pipeline logic
# ---------------------------------------------------------------------------
def crawl_kubeflow_docs(
    repos: Optional[List[str]] = None,
    output_path: str = "/tmp/documents.json",
    include_examples: bool = True,
) -> str:
    """Entry-point: clone repos, scrape web, produce documents.json.

    Parameters
    ----------
    repos:
        List of GitHub repository URLs to clone.
    output_path:
        Where to write the final JSON artifact.
    include_examples:
        If ``True`` (default), include example notebooks / scripts.

    Returns
    -------
    str
        Absolute path to the written JSON artifact.
    """
    if repos is None:
        repos = list(DEFAULT_REPOS)

    all_documents: List[dict] = []
    work_dir = Path(tempfile.mkdtemp(prefix="kf-crawler-"))
    logger.info("Working directory: %s", work_dir)

    # ------------------------------------------------------------------
    # 1. Clone & extract from each repository
    # ------------------------------------------------------------------
    for repo_url in repos:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        try:
            repo_path = clone_repo(repo_url, work_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to clone %s: %s", repo_url, exc)
            continue

        version = _infer_version(repo_path)
        docs = extract_markdown_files(repo_path, repo_name, version)

        # Optionally filter out examples
        if not include_examples and repo_name == "pipelines":
            docs = [
                d
                for d in docs
                if "examples" not in d["source"].lower()
            ]

        all_documents.extend(docs)

    # ------------------------------------------------------------------
    # 2. Download additional resources from kubeflow.org
    # ------------------------------------------------------------------
    try:
        web_docs = download_kubeflow_org_docs()
        all_documents.extend(web_docs)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Web scraping of kubeflow.org failed: %s", exc
        )

    # ------------------------------------------------------------------
    # 3. De-duplicate by id
    # ------------------------------------------------------------------
    seen_ids: set[str] = set()
    unique: List[dict] = []
    for doc in all_documents:
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            unique.append(doc)
    all_documents = unique

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {"documents": all_documents}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "Wrote %d documents to %s", len(all_documents), output
    )

    # Clean up cloned repos
    shutil.rmtree(work_dir, ignore_errors=True)

    return str(output)


# ---------------------------------------------------------------------------
# KFP v2 component wrapper
# ---------------------------------------------------------------------------
def kubeflow_docs_crawler_component(
    repos: str = "",
    output_path: str = "/tmp/outputs/documents.json",
    include_examples: bool = True,
) -> str:
    """Thin wrapper expected by the KFP component spec.

    *repos* is passed as a JSON-encoded list of strings because the
    KFP component YAML serialises complex types as strings.
    """
    repo_list: Optional[List[str]] = None
    if repos:
        repo_list = json.loads(repos)

    result_path = crawl_kubeflow_docs(
        repos=repo_list,
        output_path=output_path,
        include_examples=include_examples,
    )
    return result_path


# ---------------------------------------------------------------------------
# CLI entry-point for local testing / container execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crawl Kubeflow documentation."
    )
    parser.add_argument(
        "--repos",
        type=str,
        default="",
        help='JSON list of repo URLs, e.g. \'["https://github.com/kubeflow/website"]\'',
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/tmp/outputs/documents.json",
    )
    parser.add_argument(
        "--include-examples",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=True,
    )
    args = parser.parse_args()

    path = kubeflow_docs_crawler_component(
        repos=args.repos,
        output_path=args.output_path,
        include_examples=args.include_examples,
    )
    print(f"Output written to {path}")
