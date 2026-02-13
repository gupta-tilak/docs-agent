"""Milvus vector store SDK for the docs-agent project.

Provides a high-level, type-safe Python interface for inserting, searching,
and managing document-chunk embeddings stored in a Milvus collection.

Quick start::

    from vector_store import MilvusVectorStore, VectorStoreConfig, MilvusConnectionConfig

    store = MilvusVectorStore(
        host="localhost",
        port=19530,
        collection_name="kubeflow_docs",
    )

    results = store.search_by_text("How do I deploy a pipeline?")
    for r in results:
        print(r["text"], r["score"])

    store.close()
"""

from vector_store.config import (
    EmbeddingConfig,
    IndexConfig,
    MilvusConnectionConfig,
    RetryConfig,
    SchemaConfig,
    SearchConfig,
    VectorStoreConfig,
)
from vector_store.milvus_client import MilvusVectorStore

__all__ = [
    "MilvusVectorStore",
    "VectorStoreConfig",
    "MilvusConnectionConfig",
    "RetryConfig",
    "SearchConfig",
    "EmbeddingConfig",
    "SchemaConfig",
    "IndexConfig",
]
