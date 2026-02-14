#!/usr/bin/env python3
"""Submit the kubeflow-docs-ingestion-pipeline to a KFP cluster.

Reads runtime configuration from ``pipeline_config.yaml`` (or CLI
overrides) and creates a pipeline run via the KFP SDK v2 client.

Usage:
    # Submit with defaults from pipeline_config.yaml
    python run_pipeline.py

    # Override specific parameters
    python run_pipeline.py \\
        --host http://localhost:8083 \\
        --experiment my-experiment \\
        --chunk-size 256 \\
        --embedding-model sentence-transformers/all-mpnet-base-v2

    # Compile only (no submission)
    python run_pipeline.py --compile-only --output pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load pipeline_config.yaml and return a flat parameter dict."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        print(f"[warn] Config file not found at {path}; using built-in defaults.")
        return {}
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    return raw or {}


def merge_params(config: Dict[str, Any], cli: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config with CLI overrides (CLI wins)."""
    pipeline_cfg = config.get("pipeline_parameters", {})
    runtime_cfg = config.get("runtime", {})

    params: Dict[str, Any] = {
        "repos": json.dumps(pipeline_cfg.get("repos", [
            "https://github.com/kubeflow/website",
            "https://github.com/kubeflow/manifests",
            "https://github.com/kubeflow/pipelines",
        ])),
        "chunk_size": pipeline_cfg.get("chunk_size", 512),
        "chunk_overlap": pipeline_cfg.get("chunk_overlap", 128),
        "chunk_strategy": pipeline_cfg.get("chunk_strategy", "semantic"),
        "embedding_model": pipeline_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "batch_size": pipeline_cfg.get("batch_size", 32),
        "device": pipeline_cfg.get("device", "cpu"),
        "include_examples": pipeline_cfg.get("include_examples", True),
    }

    # CLI overrides
    if cli.repos:
        params["repos"] = json.dumps(cli.repos)
    if cli.chunk_size is not None:
        params["chunk_size"] = cli.chunk_size
    if cli.chunk_overlap is not None:
        params["chunk_overlap"] = cli.chunk_overlap
    if cli.chunk_strategy:
        params["chunk_strategy"] = cli.chunk_strategy
    if cli.embedding_model:
        params["embedding_model"] = cli.embedding_model
    if cli.batch_size is not None:
        params["batch_size"] = cli.batch_size
    if cli.device:
        params["device"] = cli.device

    return params, runtime_cfg


# ---------------------------------------------------------------------------
# Pipeline submission
# ---------------------------------------------------------------------------
def compile_pipeline(output_path: str) -> None:
    """Compile pipeline to YAML without submitting."""
    from kfp import compiler

    from ingestion_pipeline import kubeflow_docs_ingestion_pipeline

    compiler.Compiler().compile(
        pipeline_func=kubeflow_docs_ingestion_pipeline,
        package_path=output_path,
    )
    print(f"[ok] Pipeline compiled → {output_path}")


def submit_pipeline(
    host: str,
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    namespace: Optional[str] = None,
) -> None:
    """Create a pipeline run on the KFP cluster."""
    import kfp

    from ingestion_pipeline import kubeflow_docs_ingestion_pipeline

    client = kfp.Client(host=host, namespace=namespace)

    # Ensure experiment exists
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
    except Exception:
        experiment = client.create_experiment(name=experiment_name)

    print(f"[info] Submitting run '{run_name}' to experiment '{experiment_name}'")
    print(f"[info] KFP host: {host}")
    print(f"[info] Parameters:")
    for k, v in params.items():
        display = v if len(str(v)) < 80 else str(v)[:77] + "..."
        print(f"         {k}: {display}")

    run = client.create_run_from_pipeline_func(
        pipeline_func=kubeflow_docs_ingestion_pipeline,
        experiment_name=experiment_name,
        run_name=run_name,
        arguments=params,
        enable_caching=True,
    )

    print(f"\n[ok] Run created successfully!")
    print(f"     Run ID   : {run.run_id}")
    print(f"     Run URL  : {host}/#/runs/details/{run.run_id}")
    print(f"\nMonitor progress in the Kubeflow Pipelines UI.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Submit the kubeflow-docs-ingestion-pipeline to a KFP cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Submit with YAML defaults
  python run_pipeline.py --host http://localhost:8083

  # Override repos and chunk size
  python run_pipeline.py --host http://localhost:8083 \\
      --repos https://github.com/kubeflow/website \\
      --chunk-size 256

  # Compile only
  python run_pipeline.py --compile-only
""",
    )
    p.add_argument(
        "--host",
        type=str,
        help="KFP API endpoint (e.g. http://localhost:8083). "
        "Reads from pipeline_config.yaml if omitted.",
    )
    p.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Kubernetes namespace for the pipeline run.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: from config or 'docs-ingestion').",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name. Auto-generated with timestamp if omitted.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline_config.yaml.",
    )

    # Pipeline parameter overrides
    g = p.add_argument_group("Pipeline parameter overrides")
    g.add_argument("--repos", nargs="+", help="Repository URLs to crawl.")
    g.add_argument("--chunk-size", type=int, dest="chunk_size")
    g.add_argument("--chunk-overlap", type=int, dest="chunk_overlap")
    g.add_argument(
        "--chunk-strategy",
        type=str,
        dest="chunk_strategy",
        choices=["fixed", "semantic", "sentence"],
    )
    g.add_argument("--embedding-model", type=str, dest="embedding_model")
    g.add_argument("--batch-size", type=int, dest="batch_size")
    g.add_argument("--device", type=str, choices=["cpu", "cuda"])

    # Compile mode
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile pipeline to YAML without submitting.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="pipelines/kubeflow_docs_ingestion_pipeline.yaml",
        help="Output path for compiled YAML (used with --compile-only).",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # -- compile-only mode --
    if args.compile_only:
        compile_pipeline(args.output)
        return

    # -- load config & merge --
    config = load_config(args.config)
    params, runtime_cfg = merge_params(config, args)

    host = args.host or runtime_cfg.get("kfp_host", "http://localhost:8083")
    namespace = args.namespace or runtime_cfg.get("namespace")
    experiment = args.experiment or runtime_cfg.get(
        "experiment_name", "docs-ingestion"
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"docs-ingestion-{ts}"

    # -- submit --
    try:
        submit_pipeline(
            host=host,
            experiment_name=experiment,
            run_name=run_name,
            params=params,
            namespace=namespace,
        )
    except Exception as exc:
        print(f"\n[error] Failed to submit pipeline: {exc}", file=sys.stderr)
        print(
            "[hint] Make sure the KFP API is reachable at the configured host "
            "and that you have port-forwarded if running locally:\n"
            "       kubectl port-forward --address 127.0.0.1 svc/ml-pipeline-ui 8083:80 -n kubeflow",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
