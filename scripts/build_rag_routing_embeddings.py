"""Build BGE-M3 embedding cache for hierarchical routing units."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import load_jsonl
from src.rag.embeddings import DEFAULT_BGE_M3_MODEL, ensure_embedding_cache
from src.rag.pdf_paths import PDF_ROUTING_EMBEDDINGS_PATH, PDF_ROUTING_UNITS_PATH
from src.rag.routing import DEFAULT_ROUTING_EMBEDDINGS_PATH, DEFAULT_ROUTING_UNITS_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build BGE-M3 embeddings for routing units.")
    parser.add_argument(
        "--source-set",
        choices=("web", "pdf"),
        default="web",
        help="Use default web or PDF-only routing paths.",
    )
    parser.add_argument(
        "--routing-units",
        default=str(PROJECT_ROOT / DEFAULT_ROUTING_UNITS_PATH),
    )
    parser.add_argument(
        "--embeddings",
        default=str(PROJECT_ROOT / DEFAULT_ROUTING_EMBEDDINGS_PATH),
    )
    parser.add_argument("--model", default=DEFAULT_BGE_M3_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    routing_units = Path(args.routing_units)
    embeddings = Path(args.embeddings)
    defaults = build_parser()
    if args.source_set == "pdf" and args.routing_units == defaults.get_default("routing_units"):
        routing_units = PROJECT_ROOT / PDF_ROUTING_UNITS_PATH
    if args.source_set == "pdf" and args.embeddings == defaults.get_default("embeddings"):
        embeddings = PROJECT_ROOT / PDF_ROUTING_EMBEDDINGS_PATH
    routes = load_jsonl(routing_units)
    result = ensure_embedding_cache(
        routes,
        path=embeddings,
        provider="bge-m3",
        model=args.model,
        batch_size=args.batch_size,
        refresh=args.refresh,
    )
    status = "built/updated" if result.built else "already complete"
    print(f"Routing embedding cache {status}: {result.path}")
    print(f"Embedding provider: {result.provider}")
    print(f"Embedding model: {result.model}")
    print(f"Routing units: {result.total_chunks}")
    print(f"Existing complete records before build: {result.existing_complete_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
