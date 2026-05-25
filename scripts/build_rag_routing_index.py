"""Build coarse routing units for hierarchical RAG retrieval."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import load_jsonl, write_jsonl
from src.rag.routing import (
    DEFAULT_ROUTE_OVERLAP_WORDS,
    DEFAULT_ROUTE_WORDS,
    DEFAULT_ROUTING_UNITS_PATH,
    build_routing_units,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build routing units for hierarchical RAG retrieval.")
    parser.add_argument(
        "--documents",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "documents.jsonl"),
    )
    parser.add_argument(
        "--chunks",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "chunks.jsonl"),
    )
    parser.add_argument(
        "--routing-units",
        default=str(PROJECT_ROOT / DEFAULT_ROUTING_UNITS_PATH),
    )
    parser.add_argument("--words-per-route", type=int, default=DEFAULT_ROUTE_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_ROUTE_OVERLAP_WORDS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    documents = load_jsonl(args.documents)
    chunks = load_jsonl(args.chunks)
    routes = build_routing_units(
        documents,
        chunks,
        words_per_route=args.words_per_route,
        overlap_words=args.overlap_words,
    )
    write_jsonl(args.routing_units, routes)
    by_type = Counter(route["route_type"] for route in routes)
    print(f"Wrote routing units: {args.routing_units} ({len(routes)} routes)")
    print(f"Original chunks: {len(chunks)}")
    for route_type, count in sorted(by_type.items()):
        print(f"  {route_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
