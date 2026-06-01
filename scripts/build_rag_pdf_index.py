"""Build the PDF-only RAG chunk and routing indexes."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import build_chunks_from_documents, load_jsonl, write_jsonl
from src.rag.pdf_page_filter import build_page_signals_from_documents
from src.rag.pdf_paths import (
    PDF_CHUNKS_PATH,
    PDF_DOCUMENTS_PATH,
    PDF_PAGE_SIGNALS_PATH,
    PDF_ROUTING_UNITS_PATH,
)
from src.rag.routing import (
    DEFAULT_ROUTE_OVERLAP_WORDS,
    DEFAULT_ROUTE_WORDS,
    build_routing_units,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PDF-only RAG chunk and routing indexes.")
    parser.add_argument("--documents", type=Path, default=PROJECT_ROOT / PDF_DOCUMENTS_PATH)
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / PDF_CHUNKS_PATH)
    parser.add_argument("--page-signals", type=Path, default=PROJECT_ROOT / PDF_PAGE_SIGNALS_PATH)
    parser.add_argument("--routing-units", type=Path, default=PROJECT_ROOT / PDF_ROUTING_UNITS_PATH)
    parser.add_argument("--words-per-chunk", type=int, default=140)
    parser.add_argument("--chunk-overlap-words", type=int, default=30)
    parser.add_argument("--words-per-route", type=int, default=DEFAULT_ROUTE_WORDS)
    parser.add_argument("--route-overlap-words", type=int, default=DEFAULT_ROUTE_OVERLAP_WORDS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    documents = load_jsonl(args.documents)
    chunks = build_chunks_from_documents(
        documents,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.chunk_overlap_words,
    )
    write_jsonl(args.chunks, chunks)
    page_signals = build_page_signals_from_documents(documents)
    write_jsonl(args.page_signals, page_signals)
    routes = build_routing_units(
        documents,
        chunks,
        words_per_route=args.words_per_route,
        overlap_words=args.route_overlap_words,
    )
    write_jsonl(args.routing_units, routes)

    route_counts = Counter(route["route_type"] for route in routes)
    print(f"Wrote PDF chunks: {args.chunks} ({len(chunks)} chunks)")
    print(f"Wrote PDF page signals: {args.page_signals} ({len(page_signals)} pages)")
    print(f"Wrote PDF routing units: {args.routing_units} ({len(routes)} routes)")
    for route_type, count in sorted(route_counts.items()):
        print(f"  {route_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
