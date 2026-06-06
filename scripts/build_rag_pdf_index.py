"""Build all non-embedding retrieval indexes for the PDF-only RAG corpus.

Input contract:
    ``documents.jsonl`` is produced by ``scripts/ingest_rag_pdfs.py``. Each row
    points to extracted, page-marked PDF text and contains PDF source metadata.

This script writes three complete JSONL artifacts:
    - ``chunks.jsonl``: page-aware text chunks plus extracted structured facts.
    - ``page_signals.jsonl``: compact entity/predicate signals for each page.
    - ``routing_units.jsonl``: coarse source/section/fact units used by
      hierarchical retrieval before detailed chunk ranking.

The artifacts serve different stages. Page signals remove clearly irrelevant
PDF pages for a query, routing units shortlist sources/regions, and chunks are
the evidence candidates ultimately sent to the LLM.

This script does not extract PDFs and does not build embeddings. After running
it, rebuild PDF chunk embeddings and BGE-M3 routing embeddings because chunk
IDs, text hashes, page signals, or route IDs may have changed.
"""

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
    """Define PDF document-to-index transformation options."""
    parser = argparse.ArgumentParser(description="Build PDF-only RAG chunk and routing indexes.")
    parser.add_argument(
        "--documents",
        type=Path,
        default=PROJECT_ROOT / PDF_DOCUMENTS_PATH,
        help="Input PDF document metadata JSONL produced by PDF ingestion.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / PDF_CHUNKS_PATH,
        help="Output page-aware retrieval chunk JSONL.",
    )
    parser.add_argument(
        "--page-signals",
        type=Path,
        default=PROJECT_ROOT / PDF_PAGE_SIGNALS_PATH,
        help="Output per-page entity/predicate signal JSONL.",
    )
    parser.add_argument(
        "--routing-units",
        type=Path,
        default=PROJECT_ROOT / PDF_ROUTING_UNITS_PATH,
        help="Output hierarchical source-routing unit JSONL.",
    )
    parser.add_argument(
        "--words-per-chunk",
        type=int,
        default=140,
        help="Maximum approximate words in a page-aware evidence chunk.",
    )
    parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        default=30,
        help="Words repeated between adjacent evidence chunks.",
    )
    parser.add_argument(
        "--words-per-route",
        type=int,
        default=DEFAULT_ROUTE_WORDS,
        help="Maximum approximate words in a coarse section routing window.",
    )
    parser.add_argument(
        "--route-overlap-words",
        type=int,
        default=DEFAULT_ROUTE_OVERLAP_WORDS,
        help="Words repeated between adjacent section routing windows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Document rows contain metadata and clean_text_path references. The
    # extracted text retains page markers created during PDF ingestion; both
    # page-aware chunking and page-signal generation depend on those markers.
    documents = load_jsonl(args.documents)

    # Build final evidence candidates. PDF chunks preserve page_start/page_end
    # and may cross a page boundary because overlap keeps nearby context intact.
    # The shared chunker also appends synthetic structured temporal/count facts.
    chunks = build_chunks_from_documents(
        documents,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.chunk_overlap_words,
    )
    write_jsonl(args.chunks, chunks)

    # Page signals are deliberately compact: each page stores detected
    # astronomy entities, predicate hints, and size. At query time they support
    # strong-signal page filtering plus neighboring-page retention without
    # embedding or ranking every page first.
    page_signals = build_page_signals_from_documents(documents)
    write_jsonl(args.page_signals, page_signals)

    # Routing units are larger/coarser than final chunks. They include source
    # identity/catalog routes, overlapping section windows, and dedicated
    # structured-fact routes. Hierarchical retrieval scores these first to
    # shortlist sources and their child chunks.
    routes = build_routing_units(
        documents,
        chunks,
        words_per_route=args.words_per_route,
        overlap_words=args.route_overlap_words,
    )
    write_jsonl(args.routing_units, routes)

    # Route counts make the build auditable and reveal unexpected changes in
    # source identity, catalog, section-window, or structured-fact coverage.
    route_counts = Counter(route["route_type"] for route in routes)
    print(f"Wrote PDF chunks: {args.chunks} ({len(chunks)} chunks)")
    print(f"Wrote PDF page signals: {args.page_signals} ({len(page_signals)} pages)")
    print(f"Wrote PDF routing units: {args.routing_units} ({len(routes)} routes)")
    for route_type, count in sorted(route_counts.items()):
        print(f"  {route_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
