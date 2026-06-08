"""Build coarse routing units for hierarchical retrieval over the web corpus.

Inputs:
    - ``documents.jsonl`` provides source metadata and cleaned-text paths.
    - ``chunks.jsonl`` provides the final evidence chunks belonging to sources.

For each source, the routing builder creates:
    - ``source_identity``: title/URL-topic signals for precise source lookup.
    - ``source_catalog``: compact section/entity/predicate coverage metadata.
    - ``section_window``: overlapping coarse text windows linked to child chunks.
    - ``structured_fact``: dedicated routes linked to extracted fact chunks.

Hierarchical retrieval scores these smaller/coarser routing representations to
shortlist sources and candidate chunks before detailed chunk retrieval. Routing
units are not final LLM context.

The output replaces the complete ``routing_units.jsonl`` file. Rebuild its
BGE-M3 embedding cache with ``scripts/build_rag_routing_embeddings.py`` after
running this script. PDF routing units are built by
``scripts/build_rag_pdf_index.py`` instead.
"""

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
    """Define web document/chunk-to-routing-index transformation options."""
    parser = argparse.ArgumentParser(description="Build routing units for hierarchical RAG retrieval.")
    parser.add_argument(
        "--documents",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "documents.jsonl"),
        help="Input web document metadata JSONL produced by source ingestion.",
    )
    parser.add_argument(
        "--chunks",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "chunks.jsonl"),
        help="Input final evidence chunk JSONL produced by build_rag_index.py.",
    )
    parser.add_argument(
        "--routing-units",
        default=str(PROJECT_ROOT / DEFAULT_ROUTING_UNITS_PATH),
        help="Output hierarchical routing-unit JSONL.",
    )
    parser.add_argument(
        "--words-per-route",
        type=int,
        default=DEFAULT_ROUTE_WORDS,
        help="Maximum approximate words in a coarse section routing window.",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=DEFAULT_ROUTE_OVERLAP_WORDS,
        help="Words repeated between adjacent section routing windows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Documents define source boundaries and cleaned full text; chunks provide
    # the final evidence IDs that section/fact routes may reference as children.
    # Both inputs must come from the same index build to keep linkage accurate.
    documents = load_jsonl(args.documents)
    chunks = load_jsonl(args.chunks)

    # Routes are intentionally larger and fewer than final chunks. Overlapping
    # section windows preserve context while reducing the first-stage search
    # surface. Identity/catalog routes aid source selection but do not directly
    # map to LLM evidence; section/fact routes carry child_chunk_ids.
    routes = build_routing_units(
        documents,
        chunks,
        words_per_route=args.words_per_route,
        overlap_words=args.overlap_words,
    )

    # Replace the full route index. Removed sources/chunks therefore leave no
    # stale route records, but the old routing embedding cache becomes
    # incomplete or stale until rebuilt.
    write_jsonl(args.routing_units, routes)

    # Route-type counts make changes in routing coverage visible after source,
    # chunking, metadata, or structured-fact updates.
    by_type = Counter(route["route_type"] for route in routes)
    print(f"Wrote routing units: {args.routing_units} ({len(routes)} routes)")
    print(f"Original chunks: {len(chunks)}")
    for route_type, count in sorted(by_type.items()):
        print(f"  {route_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
