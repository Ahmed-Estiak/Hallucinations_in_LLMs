"""Build the retrieval chunk index from already-ingested RAG documents.

The ingestion step writes documents.jsonl with clean_text_path metadata. This
script reads those cleaned documents, splits them into overlapping text chunks,
adds any structured facts extracted from the full source text, and writes the
canonical chunks.jsonl consumed by retrievers and embedding builders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import build_chunks_from_documents, load_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Define chunk-index build options without coupling to source ingestion."""
    parser = argparse.ArgumentParser(description="Build a chunk index from RAG documents.")
    parser.add_argument(
        "--documents",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "documents.jsonl"),
        help="Input document metadata JSONL",
    )
    parser.add_argument(
        "--chunks",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "chunks.jsonl"),
        help="Output chunk JSONL",
    )
    parser.add_argument(
        "--words-per-chunk",
        type=int,
        default=140,
        help="Approximate word window size for normal text chunks.",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=30,
        help="Words repeated between adjacent chunks to preserve local context.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # documents.jsonl is produced by the ingestion/cleaning pipeline. Each row
    # points to a cleaned text file plus source metadata such as URL, title, and
    # trust level.
    documents = load_jsonl(args.documents)

    # build_chunks_from_documents creates two kinds of retrieval records:
    # 1. section-aware overlapping text chunks for general semantic retrieval;
    # 2. structured_fact chunks for extracted temporal/current moon-count facts.
    chunks = build_chunks_from_documents(
        documents,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.overlap_words,
    )

    # chunks.jsonl is the downstream contract. Changing it usually means the
    # embedding caches and hierarchical routing indexes should be rebuilt.
    write_jsonl(args.chunks, chunks)
    print(f"Wrote chunks: {args.chunks} ({len(chunks)} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
