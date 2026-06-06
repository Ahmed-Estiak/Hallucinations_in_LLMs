"""Build the retrieval chunk index from already-ingested RAG documents.

Input contract:
    ``documents.jsonl`` is produced by source ingestion. Each row describes one
    source and points to its already-cleaned text through ``clean_text_path``.
    This script does not fetch URLs or clean HTML.

Workflow:
    1. Load document metadata from JSONL.
    2. Read each document's cleaned text file.
    3. Split normal text into section-aware overlapping chunks.
    4. Detect entities and predicate hints used by lexical retrieval.
    5. Extract supported structured temporal/count facts from the full text.
    6. Write one canonical ``chunks.jsonl`` retrieval index.

The output contains text chunks and synthetic ``structured_fact`` chunks.
Retrievers, embedding builders, and routing-index builders all depend on this
file. Rebuilding it can change chunk IDs or text hashes, so dependent embedding
and routing caches should normally be rebuilt afterward.

The default paths target the web corpus. PDF ingestion uses
``scripts/build_rag_pdf_index.py`` because it also builds PDF-specific indexes
and diagnostics, although the shared chunker itself supports PDF documents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import build_chunks_from_documents, load_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Define the document-to-chunk transformation options."""
    parser = argparse.ArgumentParser(description="Build a chunk index from RAG documents.")
    parser.add_argument(
        "--documents",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "documents.jsonl"),
        help="Input document metadata JSONL produced by source ingestion.",
    )
    parser.add_argument(
        "--chunks",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "chunks.jsonl"),
        help="Output canonical retrieval chunk JSONL.",
    )
    parser.add_argument(
        "--words-per-chunk",
        type=int,
        default=140,
        help=(
            "Maximum approximate word window for normal text chunks. "
            "Section boundaries may produce shorter chunks."
        ),
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

    # Document rows contain metadata only; build_chunks_from_documents opens
    # each row's clean_text_path. A missing or stale cleaned-text file therefore
    # fails here instead of silently creating an incomplete retrieval index.
    documents = load_jsonl(args.documents)

    # Normal web text is first divided by detected headings, then each section
    # is split into word windows. Adjacent windows repeat ``overlap_words`` so a
    # fact crossing a window boundary remains visible in at least one chunk.
    #
    # Every normal chunk receives retrieval metadata such as source identity,
    # section, detected astronomy entities, predicate hints, trust level, and a
    # token estimate. The shared chunker also appends synthetic structured_fact
    # chunks for temporal/count claims it can extract from the full source.
    chunks = build_chunks_from_documents(
        documents,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.overlap_words,
    )

    # write_jsonl creates the parent directory when needed and replaces the
    # complete index. It does not merge with the previous chunks file; stale
    # chunks disappear naturally when their document/source is removed.
    write_jsonl(args.chunks, chunks)
    print(f"Wrote chunks: {args.chunks} ({len(chunks)} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
