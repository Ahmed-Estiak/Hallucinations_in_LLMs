"""Build chunk JSONL from ingested RAG documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import build_chunks_from_documents, load_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--words-per-chunk", type=int, default=140)
    parser.add_argument("--overlap-words", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    documents = load_jsonl(args.documents)
    chunks = build_chunks_from_documents(
        documents,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.overlap_words,
    )
    write_jsonl(args.chunks, chunks)
    print(f"Wrote chunks: {args.chunks} ({len(chunks)} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

