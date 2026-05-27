"""Build or refresh cached embeddings for RAG chunk retrieval.

This script is intentionally cache-first. It reads the current chunk index,
reuses embedding records whose provider/model/text hash still match, and only
embeds new or changed chunks unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import load_jsonl
from src.rag.embeddings import (
    DEFAULT_BGE_M3_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    build_chunk_embedding_records,
    default_embeddings_path_for_provider,
    default_model_for_provider,
    load_embedding_records,
    normalize_provider,
    write_embedding_records,
)


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI used by both manual rebuilds and fallback cache setup."""
    parser = argparse.ArgumentParser(description="Build embedding cache for RAG chunks.")
    parser.add_argument(
        "--chunks",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "chunks.jsonl"),
        help="Input chunk JSONL path",
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help=(
            "Output embedding JSONL path. If omitted, the provider-specific "
            "default cache path is used."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("bge-m3", "local", "openai"),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=(
            "Embedding provider. bge-m3 uses FlagEmbedding dense+sparse; "
            "local uses SentenceTransformers; openai uses the OpenAI embeddings API."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Embedding model. Defaults to "
            f"{DEFAULT_BGE_M3_MODEL} for bge-m3, "
            f"{DEFAULT_LOCAL_EMBEDDING_MODEL} for local, and "
            f"{DEFAULT_OPENAI_EMBEDDING_MODEL} for openai."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed per model/API batch.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute all records even when matching cached embeddings already exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Normalize provider/model early so the cache key is stable. Each provider
    # writes to a separate default file because dense-only and dense+sparse
    # records are not interchangeable.
    provider = normalize_provider(args.provider)
    model = args.model or default_model_for_provider(provider)
    embeddings_path = args.embeddings or str(PROJECT_ROOT / default_embeddings_path_for_provider(provider))

    # chunks.jsonl is the canonical retrieval surface. If source cleaning or
    # chunking changes, each chunk's embedding text hash changes and only those
    # affected records are rebuilt.
    chunks = load_jsonl(args.chunks)
    existing_records = load_embedding_records(embeddings_path)

    # build_chunk_embedding_records preserves existing valid records and embeds
    # only pending chunks. For bge-m3 it stores both dense vectors and sparse
    # token weights; for local/openai it stores dense vectors only.
    records = build_chunk_embedding_records(
        chunks,
        existing_records=existing_records,
        provider=provider,
        model=model,
        batch_size=args.batch_size,
        refresh=args.refresh,
    )
    write_embedding_records(embeddings_path, records)
    print(f"Embedding provider: {provider}")
    print(f"Embedding model: {model}")
    print(f"Wrote embeddings: {embeddings_path} ({len(records)} chunks)")
    print(f"Existing cache records before build: {len(existing_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
