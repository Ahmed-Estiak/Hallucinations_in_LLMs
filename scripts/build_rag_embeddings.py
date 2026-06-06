"""Build or refresh cached embeddings for RAG chunk retrieval.

Workflow:
    1. Select the web or PDF chunk index.
    2. Resolve the embedding provider, model, and output cache path.
    3. Load existing cache records.
    4. Reuse records whose provider, model, and chunk-text hash still match.
    5. Embed only missing or changed chunks, unless ``--refresh`` is passed.
    6. Write one complete JSONL cache for the selected provider/source set.

Provider outputs are intentionally separate:
    - ``bge-m3`` stores dense vectors and sparse token weights.
    - ``local`` stores dense SentenceTransformers/BGE-base vectors.
    - ``openai`` stores dense vectors returned by the OpenAI embeddings API.

The script does not build chunks. Run the relevant web/PDF index builder first.
Using the OpenAI provider sends chunk text to OpenAI and incurs API usage.
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
from src.rag.pdf_paths import (
    PDF_BGE_BASE_EMBEDDINGS_PATH,
    PDF_BGE_M3_EMBEDDINGS_PATH,
    PDF_CHUNKS_PATH,
    PDF_OPENAI_EMBEDDINGS_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    """Define cache-build options shared by the web and PDF corpora."""
    parser = argparse.ArgumentParser(description="Build embedding cache for RAG chunks.")
    parser.add_argument(
        "--source-set",
        choices=("web", "pdf"),
        default="web",
        help="Use default web or PDF-only chunk/embedding paths.",
    )
    parser.add_argument(
        "--chunks",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "chunks.jsonl"),
        help=(
            "Input chunk JSONL path. An explicit path overrides the default "
            "selected by --source-set."
        ),
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
        help=(
            "Number of pending chunks sent through each model/API batch. "
            "Larger local batches use more GPU/CPU memory."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute all records even when matching cached embeddings already exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Provider aliases are normalized before resolving defaults because the
    # provider and model names form part of every record's cache identity.
    provider = normalize_provider(args.provider)
    model = args.model or default_model_for_provider(provider)

    # Path precedence:
    #   explicit --chunks/--embeddings
    #   -> defaults for --source-set pdf
    #   -> defaults for the web corpus
    #
    # Provider caches never share a file: their vector formats, dimensions,
    # models, and retrieval behavior are not interchangeable.
    chunks_path = Path(args.chunks)
    embeddings_path = args.embeddings or str(PROJECT_ROOT / default_embeddings_path_for_provider(provider))
    if args.source_set == "pdf" and args.chunks == build_parser().get_default("chunks"):
        chunks_path = PROJECT_ROOT / PDF_CHUNKS_PATH
    if args.source_set == "pdf" and args.embeddings is None:
        if provider == "bge-m3":
            embeddings_path = str(PROJECT_ROOT / PDF_BGE_M3_EMBEDDINGS_PATH)
        elif provider == "local":
            embeddings_path = str(PROJECT_ROOT / PDF_BGE_BASE_EMBEDDINGS_PATH)
        elif provider == "openai":
            embeddings_path = str(PROJECT_ROOT / PDF_OPENAI_EMBEDDINGS_PATH)

    # The chunk index is the canonical embedding input. Cache records include a
    # text hash, so editing source text or rebuilding chunks invalidates only
    # the affected records instead of forcing an unconditional full rebuild.
    chunks = load_jsonl(chunks_path)

    # Missing cache files load as an empty record collection. Existing records
    # are candidates for reuse; build_chunk_embedding_records validates each
    # record against the requested provider, model, and current chunk content.
    existing_records = load_embedding_records(embeddings_path)

    # Only this call performs model inference or an OpenAI embeddings request.
    # ``--refresh`` deliberately bypasses otherwise valid cached records.
    records = build_chunk_embedding_records(
        chunks,
        existing_records=existing_records,
        provider=provider,
        model=model,
        batch_size=args.batch_size,
        refresh=args.refresh,
    )

    # Write the complete resolved cache, not merely the newly embedded subset.
    # This keeps the JSONL aligned with the current chunk index and removes
    # stale records belonging to chunks that no longer exist.
    write_embedding_records(embeddings_path, records)
    print(f"Embedding provider: {provider}")
    print(f"Embedding model: {model}")
    print(f"Wrote embeddings: {embeddings_path} ({len(records)} chunks)")
    print(f"Existing cache records before build: {len(existing_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
