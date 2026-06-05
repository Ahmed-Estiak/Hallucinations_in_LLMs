"""Main entry point for the RAG+LLM benchmark.

Usage:
    python main_rag.py
"""

from __future__ import annotations

import argparse

from src.rag.chunker import load_jsonl
from src.rag.embeddings import (
    DEFAULT_BGE_BASE_EMBEDDINGS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_OPENAI_EMBEDDINGS_PATH,
    embedding_cache_request_for_retrieval_mode,
    ensure_embedding_cache,
)
from src.rag.retriever import DEFAULT_RETRIEVAL_MODE, RETRIEVAL_MODES
from src.rag.routing import DEFAULT_ROUTING_EMBEDDINGS_PATH, DEFAULT_ROUTING_UNITS_PATH
from src.rag_runner import run_rag_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG+LLM benchmark.")
    parser.add_argument("--ids", nargs="+", type=int, help="Question ids to run")
    parser.add_argument("--chunks", default="data/rag_sources/rag_index/chunks.jsonl")
    parser.add_argument("--retrieval-mode", choices=sorted(RETRIEVAL_MODES), default=DEFAULT_RETRIEVAL_MODE)
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS_PATH))
    parser.add_argument("--bge-base-embeddings", default=str(DEFAULT_BGE_BASE_EMBEDDINGS_PATH))
    parser.add_argument("--openai-embeddings", default=str(DEFAULT_OPENAI_EMBEDDINGS_PATH))
    parser.add_argument("--routing-units", default=str(DEFAULT_ROUTING_UNITS_PATH))
    parser.add_argument("--routing-embeddings", default=str(DEFAULT_ROUTING_EMBEDDINGS_PATH))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--per-source-limit", type=int, default=4)
    parser.add_argument("--top-n-sources", type=int, default=12)
    parser.add_argument("--output", default="reports/final/results_rag_llm.csv")
    parser.add_argument(
        "--build-missing-embeddings",
        action="store_true",
        help=(
            "Build the selected retrieval embedding cache when it is missing or incomplete. "
            "For openai-embedding-rrf this calls the OpenAI embeddings API before the LLM run."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.build_missing_embeddings:
        request = embedding_cache_request_for_retrieval_mode(
            args.retrieval_mode,
            embeddings_path=args.embeddings,
            bge_base_embeddings_path=args.bge_base_embeddings,
            openai_embeddings_path=args.openai_embeddings,
        )
        if request is None:
            print(f"No embedding cache required for retrieval mode: {args.retrieval_mode}")
        else:
            provider, path = request
            if provider == "openai":
                print("Building missing OpenAI embedding cache. This calls the OpenAI embeddings API.")
            else:
                print(f"Ensuring {provider} embedding cache: {path}")
            result = ensure_embedding_cache(
                load_jsonl(args.chunks),
                path=path,
                provider=provider,
                batch_size=args.embedding_batch_size,
            )
            status = "built/updated" if result.built else "already complete"
            print(
                f"Embedding cache {status}: {result.path} "
                f"({result.total_chunks} chunks, provider={result.provider}, model={result.model})"
            )
            if args.retrieval_mode == "hierarchical-bge-m3-rrf":
                routing_units_path = args.routing_units
                routing_result = ensure_embedding_cache(
                    load_jsonl(routing_units_path),
                    path=args.routing_embeddings,
                    provider="bge-m3",
                    batch_size=args.embedding_batch_size,
                )
                routing_status = "built/updated" if routing_result.built else "already complete"
                print(
                    f"Routing embedding cache {routing_status}: {routing_result.path} "
                    f"({routing_result.total_chunks} routes)"
                )
    run_rag_benchmark(
        question_ids=args.ids,
        chunks_path=args.chunks,
        output_path=args.output,
        embeddings_path=args.embeddings,
        bge_base_embeddings_path=args.bge_base_embeddings,
        openai_embeddings_path=args.openai_embeddings,
        routing_units_path=args.routing_units,
        routing_embeddings_path=args.routing_embeddings,
        top_k=args.top_k,
        per_source_limit=args.per_source_limit,
        retrieval_mode=args.retrieval_mode,
        top_n_sources=args.top_n_sources,
    )
