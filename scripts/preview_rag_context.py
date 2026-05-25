"""Preview retrieved RAG context without calling any LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import DEFAULT_RETRIEVAL_MODE, RETRIEVAL_MODES, RagRetriever
from src.rag.embeddings import (
    DEFAULT_BGE_BASE_EMBEDDINGS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_OPENAI_EMBEDDINGS_PATH,
    embedding_cache_request_for_retrieval_mode,
    ensure_embedding_cache,
)
from src.rag.chunker import load_jsonl
from src.rag.routing import DEFAULT_ROUTING_EMBEDDINGS_PATH, DEFAULT_ROUTING_UNITS_PATH


DEFAULT_QUESTION_ID = 9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview the RAG context that would be sent to the LLM.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--question", help="Question text to retrieve context for")
    source.add_argument("--id", type=int, default=DEFAULT_QUESTION_ID, help="Question id from data/qa_92.json")
    parser.add_argument(
        "--chunks",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "chunks.jsonl"),
        help="Chunk index JSONL path",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--per-source-limit", type=int, default=4)
    parser.add_argument("--retrieval-mode", choices=sorted(RETRIEVAL_MODES), default=DEFAULT_RETRIEVAL_MODE)
    parser.add_argument(
        "--embeddings",
        default=str(PROJECT_ROOT / DEFAULT_EMBEDDINGS_PATH),
        help="Embedding JSONL path for bge-m3-rrf/vector/hybrid retrieval",
    )
    parser.add_argument(
        "--bge-base-embeddings",
        default=str(PROJECT_ROOT / DEFAULT_BGE_BASE_EMBEDDINGS_PATH),
        help="Embedding JSONL path for bge-base-rrf fallback",
    )
    parser.add_argument(
        "--openai-embeddings",
        default=str(PROJECT_ROOT / DEFAULT_OPENAI_EMBEDDINGS_PATH),
        help="Embedding JSONL path for explicit openai-embedding-rrf mode",
    )
    parser.add_argument(
        "--routing-units",
        default=str(PROJECT_ROOT / DEFAULT_ROUTING_UNITS_PATH),
        help="Routing-unit JSONL path for hierarchical-bge-m3-rrf retrieval",
    )
    parser.add_argument(
        "--routing-embeddings",
        default=str(PROJECT_ROOT / DEFAULT_ROUTING_EMBEDDINGS_PATH),
        help="BGE-M3 routing embedding JSONL path for hierarchical retrieval",
    )
    parser.add_argument("--top-n-sources", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--preview-chars", type=int, default=180)
    parser.add_argument(
        "--build-missing-embeddings",
        action="store_true",
        help=(
            "Build the selected retrieval embedding cache when it is missing or incomplete. "
            "For openai-embedding-rrf this calls the OpenAI embeddings API."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    question = args.question or question_by_id(args.id)
    maybe_build_missing_embeddings(args)

    retriever = RagRetriever(
        args.chunks,
        embeddings_path=args.embeddings,
        bge_base_embeddings_path=args.bge_base_embeddings,
        openai_embeddings_path=args.openai_embeddings,
        routing_units_path=args.routing_units,
        routing_embeddings_path=args.routing_embeddings,
    )
    result = retriever.retrieve_with_details(
        question,
        top_k=args.top_k,
        per_source_limit=args.per_source_limit,
        mode=args.retrieval_mode,
        top_n_sources=args.top_n_sources,
    )
    retrieved = result.retrieved_chunks
    context = retriever.format_context(retrieved, max_chars=args.max_chars)

    print(f"Question: {question}")
    print(f"Requested mode: {result.requested_mode or args.retrieval_mode}")
    print(f"Executed mode: {result.retrieval_mode}")
    if result.embedding_provider:
        print(f"Embedding provider: {result.embedding_provider}")
        print(f"Embedding model: {result.embedding_model}")
        print(f"Embeddings path: {result.embeddings_path}")
    if result.fallback_used:
        print(f"Fallback used: {result.fallback_reason}")
    if result.routing_units_scored:
        print(f"Routing units scored: {result.routing_units_scored}/{result.routing_units_total}")
        print(f"Candidate chunks scored: {result.candidate_chunks_scored}")
        print(f"Full-chunk baseline: {result.full_chunk_count}")
        print(f"Comparison reduction vs full chunk scan: {result.comparison_reduction_percent:.2f}%")
        print(f"ColBERT candidates: {result.colbert_candidates}")
        print(f"Selected routes: {', '.join(result.selected_route_ids)}")
    print()
    if result.source_selection:
        print_source_selection(result.source_selection)
        print()
    print("Retrieved Chunk Summary")
    print("-" * 120)
    print(f"{'rank':>4}  {'source_id':<28}  {'chunk_id':<32}  {'score':>7}  {'chars':>6}  section")
    print("-" * 120)
    for rank, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        section = one_line(chunk.get("section", ""), 42)
        print(
            f"{rank:>4}  {chunk['source_id']:<28}  {chunk['chunk_id']:<32}  "
            f"{item.score:>7.2f}  {len(chunk['text']):>6}  {section}"
        )
        print(f"      preview: {one_line(chunk['text'], args.preview_chars)}")
    print("-" * 120)
    print(f"Retrieved chunks: {len(retrieved)}")
    print(f"Exact context chars: {len(context)}")
    print()
    print("Exact Prompt Context")
    print("=" * 120)
    print(context)
    return 0


def print_source_selection(source_selection) -> None:
    print("Source Selection Summary")
    print("-" * 120)
    print(f"{'rank':>4}  {'source_id':<28}  {'score':>7}  {'chunks':>6}  {'chars':>7}  title")
    print("-" * 120)
    selected = set(source_selection.selected_source_ids)
    for rank, source_score in enumerate(source_selection.scores[:12], start=1):
        marker = "*" if source_score.source_id in selected else " "
        print(
            f"{rank:>4}{marker} {source_score.source_id:<28}  {source_score.score:>7.2f}  "
            f"{source_score.chunk_count:>6}  {source_score.char_count:>7}  "
            f"{one_line(source_score.title, 40)}"
        )
        print(f"      reasons: {', '.join(source_score.reasons[:8])}")
    print("-" * 120)
    print(f"Selected sources: {', '.join(source_selection.selected_source_ids) or 'none'}")


def question_by_id(question_id: int) -> str:
    questions_path = PROJECT_ROOT / "data" / "qa_92.json"
    with questions_path.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    for question in questions:
        if int(question["id"]) == question_id:
            return question["question"]
    raise ValueError(f"Question id not found: {question_id}")


def maybe_build_missing_embeddings(args: argparse.Namespace) -> None:
    if not args.build_missing_embeddings:
        return
    request = embedding_cache_request_for_retrieval_mode(
        args.retrieval_mode,
        embeddings_path=args.embeddings,
        bge_base_embeddings_path=args.bge_base_embeddings,
        openai_embeddings_path=args.openai_embeddings,
    )
    if request is None:
        print(f"No embedding cache required for retrieval mode: {args.retrieval_mode}")
        return

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
        routing_units_path = Path(args.routing_units)
        if not routing_units_path.exists():
            raise FileNotFoundError(
                f"Routing units not found: {routing_units_path}. "
                "Run: python scripts\\build_rag_routing_index.py"
            )
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


def one_line(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
