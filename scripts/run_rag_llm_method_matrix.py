"""Run selected RAG retrieval methods through OpenAI and Gemini.

This is intentionally separate from ``src/rag_runner.py`` because this report
compares multiple retrieval methods over the same small question set. It calls
LLMs, so running it has provider cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import evaluate_answer
from src.rag.pdf_paths import (
    PDF_BGE_BASE_EMBEDDINGS_PATH,
    PDF_BGE_M3_EMBEDDINGS_PATH,
    PDF_CHUNKS_PATH,
    PDF_DOCUMENTS_PATH,
    PDF_OPENAI_EMBEDDINGS_PATH,
    PDF_ROUTING_EMBEDDINGS_PATH,
    PDF_ROUTING_UNITS_PATH,
)
from src.rag.retriever import RagRetriever
from src.rag_models import ask_gemini_with_rag, ask_openai_with_rag, build_rag_prompt


DEFAULT_QUESTION_IDS = [9, 11, 15]
DEFAULT_METHODS = [
    "hierarchical-bge-m3-rrf",
    "bge-m3-rrf",
    "bge-base-rrf",
    "openai-embedding-rrf",
    "auto-source",
    "global",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "rag_audit" / "rag_llm_method_matrix.csv"


def load_questions(path: Path, ids: list[int]) -> list[dict[str, Any]]:
    selected = set(ids)
    with path.open("r", encoding="utf-8") as handle:
        return [item for item in json.load(handle) if int(item["id"]) in selected]


def expected_answer(question: dict[str, Any]) -> str:
    spec = question.get("answer_spec", {})
    if "value" in spec:
        value = spec["value"]
    else:
        value = spec.get("fields", "")
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)


def token_count_estimate(text: str) -> int:
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return len(pieces)


def call_provider(provider: str, question: str, context: str) -> str:
    if provider == "openai":
        return ask_openai_with_rag(question, context)
    if provider == "gemini":
        return ask_gemini_with_rag(question, context)
    raise ValueError(f"Unknown provider: {provider}")


def run_matrix(
    *,
    questions: list[dict[str, Any]],
    methods: list[str],
    providers: list[str],
    output: Path,
    max_chars: int,
    source_set: str,
    enable_temporal_first: bool,
) -> list[dict[str, Any]]:
    retriever_init_started = time.perf_counter()
    retriever = build_retriever(source_set, enable_temporal_first=enable_temporal_first)
    retriever_init_seconds = time.perf_counter() - retriever_init_started
    rows: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for method in methods:
        print(f"\n{method}", flush=True)
        for question_row in questions:
            question_id = int(question_row["id"])
            question = question_row["question"]
            truth = expected_answer(question_row)
            row_started = time.perf_counter()
            print(f"  Q{question_id}: retrieving...", flush=True)
            retrieval_started = time.perf_counter()
            retrieval_result = retriever.retrieve_with_details(
                question,
                mode=method,
                top_n_sources=3 if source_set == "pdf" else 12,
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            context_started = time.perf_counter()
            context = retriever.format_context_for_llm(
                retrieval_result.retrieved_chunks,
                max_chars=max_chars,
            )
            context_format_seconds = time.perf_counter() - context_started
            prompt_started = time.perf_counter()
            prompt = build_rag_prompt(question, context)
            prompt_build_seconds = time.perf_counter() - prompt_started
            sufficiency_started = time.perf_counter()
            context_sufficient = (
                bool(retrieval_result.retrieved_chunks)
                and len(context) >= 200
                and retrieval_result.temporal_evidence_status not in {"insufficient", "conflict"}
                and retrieval_result.current_evidence_status not in {"insufficient", "conflict"}
            )
            context_sufficiency_seconds = time.perf_counter() - sufficiency_started

            answers: dict[str, str] = {}
            evals: dict[str, dict[str, Any]] = {}
            provider_call_seconds: dict[str, float] = {}
            provider_eval_seconds: dict[str, float] = {}
            for provider in providers:
                print(f"    {provider}: calling...", flush=True)
                provider_started = time.perf_counter()
                if context_sufficient:
                    try:
                        answer = call_provider(provider, question, context)
                    except Exception as exc:
                        answer = f"ERROR: {type(exc).__name__}: {exc}"
                else:
                    answer = "insufficient context"
                provider_call_seconds[provider] = time.perf_counter() - provider_started
                answers[provider] = answer
                eval_started = time.perf_counter()
                evals[provider] = evaluate_answer(question_row, answer)
                provider_eval_seconds[provider] = time.perf_counter() - eval_started

            fallback_note = (
                f", fallback: {retrieval_result.fallback_reason}"
                if retrieval_result.fallback_used
                else ""
            )
            print(
                "    "
                + ", ".join(
                    f"{provider}={answers[provider]} ({evals[provider]['reason']})"
                    for provider in providers
                )
                + ", "
                f"executed={retrieval_result.retrieval_mode}{fallback_note}"
                ,
                flush=True,
            )

            rows.append(
                {
                    "generated_at_utc": generated_at,
                    "question_id": question_id,
                    "question": question,
                    "expected_answer": truth,
                    "requested_method": method,
                    "executed_method": retrieval_result.retrieval_mode,
                    "temporal_first_enabled": enable_temporal_first,
                    "fallback_used": retrieval_result.fallback_used,
                    "fallback_reason": retrieval_result.fallback_reason,
                    "context_sufficient": context_sufficient,
                    "context_chars": len(context),
                    "context_tokens_estimate": token_count_estimate(context),
                    "prompt_chars": len(prompt),
                    "prompt_tokens_estimate": token_count_estimate(prompt),
                    "retrieved_chunks_count": len(retrieval_result.retrieved_chunks),
                    "retrieved_chunk_ids": json.dumps(
                        [item.chunk.get("chunk_id", "") for item in retrieval_result.retrieved_chunks],
                        ensure_ascii=False,
                    ),
                    "timing_retriever_init_seconds": round(retriever_init_seconds, 6),
                    "timing_retrieval_seconds": round(retrieval_seconds, 6),
                    "timing_context_format_seconds": round(context_format_seconds, 6),
                    "timing_prompt_build_seconds": round(prompt_build_seconds, 6),
                    "timing_context_sufficiency_seconds": round(context_sufficiency_seconds, 6),
                    "timing_temporal_intent_seconds": round(retrieval_result.retrieval_timings.get("temporal_intent_seconds", 0.0), 6),
                    "timing_temporal_resolver_seconds": round(retrieval_result.retrieval_timings.get("temporal_resolver_seconds", 0.0), 6),
                    "timing_temporal_anchor_merge_seconds": round(retrieval_result.retrieval_timings.get("temporal_anchor_merge_seconds", 0.0), 6),
                    "timing_temporal_raw_scan_seconds": round(retrieval_result.retrieval_timings.get("temporal_raw_scan_seconds", 0.0), 6),
                    "timing_temporal_raw_candidate_count": int(retrieval_result.retrieval_timings.get("temporal_raw_candidate_count", 0.0)),
                    "timing_temporal_raw_rank_seconds": round(retrieval_result.retrieval_timings.get("temporal_raw_rank_seconds", 0.0), 6),
                    "timing_query_encode_seconds": round(retrieval_result.retrieval_timings.get("query_encode_seconds", 0.0), 6),
                    "timing_route_dense_sparse_seconds": round(retrieval_result.retrieval_timings.get("route_dense_sparse_seconds", 0.0), 6),
                    "timing_route_rank_seconds": round(retrieval_result.retrieval_timings.get("route_rank_seconds", 0.0), 6),
                    "timing_source_shortlist_seconds": round(retrieval_result.retrieval_timings.get("source_shortlist_seconds", 0.0), 6),
                    "timing_chunk_dense_sparse_seconds": round(retrieval_result.retrieval_timings.get("chunk_dense_sparse_seconds", 0.0), 6),
                    "timing_chunk_lexical_seconds": round(retrieval_result.retrieval_timings.get("chunk_lexical_seconds", 0.0), 6),
                    "timing_chunk_rrf_seconds": round(retrieval_result.retrieval_timings.get("chunk_rrf_seconds", 0.0), 6),
                    "timing_colbert_seconds": round(retrieval_result.retrieval_timings.get("colbert_seconds", 0.0), 6),
                    "timing_final_rrf_seconds": round(retrieval_result.retrieval_timings.get("final_rrf_seconds", 0.0), 6),
                    "timing_dedupe_seconds": round(retrieval_result.retrieval_timings.get("dedupe_seconds", 0.0), 6),
                    "timing_moon_gate_seconds": round(retrieval_result.retrieval_timings.get("moon_gate_seconds", 0.0), 6),
                    "timing_openai_call_seconds": round(provider_call_seconds.get("openai", 0.0), 6),
                    "timing_openai_eval_seconds": round(provider_eval_seconds.get("openai", 0.0), 6),
                    "timing_gemini_call_seconds": round(provider_call_seconds.get("gemini", 0.0), 6),
                    "timing_gemini_eval_seconds": round(provider_eval_seconds.get("gemini", 0.0), 6),
                    "timing_row_total_seconds": round(time.perf_counter() - row_started, 6),
                    "temporal_evidence_status": retrieval_result.temporal_evidence_status,
                    "temporal_evidence_reason": retrieval_result.temporal_evidence_reason,
                    "current_evidence_status": retrieval_result.current_evidence_status,
                    "current_evidence_reason": retrieval_result.current_evidence_reason,
                    "openai_answer": answers.get("openai", ""),
                    "openai_is_correct": evals.get("openai", {}).get("is_correct", ""),
                    "openai_reason": evals.get("openai", {}).get("reason", ""),
                    "gemini_answer": answers.get("gemini", ""),
                    "gemini_is_correct": evals.get("gemini", {}).get("is_correct", ""),
                    "gemini_reason": evals.get("gemini", {}).get("reason", ""),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG method matrix through OpenAI and Gemini.")
    parser.add_argument("--source-set", choices=("web", "pdf"), default="web")
    parser.add_argument("--questions", nargs="+", type=int, default=DEFAULT_QUESTION_IDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--providers", nargs="+", choices=("openai", "gemini"), default=["openai", "gemini"])
    parser.add_argument("--questions-path", type=Path, default=PROJECT_ROOT / "data" / "qa_92.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument(
        "--disable-temporal-first",
        action="store_true",
        help="Disable the production temporal-first resolver for fair method comparison.",
    )
    return parser.parse_args()


def build_retriever(source_set: str, *, enable_temporal_first: bool = True) -> RagRetriever:
    if source_set != "pdf":
        return RagRetriever(enable_temporal_first=enable_temporal_first)
    return RagRetriever(
        PROJECT_ROOT / PDF_CHUNKS_PATH,
        documents_path=PROJECT_ROOT / PDF_DOCUMENTS_PATH,
        embeddings_path=PROJECT_ROOT / PDF_BGE_M3_EMBEDDINGS_PATH,
        bge_base_embeddings_path=PROJECT_ROOT / PDF_BGE_BASE_EMBEDDINGS_PATH,
        openai_embeddings_path=PROJECT_ROOT / PDF_OPENAI_EMBEDDINGS_PATH,
        routing_units_path=PROJECT_ROOT / PDF_ROUTING_UNITS_PATH,
        routing_embeddings_path=PROJECT_ROOT / PDF_ROUTING_EMBEDDINGS_PATH,
        enable_temporal_first=enable_temporal_first,
    )


def main() -> None:
    args = parse_args()
    if args.source_set == "pdf" and args.output == DEFAULT_OUTPUT:
        args.output = PROJECT_ROOT / "reports" / "rag_pdf_audit" / "rag_llm_method_matrix.csv"
    questions = load_questions(args.questions_path, args.questions)
    rows = run_matrix(
        questions=questions,
        methods=args.methods,
        providers=args.providers,
        output=args.output,
        max_chars=args.max_chars,
        source_set=args.source_set,
        enable_temporal_first=not args.disable_temporal_first,
    )
    print(f"\nRows: {len(rows)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
