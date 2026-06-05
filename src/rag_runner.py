"""Run selected benchmark questions through LLM + RAG."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import evaluate_answer
from src.rag.embeddings import (
    DEFAULT_BGE_BASE_EMBEDDINGS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_OPENAI_EMBEDDINGS_PATH,
)
from src.rag.retriever import DEFAULT_RETRIEVAL_MODE, RagRetriever
from src.rag.routing import DEFAULT_ROUTING_EMBEDDINGS_PATH, DEFAULT_ROUTING_UNITS_PATH
from src.rag_models import ask_gemini_with_rag, ask_openai_with_rag


DEFAULT_QUESTION_IDS = [9, 11, 15]


def _serialize_ground_truth(answer_spec: dict[str, Any]) -> str:
    if "value" in answer_spec:
        return json.dumps(answer_spec["value"], ensure_ascii=False)
    if "fields" in answer_spec:
        return json.dumps(answer_spec["fields"], ensure_ascii=False)
    return ""


def load_questions(question_ids: list[int] | None = None) -> list[dict[str, Any]]:
    with open("data/qa_92.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    if question_ids is None:
        return questions
    selected = set(question_ids)
    return [question for question in questions if int(question["id"]) in selected]


def run_rag_benchmark(
    *,
    question_ids: list[int] | None = None,
    chunks_path: str | Path = "data/rag_sources/rag_index/chunks.jsonl",
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    bge_base_embeddings_path: str | Path = DEFAULT_BGE_BASE_EMBEDDINGS_PATH,
    openai_embeddings_path: str | Path = DEFAULT_OPENAI_EMBEDDINGS_PATH,
    routing_units_path: str | Path = DEFAULT_ROUTING_UNITS_PATH,
    routing_embeddings_path: str | Path = DEFAULT_ROUTING_EMBEDDINGS_PATH,
    output_path: str | Path = "reports/final/results_rag_llm.csv",
    top_k: int = 12,
    per_source_limit: int = 4,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    top_n_sources: int = 12,
) -> None:
    start_time = time.time()
    questions = load_questions(question_ids or DEFAULT_QUESTION_IDS)
    retriever = RagRetriever(
        chunks_path,
        embeddings_path=embeddings_path,
        bge_base_embeddings_path=bge_base_embeddings_path,
        openai_embeddings_path=openai_embeddings_path,
        routing_units_path=routing_units_path,
        routing_embeddings_path=routing_embeddings_path,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    openai_correct = 0
    gemini_correct = 0

    print(f"Starting RAG+LLM benchmark ({len(questions)} questions)")
    print("=" * 80)
    for index, question_row in enumerate(questions, start=1):
        question = question_row["question"]
        print(f"[{index}/{len(questions)}] Q{question_row['id']}: {question}")
        question_start = time.time()

        retrieval_result = retriever.retrieve_with_details(
            question,
            top_k=top_k,
            per_source_limit=per_source_limit,
            mode=retrieval_mode,
            top_n_sources=top_n_sources,
        )
        retrieved = retrieval_result.retrieved_chunks
        rag_context = retriever.format_context_for_llm(retrieved)
        rag_debug_context = retriever.format_context(retrieved)
        temporal_blocked = retrieval_result.temporal_evidence_status in {"insufficient", "conflict"}
        current_blocked = retrieval_result.current_evidence_status in {"insufficient", "conflict"}
        context_sufficient = (
            len(retrieved) > 0
            and len(rag_context) >= 200
            and not temporal_blocked
            and not current_blocked
        )

        if context_sufficient:
            openai_answer = ask_openai_with_rag(question, rag_context)
            gemini_answer = ask_gemini_with_rag(question, rag_context)
        else:
            openai_answer = "insufficient context"
            gemini_answer = "insufficient context"

        openai_eval = evaluate_answer(question_row, openai_answer)
        gemini_eval = evaluate_answer(question_row, gemini_answer)
        openai_correct += int(bool(openai_eval["is_correct"]))
        gemini_correct += int(bool(gemini_eval["is_correct"]))

        rows.append({
            "id": question_row["id"],
            "question": question,
            "kind": question_row["answer_spec"]["kind"],
            "type": question_row.get("type", ""),
            "ground_truth": _serialize_ground_truth(question_row["answer_spec"]),
            "retrieval_mode": retrieval_result.retrieval_mode,
            "requested_retrieval_mode": retrieval_result.requested_mode,
            "embedding_provider": retrieval_result.embedding_provider,
            "embedding_model": retrieval_result.embedding_model,
            "embeddings_path": retrieval_result.embeddings_path,
            "fallback_used": retrieval_result.fallback_used,
            "fallback_reason": retrieval_result.fallback_reason,
            "routing_units_scored": retrieval_result.routing_units_scored,
            "selected_route_ids": json.dumps(retrieval_result.selected_route_ids, ensure_ascii=False),
            "candidate_chunks_scored": retrieval_result.candidate_chunks_scored,
            "full_chunk_count": retrieval_result.full_chunk_count,
            "comparison_reduction_percent": retrieval_result.comparison_reduction_percent,
            "colbert_candidates": retrieval_result.colbert_candidates,
            "temporal_evidence_status": retrieval_result.temporal_evidence_status,
            "temporal_evidence_reason": retrieval_result.temporal_evidence_reason,
            "current_evidence_status": retrieval_result.current_evidence_status,
            "current_evidence_reason": retrieval_result.current_evidence_reason,
            "selected_sources": json.dumps(
                retrieval_result.source_selection.selected_source_ids if retrieval_result.source_selection else [],
                ensure_ascii=False,
            ),
            "source_scores": json.dumps(
                [score.to_dict() for score in retrieval_result.source_selection.scores[:12]]
                if retrieval_result.source_selection else [],
                ensure_ascii=False,
            ),
            "rag_context_sufficient": context_sufficient,
            "retrieved_chunk_ids": json.dumps([item.chunk["chunk_id"] for item in retrieved], ensure_ascii=False),
            "retrieved_sources": json.dumps([item.chunk["source_id"] for item in retrieved], ensure_ascii=False),
            "retrieval_scores": json.dumps([round(item.score, 4) for item in retrieved], ensure_ascii=False),
            "rag_context_text": rag_context,
            "rag_debug_context_text": rag_debug_context,
            "openai_rag_answer": openai_answer,
            "gemini_rag_answer": gemini_answer,
            "openai_rag_is_correct": openai_eval["is_correct"],
            "gemini_rag_is_correct": gemini_eval["is_correct"],
            "openai_rag_reason": openai_eval["reason"],
            "gemini_rag_reason": gemini_eval["reason"],
        })

        elapsed = time.time() - question_start
        print(f"  OpenAI: {openai_answer} ({openai_eval['reason']})")
        print(f"  Gemini: {gemini_answer} ({gemini_eval['reason']})")
        fallback_note = f" | Fallback: {retrieval_result.fallback_reason}" if retrieval_result.fallback_used else ""
        print(
            f"  Retrieved chunks: {len(retrieved)} | Mode: {retrieval_result.retrieval_mode}"
            f"{fallback_note} | Timing: {elapsed:.2f}s"
        )
        if retrieval_result.routing_units_scored:
            print(
                f"  Routing units: {retrieval_result.routing_units_scored} | "
                f"Candidate chunks: {retrieval_result.candidate_chunks_scored} | "
                f"Reduction: {retrieval_result.comparison_reduction_percent:.2f}%"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    elapsed_seconds = time.time() - start_time
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - RAG+LLM")
    print("=" * 80)
    print(f"Total questions: {len(questions)}")
    print(f"OpenAI RAG: {openai_correct}/{len(questions)}")
    print(f"Gemini RAG: {gemini_correct}/{len(questions)}")
    print(f"Results saved to: {output_path}")
    print(f"Total runtime: {elapsed_seconds:.2f}s")


if __name__ == "__main__":
    run_rag_benchmark()
