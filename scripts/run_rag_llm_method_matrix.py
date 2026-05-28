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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import evaluate_answer
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
    output: Path,
    max_chars: int,
) -> list[dict[str, Any]]:
    retriever = RagRetriever()
    rows: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for method in methods:
        print(f"\n{method}")
        for question_row in questions:
            question_id = int(question_row["id"])
            question = question_row["question"]
            truth = expected_answer(question_row)
            print(f"  Q{question_id}: retrieving...")
            retrieval_result = retriever.retrieve_with_details(question, mode=method)
            context = retriever.format_context_for_llm(
                retrieval_result.retrieved_chunks,
                max_chars=max_chars,
            )
            prompt = build_rag_prompt(question, context)
            context_sufficient = (
                bool(retrieval_result.retrieved_chunks)
                and len(context) >= 200
                and retrieval_result.temporal_evidence_status not in {"insufficient", "conflict"}
                and retrieval_result.current_evidence_status not in {"insufficient", "conflict"}
            )

            answers: dict[str, str] = {}
            evals: dict[str, dict[str, Any]] = {}
            for provider in ("openai", "gemini"):
                if context_sufficient:
                    try:
                        answer = call_provider(provider, question, context)
                    except Exception as exc:
                        answer = f"ERROR: {type(exc).__name__}: {exc}"
                else:
                    answer = "insufficient context"
                answers[provider] = answer
                evals[provider] = evaluate_answer(question_row, answer)

            fallback_note = (
                f", fallback: {retrieval_result.fallback_reason}"
                if retrieval_result.fallback_used
                else ""
            )
            print(
                "    "
                f"OpenAI={answers['openai']} ({evals['openai']['reason']}), "
                f"Gemini={answers['gemini']} ({evals['gemini']['reason']}), "
                f"executed={retrieval_result.retrieval_mode}{fallback_note}"
            )

            rows.append(
                {
                    "generated_at_utc": generated_at,
                    "question_id": question_id,
                    "question": question,
                    "expected_answer": truth,
                    "requested_method": method,
                    "executed_method": retrieval_result.retrieval_mode,
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
                    "temporal_evidence_status": retrieval_result.temporal_evidence_status,
                    "temporal_evidence_reason": retrieval_result.temporal_evidence_reason,
                    "current_evidence_status": retrieval_result.current_evidence_status,
                    "current_evidence_reason": retrieval_result.current_evidence_reason,
                    "openai_answer": answers["openai"],
                    "openai_is_correct": evals["openai"]["is_correct"],
                    "openai_reason": evals["openai"]["reason"],
                    "gemini_answer": answers["gemini"],
                    "gemini_is_correct": evals["gemini"]["is_correct"],
                    "gemini_reason": evals["gemini"]["reason"],
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
    parser.add_argument("--questions", nargs="+", type=int, default=DEFAULT_QUESTION_IDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--questions-path", type=Path, default=PROJECT_ROOT / "data" / "qa_92.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions_path, args.questions)
    rows = run_matrix(
        questions=questions,
        methods=args.methods,
        output=args.output,
        max_chars=args.max_chars,
    )
    print(f"\nRows: {len(rows)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
