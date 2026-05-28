"""Generate a retrieval/prompt audit report for selected RAG questions.

This script does not call an LLM. It runs the configured retriever modes,
formats the same context text that would be placed in the RAG prompt, and
writes CSV files that make prompt size and selected evidence easy to compare.
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

from src.rag.retriever import DEFAULT_RETRIEVAL_MODE, RagRetriever
from src.rag_models import build_rag_prompt


DEFAULT_QUESTION_IDS = [9, 11, 15]
DEFAULT_METHODS = [
    "hierarchical-bge-m3-rrf",
    "bge-m3-rrf",
    "bge-base-rrf",
    "openai-embedding-rrf",
    "auto-source",
    "global",
]
LEGACY_METHODS = ["vector", "hybrid"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "rag_audit"


def load_questions(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return {int(item["id"]): item for item in json.load(handle)}


def answer_spec_text(question: dict[str, Any]) -> str:
    spec = question.get("answer_spec", {})
    value = spec.get("value", "")
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def token_count(text: str) -> tuple[int, str]:
    """Return a deterministic token estimate without requiring provider APIs.

    If tiktoken is installed locally, use it for a closer OpenAI-style count.
    Otherwise, use a conservative regex estimate that splits words, numbers,
    and punctuation-like symbols. The CSV records the method so the number is
    not mistaken for an exact provider billable token count.
    """

    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text)), "tiktoken:cl100k_base"
    except Exception:
        pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return len(pieces), "regex_estimate"


def source_ids(result: Any) -> str:
    selection = getattr(result, "source_selection", None)
    if selection:
        return "; ".join(selection.selected_source_ids)
    seen: list[str] = []
    for item in result.retrieved_chunks:
        source_id = item.chunk.get("source_id", "")
        if source_id and source_id not in seen:
            seen.append(source_id)
    return "; ".join(seen)


def safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def build_rows(
    *,
    question_ids: list[int],
    methods: list[str],
    questions: dict[int, dict[str, Any]],
    max_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retriever = RagRetriever()
    run_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for question_id in question_ids:
        question_row = questions[question_id]
        question = question_row["question"]
        expected = answer_spec_text(question_row)
        for method in methods:
            try:
                result = retriever.retrieve_with_details(
                    question,
                    mode=method,
                    top_k=12,
                )
                context_text = retriever.format_context_for_llm(
                    result.retrieved_chunks,
                    max_chars=max_chars,
                )
                prompt_text = build_rag_prompt(question, context_text)
                context_tokens, token_method = token_count(context_text)
                prompt_tokens, _ = token_count(prompt_text)
                debug_context = retriever.format_context(
                    result.retrieved_chunks,
                    max_chars=max_chars,
                )

                run_rows.append(
                    {
                        "generated_at_utc": generated_at,
                        "question_id": question_id,
                        "question": question,
                        "expected_answer": expected,
                        "requested_method": method,
                        "executed_method": result.retrieval_mode,
                        "fallback_used": result.fallback_used,
                        "fallback_reason": result.fallback_reason,
                        "embedding_provider": result.embedding_provider,
                        "embedding_model": result.embedding_model,
                        "context_chars": len(context_text),
                        "context_tokens_estimate": context_tokens,
                        "prompt_chars": len(prompt_text),
                        "prompt_tokens_estimate": prompt_tokens,
                        "token_count_method": token_method,
                        "retrieved_chunks_count": len(result.retrieved_chunks),
                        "selected_sources_or_scope": source_ids(result),
                        "prompt_chunk_ids": "; ".join(
                            item.chunk.get("chunk_id", "")
                            for item in result.retrieved_chunks
                        ),
                        "temporal_evidence_status": result.temporal_evidence_status,
                        "temporal_evidence_reason": result.temporal_evidence_reason,
                        "current_evidence_status": result.current_evidence_status,
                        "current_evidence_reason": result.current_evidence_reason,
                        "routing_units_scored": result.routing_units_scored,
                        "candidate_chunks_scored": result.candidate_chunks_scored,
                        "full_chunk_count": result.full_chunk_count,
                        "comparison_reduction_percent": result.comparison_reduction_percent,
                        "colbert_candidates": result.colbert_candidates,
                        "context_preview": context_text[:500].replace("\n", " "),
                        "debug_context": debug_context,
                    }
                )

                for rank, item in enumerate(result.retrieved_chunks, start=1):
                    chunk = item.chunk
                    chunk_text = safe_text(chunk.get("text"))
                    chunk_tokens, _ = token_count(chunk_text)
                    chunk_rows.append(
                        {
                            "generated_at_utc": generated_at,
                            "question_id": question_id,
                            "requested_method": method,
                            "executed_method": result.retrieval_mode,
                            "rank": rank,
                            "chunk_id": chunk.get("chunk_id", ""),
                            "source_id": chunk.get("source_id", ""),
                            "section": chunk.get("section", ""),
                            "score": f"{item.score:.4f}",
                            "chunk_chars": len(chunk_text),
                            "chunk_tokens_estimate": chunk_tokens,
                            "token_count_method": token_method,
                            "reasons": "; ".join(item.reasons),
                            "text_preview": chunk_text[:500].replace("\n", " "),
                        }
                    )
            except Exception as exc:
                run_rows.append(
                    {
                        "generated_at_utc": generated_at,
                        "question_id": question_id,
                        "question": question,
                        "expected_answer": expected,
                        "requested_method": method,
                        "executed_method": "",
                        "fallback_used": True,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                        "embedding_provider": "",
                        "embedding_model": "",
                        "context_chars": 0,
                        "context_tokens_estimate": 0,
                        "prompt_chars": 0,
                        "prompt_tokens_estimate": 0,
                        "token_count_method": "not_counted",
                        "retrieved_chunks_count": 0,
                        "selected_sources_or_scope": "",
                        "prompt_chunk_ids": "",
                        "temporal_evidence_status": "",
                        "temporal_evidence_reason": "",
                        "current_evidence_status": "",
                        "current_evidence_reason": "",
                        "routing_units_scored": 0,
                        "candidate_chunks_scored": 0,
                        "full_chunk_count": 0,
                        "comparison_reduction_percent": "",
                        "colbert_candidates": 0,
                        "context_preview": "",
                        "debug_context": "",
                    }
                )

    return run_rows, chunk_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAG retrieval and prompt-size audit CSVs.")
    parser.add_argument("--questions", nargs="+", type=int, default=DEFAULT_QUESTION_IDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--include-legacy", action="store_true", help="Also run vector and hybrid modes.")
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--questions-path", type=Path, default=PROJECT_ROOT / "data" / "qa_92.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = list(args.methods)
    if args.include_legacy:
        methods.extend(method for method in LEGACY_METHODS if method not in methods)

    questions = load_questions(args.questions_path)
    missing = [question_id for question_id in args.questions if question_id not in questions]
    if missing:
        raise ValueError(f"Question ids not found: {missing}")

    run_rows, chunk_rows = build_rows(
        question_ids=args.questions,
        methods=methods,
        questions=questions,
        max_chars=args.max_chars,
    )
    runs_path = args.output_dir / "rag_method_prompt_token_audit.csv"
    chunks_path = args.output_dir / "rag_method_chunk_token_audit.csv"
    write_csv(runs_path, run_rows)
    write_csv(chunks_path, chunk_rows)

    print(f"Default retrieval mode: {DEFAULT_RETRIEVAL_MODE}")
    print(f"Run audit rows: {len(run_rows)} -> {runs_path}")
    print(f"Chunk audit rows: {len(chunk_rows)} -> {chunks_path}")


if __name__ == "__main__":
    main()
