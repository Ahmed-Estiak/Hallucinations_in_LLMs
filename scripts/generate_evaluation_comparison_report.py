"""Build a comparison report from existing benchmark CSVs.

This script does not call any LLM or embedding API. It only reads completed
evaluation outputs and reshapes them into:

- a normalized CSV where every row is one provider/system/question result;
- a short Markdown summary with aggregate correctness tables.

The report is intended for the project-level comparison:

1. LLM without KG vs LLM with KG on the 15 selected KG questions.
2. LLM without RAG vs LLM with Web RAG on Q9, Q11, Q15.
3. LLM without RAG vs LLM with PDF RAG on Q9, Q11, Q15.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KG_BASELINE = PROJECT_ROOT / "reports" / "final" / "results.csv"
DEFAULT_KG_WITH_CONTEXT = PROJECT_ROOT / "reports" / "final" / "results_with_kg.csv"
DEFAULT_WEB_RAG = PROJECT_ROOT / "reports" / "final" / "rag_llm_all_modes_q9_q11_q15.csv"
DEFAULT_PDF_RAG = PROJECT_ROOT / "reports" / "debug" / "pdf" / "rag_llm_method_matrix.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "final" / "evaluation_comparison_normalized.csv"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "final" / "evaluation_comparison_summary.md"
RAG_QUESTION_IDS = {9, 11, 15}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_text(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return ""


def is_true(value: object) -> bool:
    return bool_text(value) == "true"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def provider_rows_from_kg_baseline(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        for provider in ("openai", "gemini"):
            normalized.append({
                "experiment": "kg_15_questions",
                "source_set": "none",
                "system": "llm_without_kg",
                "provider": provider,
                "method": "vanilla",
                "question_id": row.get("id", ""),
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", ""),
                "answer": row.get(f"{provider}_answer", ""),
                "is_correct": bool_text(row.get(f"{provider}_is_correct", "")),
                "reason": row.get(f"{provider}_reason", ""),
                "fallback_used": "",
                "executed_method": "",
                "context_tokens_estimate": "",
                "prompt_tokens_estimate": "",
                "retrieved_chunks_count": "",
            })
    return normalized


def provider_rows_from_kg_context(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        for provider in ("openai", "gemini"):
            normalized.append({
                "experiment": "kg_15_questions",
                "source_set": "kg",
                "system": "llm_with_kg",
                "provider": provider,
                "method": "kg_context",
                "question_id": row.get("id", ""),
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", ""),
                "answer": row.get(f"{provider}_kg_answer", ""),
                "is_correct": bool_text(row.get(f"{provider}_kg_is_correct", "")),
                "reason": row.get(f"{provider}_kg_reason", ""),
                "fallback_used": "",
                "executed_method": "",
                "context_tokens_estimate": "",
                "prompt_tokens_estimate": "",
                "retrieved_chunks_count": "",
            })
    return normalized


def provider_rows_from_rag_baseline(rows: Iterable[dict[str, str]], source_set: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        try:
            qid = int(row.get("id", ""))
        except ValueError:
            continue
        if qid not in RAG_QUESTION_IDS:
            continue
        for provider in ("openai", "gemini"):
            normalized.append({
                "experiment": f"{source_set}_rag_3_questions",
                "source_set": "none",
                "system": f"llm_without_{source_set}_rag",
                "provider": provider,
                "method": "vanilla",
                "question_id": row.get("id", ""),
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", ""),
                "answer": row.get(f"{provider}_answer", ""),
                "is_correct": bool_text(row.get(f"{provider}_is_correct", "")),
                "reason": row.get(f"{provider}_reason", ""),
                "fallback_used": "",
                "executed_method": "",
                "context_tokens_estimate": "",
                "prompt_tokens_estimate": "",
                "retrieved_chunks_count": "",
            })
    return normalized


def provider_rows_from_rag(rows: Iterable[dict[str, str]], *, source_set: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        question_id = row.get("id") or row.get("question_id", "")
        method = row.get("run_requested_mode") or row.get("requested_method") or row.get("requested_retrieval_mode", "")
        executed = row.get("retrieval_mode") or row.get("executed_method", "")
        for provider in ("openai", "gemini"):
            answer_key = f"{provider}_rag_answer" if f"{provider}_rag_answer" in row else f"{provider}_answer"
            correct_key = f"{provider}_rag_is_correct" if f"{provider}_rag_is_correct" in row else f"{provider}_is_correct"
            reason_key = f"{provider}_rag_reason" if f"{provider}_rag_reason" in row else f"{provider}_reason"
            if answer_key not in row and correct_key not in row:
                continue
            normalized.append({
                "experiment": f"{source_set}_rag_3_questions",
                "source_set": source_set,
                "system": f"llm_with_{source_set}_rag",
                "provider": provider,
                "method": method,
                "question_id": question_id,
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth") or row.get("expected_answer", ""),
                "answer": row.get(answer_key, ""),
                "is_correct": bool_text(row.get(correct_key, "")),
                "reason": row.get(reason_key, ""),
                "fallback_used": row.get("fallback_used", ""),
                "executed_method": executed,
                "context_tokens_estimate": row.get("context_tokens_estimate", ""),
                "prompt_tokens_estimate": row.get("prompt_tokens_estimate", ""),
                "retrieved_chunks_count": row.get("retrieved_chunks_count", ""),
            })
    return normalized


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        key = (
            row["experiment"],
            row["source_set"],
            row["system"],
            row["provider"],
            row["method"],
        )
        groups[key]["total"] += 1
        groups[key]["correct"] += int(is_true(row["is_correct"]))

    summary = []
    for key, counts in sorted(groups.items()):
        total = counts["total"]
        correct = counts["correct"]
        summary.append({
            "experiment": key[0],
            "source_set": key[1],
            "system": key[2],
            "provider": key[3],
            "method": key[4],
            "correct": str(correct),
            "total": str(total),
            "accuracy_percent": f"{(correct / total * 100):.1f}" if total else "0.0",
        })
    return summary


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows found._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, str]], summary: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kg_rows = [row for row in summary if row["experiment"] == "kg_15_questions"]
    web_rows = [row for row in summary if row["experiment"] == "web_rag_3_questions"]
    pdf_rows = [row for row in summary if row["experiment"] == "pdf_rag_3_questions"]
    detail_rows = [
        row for row in rows
        if row["experiment"] in {"web_rag_3_questions", "pdf_rag_3_questions"}
    ]
    detail_columns = [
        "experiment",
        "provider",
        "method",
        "question_id",
        "ground_truth",
        "answer",
        "is_correct",
        "reason",
        "fallback_used",
        "executed_method",
    ]
    summary_columns = ["experiment", "source_set", "system", "provider", "method", "correct", "total", "accuracy_percent"]

    content = f"""# Evaluation Comparison Report

This report is generated from existing CSV outputs only. It does not run new LLM
or embedding calls.

## KG: LLM Without KG vs With KG

{markdown_table(kg_rows, summary_columns)}

## Web RAG: 3-Question Comparison

{markdown_table(web_rows, summary_columns)}

## PDF RAG: 3-Question Comparison

{markdown_table(pdf_rows, summary_columns)}

## RAG Question-Level Details

{markdown_table(detail_rows, detail_columns)}
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the evaluation comparison report.")
    parser.add_argument("--kg-baseline", type=Path, default=DEFAULT_KG_BASELINE)
    parser.add_argument("--kg-with-context", type=Path, default=DEFAULT_KG_WITH_CONTEXT)
    parser.add_argument("--web-rag", type=Path, default=DEFAULT_WEB_RAG)
    parser.add_argument("--pdf-rag", type=Path, default=DEFAULT_PDF_RAG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    kg_baseline = read_csv(args.kg_baseline)
    normalized = []
    normalized.extend(provider_rows_from_kg_baseline(kg_baseline))
    normalized.extend(provider_rows_from_kg_context(read_csv(args.kg_with_context)))
    normalized.extend(provider_rows_from_rag_baseline(kg_baseline, "web"))
    normalized.extend(provider_rows_from_rag(read_csv(args.web_rag), source_set="web"))
    normalized.extend(provider_rows_from_rag_baseline(kg_baseline, "pdf"))
    normalized.extend(provider_rows_from_rag(read_csv(args.pdf_rag), source_set="pdf"))

    summary = summarize(normalized)
    write_csv(args.output_csv, normalized)
    write_markdown(args.output_md, normalized, summary)
    print(f"Wrote normalized CSV: {args.output_csv}")
    print(f"Wrote Markdown summary: {args.output_md}")


if __name__ == "__main__":
    main()
