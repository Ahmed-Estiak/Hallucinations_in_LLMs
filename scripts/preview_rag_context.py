"""Preview retrieved RAG context without calling any LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import RagRetriever


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
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--preview-chars", type=int, default=180)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    question = args.question or question_by_id(args.id)

    retriever = RagRetriever(args.chunks)
    retrieved = retriever.retrieve(
        question,
        top_k=args.top_k,
        per_source_limit=args.per_source_limit,
    )
    context = retriever.format_context(retrieved, max_chars=args.max_chars)

    print(f"Question: {question}")
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


def question_by_id(question_id: int) -> str:
    questions_path = PROJECT_ROOT / "data" / "qa_92.json"
    with questions_path.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    for question in questions:
        if int(question["id"]) == question_id:
            return question["question"]
    raise ValueError(f"Question id not found: {question_id}")


def one_line(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())

