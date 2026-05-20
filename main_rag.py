"""Main entry point for the RAG+LLM benchmark.

Usage:
    python main_rag.py
"""

from __future__ import annotations

import argparse

from src.rag_runner import run_rag_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG+LLM benchmark.")
    parser.add_argument("--ids", nargs="+", type=int, help="Question ids to run")
    parser.add_argument("--retrieval-mode", choices=("global", "auto-source"), default="global")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--per-source-limit", type=int, default=4)
    parser.add_argument("--top-n-sources", type=int, default=12)
    parser.add_argument("--output", default="results/results_rag_llm.csv")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_rag_benchmark(
        question_ids=args.ids,
        output_path=args.output,
        top_k=args.top_k,
        per_source_limit=args.per_source_limit,
        retrieval_mode=args.retrieval_mode,
        top_n_sources=args.top_n_sources,
    )
