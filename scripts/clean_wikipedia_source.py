"""Fetch and clean one Wikipedia page for manual RAG-source preparation.

This is a single-source utility around the external
``Ahmed-Estiak/Wikipedia_text_extractor`` adapter. It is useful for inspecting
cleaner output or preparing one page outside the manifest-driven ingestion
pipeline.

Exactly one source selector is required:
    - ``--url`` for a complete Wikipedia page URL, or
    - ``--title`` plus an optional Wikipedia language code.

By default the script writes:
    - cleaned retrieval text under ``data/rag_sources/docs_clean/wikipedia``;
    - fetched raw HTML under ``data/rag_sources/web_raw/wikipedia``.

Optional reference-preserving text is a separate output and does not replace
the normal cleaned text. The command prints a JSON result describing paths,
resolved page metadata, content hash, and character count.

This command performs a network fetch. It does not update ``documents.jsonl``,
build chunks, or build embeddings. Use ``scripts/ingest_rag_sources.py`` for
manifest-driven ingestion into the complete web RAG pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.cleaners.wikipedia_cleaner_adapter import clean_wikipedia_page


def build_parser() -> argparse.ArgumentParser:
    """Define source-selection, cleaner, and output options."""
    parser = argparse.ArgumentParser(description="Clean a Wikipedia page into data/rag_sources.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Complete Wikipedia page URL to fetch.")
    source.add_argument("--title", help="Wikipedia page title to fetch.")
    parser.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language code used only with --title.",
    )
    parser.add_argument(
        "--source-id",
        help=(
            "Stable source ID used for output filenames. If omitted, one is "
            "derived from the URL/title."
        ),
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="remove",
        help="How math equations are represented in cleaned retrieval text.",
    )
    parser.add_argument(
        "--cleaner-dir",
        help=(
            "Path to Ahmed-Estiak/Wikipedia_text_extractor. Overrides "
            "WIKIPEDIA_CLEANER_PATH and the external/Wikipedia_text_extractor default."
        ),
    )
    parser.add_argument(
        "--docs-clean-dir",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "docs_clean" / "wikipedia"),
        help="Directory where cleaned retrieval text is written.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "web_raw" / "wikipedia"),
        help="Directory where fetched raw HTML is cached.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Fetch and clean the page without retaining raw HTML.",
    )
    parser.add_argument(
        "--save-references",
        action="store_true",
        help="Also write a separate references-preserving cleaned-text file.",
    )
    parser.add_argument(
        "--references-end-only",
        action="store_true",
        help=(
            "With --save-references, omit inline reference markers while "
            "retaining the numbered sources at the end."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # clean_wikipedia_page resolves the external cleaner, fetches the requested
    # page, derives/uses a stable source ID, cleans the HTML, and writes the
    # requested artifacts. Passing --cleaner-dir takes precedence over the
    # environment variable and repository-local cleaner path.
    result = clean_wikipedia_page(
        url=args.url,
        title=args.title,
        lang=args.lang,
        source_id=args.source_id,
        math_mode=args.math,
        docs_clean_dir=args.docs_clean_dir,
        raw_dir=args.raw_dir,
        cleaner_dir=args.cleaner_dir,
        save_raw=not args.no_raw,
        save_references=args.save_references,
        references_end_only=args.references_end_only,
    )

    # Machine-readable output makes the utility useful in manual workflows and
    # small automation without registering the page in documents.jsonl.
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
