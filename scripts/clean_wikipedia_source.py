"""Clean one Wikipedia source page for the RAG source store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.cleaners.wikipedia_cleaner_adapter import clean_wikipedia_page


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean a Wikipedia page into data/rag_sources.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code for --title")
    parser.add_argument("--source-id", help="Stable source id used for output filenames")
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="remove",
        help="How math equations should be handled",
    )
    parser.add_argument(
        "--cleaner-dir",
        help="Path to Ahmed-Estiak/Wikipedia_text_extractor. Defaults to external/Wikipedia_text_extractor.",
    )
    parser.add_argument(
        "--docs-clean-dir",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "docs_clean" / "wikipedia"),
        help="Directory where cleaned text files are written",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "web_raw" / "wikipedia"),
        help="Directory where raw HTML is cached",
    )
    parser.add_argument("--no-raw", action="store_true", help="Do not save raw fetched HTML")
    parser.add_argument("--save-references", action="store_true", help="Also save a references-preserving text file")
    parser.add_argument(
        "--references-end-only",
        action="store_true",
        help="When saving references, omit inline markers and keep numbered sources at the end",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

