"""Ingest a RAG source manifest into cleaned text files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.cleaners.generic_html_cleaner import clean_generic_html_source
from src.rag.cleaners.wikipedia_cleaner_adapter import clean_wikipedia_page
from src.rag.source_manifest import RagSource, load_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest RAG source links into cleaned text files.")
    parser.add_argument(
        "--sources",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "sources_q9.json"),
        help="Path to source manifest JSON",
    )
    parser.add_argument(
        "--documents",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "documents.jsonl"),
        help="Path where document metadata JSONL is written",
    )
    parser.add_argument("--limit", type=int, help="Only ingest the first N sources")
    return parser


def ingest_source(source: RagSource) -> dict[str, Any]:
    if source.cleaner == "wikipedia":
        result = clean_wikipedia_page(
            url=source.url,
            source_id=source.source_id,
            docs_clean_dir=PROJECT_ROOT / "data" / "rag_sources" / "docs_clean" / "wikipedia",
            raw_dir=PROJECT_ROOT / "data" / "rag_sources" / "web_raw" / "wikipedia",
            save_raw=True,
        ).to_dict()
    elif source.cleaner == "generic_html":
        result = clean_generic_html_source(
            url=source.url,
            source_id=source.source_id,
            docs_clean_dir=PROJECT_ROOT / "data" / "rag_sources" / "docs_clean" / "web",
            raw_dir=PROJECT_ROOT / "data" / "rag_sources" / "web_raw" / "web",
        ).to_dict()
    else:
        raise ValueError(f"Unsupported cleaner: {source.cleaner}")

    return {
        "document_id": f"doc_{source.source_id}",
        "source_id": source.source_id,
        "url": source.url,
        "cleaner": source.cleaner,
        "domain": source.domain,
        "target_questions": source.target_questions,
        "needed_evidence": source.needed_evidence,
        "title": result.get("title", source.source_id),
        "clean_text_path": result["clean_text_path"],
        "raw_path": result.get("raw_html_path"),
        "content_hash": result["content_hash"],
        "char_count": result["char_count"],
    }


def main() -> int:
    args = build_parser().parse_args()
    sources = load_sources(args.sources)
    if args.limit is not None:
        sources = sources[: args.limit]

    documents_path = Path(args.documents)
    documents_path.parent.mkdir(parents=True, exist_ok=True)

    documents = []
    for source in sources:
        print(f"Ingesting {source.source_id}: {source.url}")
        document = ingest_source(source)
        documents.append(document)
        print(f"  saved: {document['clean_text_path']} ({document['char_count']} chars)")

    with documents_path.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(document, ensure_ascii=False) + "\n")
    print(f"Wrote documents: {documents_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

