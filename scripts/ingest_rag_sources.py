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
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "sources_master.json"),
        help="Path to source manifest JSON",
    )
    parser.add_argument(
        "--documents",
        default=str(PROJECT_ROOT / "data" / "rag_sources" / "rag_index" / "documents.jsonl"),
        help="Path where document metadata JSONL is written",
    )
    parser.add_argument("--limit", type=int, help="Only ingest the first N sources")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch sources even if cached clean text exists")
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
        "trust_level": source.trust_level,
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
    existing_documents = load_existing_documents(documents_path)

    documents = []
    for source in sources:
        existing_document = existing_documents.get(source.source_id)
        if existing_document and not args.refresh and Path(existing_document.get("clean_text_path", "")).exists():
            print(f"Skipping cached {source.source_id}: {source.url}")
            document = update_cached_document(existing_document, source)
        else:
            print(f"Ingesting {source.source_id}: {source.url}")
            document = ingest_source(source)
            print(f"  saved: {document['clean_text_path']} ({document['char_count']} chars)")
        documents.append(document)

    with documents_path.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(document, ensure_ascii=False) + "\n")
    print(f"Wrote documents: {documents_path}")
    return 0


def load_existing_documents(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    documents = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            document = json.loads(line)
            documents[document["source_id"]] = document
    return documents


def update_cached_document(document: dict[str, Any], source: RagSource) -> dict[str, Any]:
    updated = dict(document)
    updated.update({
        "url": source.url,
        "cleaner": source.cleaner,
        "domain": source.domain,
        "trust_level": source.trust_level,
        "target_questions": source.target_questions,
        "needed_evidence": source.needed_evidence,
    })
    return updated


if __name__ == "__main__":
    raise SystemExit(main())
