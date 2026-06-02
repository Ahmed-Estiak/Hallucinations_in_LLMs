"""Ingest copyable PDFs into the PDF-only RAG document index."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import write_jsonl
from src.rag.pdf_extractor import extract_pdf_text
from src.rag.pdf_paths import PDF_DOCUMENTS_PATH, PDF_LAYOUT_DIAGNOSTICS_PATH, PDF_RAW_DIR, PDF_TEXT_DIR


PDF_EXTENSIONS = {".pdf"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract copyable PDFs into PDF-only RAG documents.")
    parser.add_argument("--pdf-dir", type=Path, default=PROJECT_ROOT / PDF_RAW_DIR)
    parser.add_argument("--text-dir", type=Path, default=PROJECT_ROOT / PDF_TEXT_DIR)
    parser.add_argument("--documents", type=Path, default=PROJECT_ROOT / PDF_DOCUMENTS_PATH)
    parser.add_argument("--layout-diagnostics", type=Path, default=PROJECT_ROOT / PDF_LAYOUT_DIAGNOSTICS_PATH)
    parser.add_argument("--refresh", action="store_true", help="Re-extract even when cleaned text exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.text_dir.mkdir(parents=True, exist_ok=True)
    args.documents.parent.mkdir(parents=True, exist_ok=True)
    args.layout_diagnostics.parent.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(
        path for path in args.pdf_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PDF_EXTENSIONS
    )
    documents = []
    layout_rows = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        source_id = source_id_from_pdf(pdf_path)
        clean_path = args.text_dir / f"{source_id}.txt"
        if clean_path.exists() and not args.refresh:
            text = clean_path.read_text(encoding="utf-8")
            page_count = text.count("[[PAGE ")
            raw_char_count = len(text)
            cleaned_char_count = len(text)
            repeated_lines_removed = 0
            status = "cached"
        else:
            extracted = extract_pdf_text(pdf_path)
            clean_path.write_text(extracted.text, encoding="utf-8")
            page_count = extracted.page_count
            raw_char_count = extracted.raw_char_count
            cleaned_char_count = extracted.cleaned_char_count
            repeated_lines_removed = extracted.repeated_lines_removed
            for row in extracted.page_diagnostics:
                layout_rows.append({
                    "source_id": source_id,
                    "title": title_from_pdf(pdf_path),
                    "file_name": pdf_path.name,
                    **row,
                })
            status = "extracted"

        documents.append({
            "document_id": f"doc_{source_id}",
            "source_id": source_id,
            "source_type": "pdf",
            "url": "",
            "title": title_from_pdf(pdf_path),
            "file_path": str(pdf_path),
            "clean_text_path": str(clean_path),
            "cleaner": "copyable_pdf_text",
            "trust_level": "pdf",
            "target_questions": [],
            "needed_evidence": [],
            "page_count": page_count,
            "raw_char_count": raw_char_count,
            "char_count": cleaned_char_count,
            "repeated_lines_removed": repeated_lines_removed,
        })
        print(f"{status}: {pdf_path.name} -> {clean_path.name} ({page_count} pages)")

    write_jsonl(args.documents, documents)
    if layout_rows or not args.layout_diagnostics.exists():
        write_layout_diagnostics(args.layout_diagnostics, layout_rows)
    print(f"Wrote PDF documents: {args.documents} ({len(documents)} documents)")
    if layout_rows:
        print(f"Wrote PDF layout diagnostics: {args.layout_diagnostics} ({len(layout_rows)} rows)")
    else:
        print(f"PDF layout diagnostics unchanged: {args.layout_diagnostics}")
    if not documents:
        print(f"No PDFs found in: {args.pdf_dir}")
    return 0


def source_id_from_pdf(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower()
    return f"pdf_{stem or 'document'}"


def title_from_pdf(path: Path) -> str:
    title = re.sub(r"[_-]+", " ", path.stem).strip()
    return title or path.name


def write_layout_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source_id",
        "title",
        "file_name",
        "page",
        "layout_class",
        "chosen_mode",
        "column_count",
        "table_region_count",
        "block_count",
        "line_count",
        "avg_words_per_block",
        "short_block_ratio",
        "numeric_block_ratio",
        "warning",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
