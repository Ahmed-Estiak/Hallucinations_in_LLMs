"""Text extraction and cleanup helpers for copyable PDF RAG sources."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAGE_MARKER_RE = re.compile(r"^\[\[PAGE\s+(\d+)\]\]$")


@dataclass(frozen=True)
class ExtractedPdf:
    text: str
    page_count: int
    raw_char_count: int
    cleaned_char_count: int
    repeated_lines_removed: int


def extract_pdf_text(path: str | Path) -> ExtractedPdf:
    """Extract text from a copyable PDF and apply conservative cleanup.

    The extractor deliberately does not remove references/bibliography sections.
    It only removes repeated boilerplate lines, repairs hyphenated line breaks,
    and reconstructs paragraphs enough to make downstream chunking less noisy.
    """

    pdf_path = Path(path)
    raw_pages = _extract_pages(pdf_path)
    repeated_lines = _find_repeated_boilerplate(raw_pages)
    cleaned_pages: list[str] = []
    removed = 0
    for page in raw_pages:
        lines = []
        for line in page.splitlines():
            normalized = normalize_line(line)
            if not normalized:
                continue
            if normalized in repeated_lines or is_page_number_line(normalized):
                removed += 1
                continue
            lines.append(line.rstrip())
        cleaned_pages.append(reconstruct_paragraphs(repair_hyphenation("\n".join(lines))))

    text_parts = []
    for index, page_text in enumerate(cleaned_pages, start=1):
        text_parts.append(f"[[PAGE {index}]]")
        if page_text.strip():
            text_parts.append(page_text.strip())
    text = "\n\n".join(text_parts).strip() + "\n"
    return ExtractedPdf(
        text=text,
        page_count=len(raw_pages),
        raw_char_count=sum(len(page) for page in raw_pages),
        cleaned_char_count=len(text),
        repeated_lines_removed=removed,
    )


def _extract_pages(path: Path) -> list[str]:
    try:
        import fitz  # type: ignore

        pages = []
        with fitz.open(path) as document:
            for page in document:
                pages.append(page.get_text("text") or "")
        return pages
    except ImportError:
        pass

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF or pypdf. Install one of them, "
            "for example: pip install pymupdf"
        ) from exc


def _find_repeated_boilerplate(pages: list[str]) -> set[str]:
    """Detect short repeated header/footer lines across pages."""

    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        seen_on_page = set()
        lines = [normalize_line(line) for line in page.splitlines()]
        candidates = [line for line in lines if is_boilerplate_candidate(line)]
        for line in candidates:
            seen_on_page.add(line)
        counts.update(seen_on_page)
    threshold = max(3, int(len(pages) * 0.6))
    return {line for line, count in counts.items() if count >= threshold}


def is_boilerplate_candidate(line: str) -> bool:
    if not line or len(line) > 90:
        return False
    if is_page_number_line(line):
        return True
    return len(line.split()) <= 10


def is_page_number_line(line: str) -> bool:
    return bool(re.fullmatch(r"(?:page\s*)?\d{1,4}", line.strip().lower()))


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).lower()


def repair_hyphenation(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1-\2", text)


def reconstruct_paragraphs(text: str) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if looks_like_heading(line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
            continue
        current.append(line)
        if re.search(r"[.!?;:)]$", line):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def looks_like_heading(line: str) -> bool:
    text = line.strip()
    if len(text) > 90 or len(text.split()) > 12:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,:'()/.-]+", text):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9 ,:'()/.-]+", text))


def iter_page_marked_blocks(text: str) -> Iterable[tuple[int, list[str]]]:
    page = 1
    lines: list[str] = []
    for line in text.splitlines():
        marker = PAGE_MARKER_RE.match(line.strip())
        if marker:
            if lines:
                yield page, lines
            page = int(marker.group(1))
            lines = []
        else:
            lines.append(line)
    if lines:
        yield page, lines
