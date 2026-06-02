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


@dataclass(frozen=True)
class PdfTextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


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
                pages.append(extract_page_text_with_layout(page))
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


def extract_page_text_with_layout(page: object) -> str:
    """Extract one PyMuPDF page with conservative multi-column ordering.

    PyMuPDF's plain text mode can interleave columns depending on the PDF text
    layer. Blocks provide coordinates, so clean two- and three-column bodies can
    be sorted by column before downstream paragraph cleanup.
    """

    blocks = page_text_blocks(page)
    if not blocks:
        get_text = getattr(page, "get_text")
        return get_text("text") or ""
    width = float(getattr(getattr(page, "rect", None), "width", 0.0) or 0.0)
    ordered = order_page_blocks(blocks, width)
    return "\n\n".join(block.text.strip() for block in ordered if block.text.strip())


def page_text_blocks(page: object) -> list[PdfTextBlock]:
    get_text = getattr(page, "get_text")
    raw_blocks = get_text("blocks") or []
    blocks: list[PdfTextBlock] = []
    for raw in raw_blocks:
        if len(raw) < 5:
            continue
        x0, y0, x1, y1, text = raw[:5]
        if not isinstance(text, str) or not text.strip():
            continue
        block_type = raw[6] if len(raw) > 6 else 0
        if block_type != 0:
            continue
        blocks.append(PdfTextBlock(
            x0=float(x0),
            y0=float(y0),
            x1=float(x1),
            y1=float(y1),
            text=text.strip(),
        ))
    return blocks


def order_page_blocks(blocks: list[PdfTextBlock], page_width: float) -> list[PdfTextBlock]:
    """Return blocks in reading order, using up to three clean columns."""

    if len(blocks) < 4 or page_width <= 0:
        return sorted(blocks, key=lambda block: (block.y0, block.x0))

    full_width, body = split_full_width_blocks(blocks, page_width)
    columns = detect_columns(body, page_width)
    if columns is None:
        return sorted(blocks, key=lambda block: (block.y0, block.x0))

    column_blocks = [block for column in columns for block in column]
    column_top = min(block.y0 for block in column_blocks)
    column_bottom = max(block.y1 for block in column_blocks)
    before = [block for block in full_width if block.y1 <= column_top]
    after = [block for block in full_width if block.y1 > column_top]

    return [
        *sorted(before, key=lambda block: (block.y0, block.x0)),
        *[
            block
            for column in columns
            for block in sorted(column, key=lambda block: (block.y0, block.x0))
        ],
        *sorted(after, key=lambda block: (block.y0, block.x0)),
    ]


def split_full_width_blocks(
    blocks: list[PdfTextBlock],
    page_width: float,
) -> tuple[list[PdfTextBlock], list[PdfTextBlock]]:
    full_width = []
    body = []
    for block in blocks:
        block_width = block.x1 - block.x0
        spans_page = block.x0 <= page_width * 0.20 and block.x1 >= page_width * 0.80
        if block_width >= page_width * 0.65 or spans_page:
            full_width.append(block)
        else:
            body.append(block)
    return full_width, body


def detect_two_columns(
    blocks: list[PdfTextBlock],
    page_width: float,
) -> tuple[list[PdfTextBlock], list[PdfTextBlock]] | None:
    columns = detect_columns(blocks, page_width)
    if columns is None or len(columns) != 2:
        return None
    return columns[0], columns[1]


def detect_columns(
    blocks: list[PdfTextBlock],
    page_width: float,
) -> list[list[PdfTextBlock]] | None:
    """Detect clean two- or three-column text bodies.

    More than three clusters, overlapping columns, or extremely sparse columns
    are treated as complex layout and left in top-to-bottom block order.
    """

    if len(blocks) < 4:
        return None

    columns = cluster_blocks_by_x_position(blocks, page_width)
    if columns is None or len(columns) not in {2, 3}:
        return None
    if not columns_have_clear_gaps(columns, page_width):
        return None
    if not looks_like_columnar_text(columns):
        return None
    return columns


def cluster_blocks_by_x_position(
    blocks: list[PdfTextBlock],
    page_width: float,
) -> list[list[PdfTextBlock]] | None:
    sorted_blocks = sorted(blocks, key=lambda block: ((block.x0 + block.x1) / 2, block.y0))
    columns: list[list[PdfTextBlock]] = []
    for block in sorted_blocks:
        center = (block.x0 + block.x1) / 2
        if not columns:
            columns.append([block])
            continue

        last_column = columns[-1]
        last_center = median_block_center(last_column)
        center_gap = center - last_center
        if center_gap >= page_width * 0.18:
            columns.append([block])
            if len(columns) > 3:
                return None
        else:
            last_column.append(block)

    if len(columns) < 2:
        return None
    if any(len(column) < 2 for column in columns):
        return None
    return columns


def median_block_center(blocks: list[PdfTextBlock]) -> float:
    centers = sorted((block.x0 + block.x1) / 2 for block in blocks)
    middle = len(centers) // 2
    if len(centers) % 2:
        return centers[middle]
    return (centers[middle - 1] + centers[middle]) / 2


def columns_have_clear_gaps(columns: list[list[PdfTextBlock]], page_width: float) -> bool:
    for left, right in zip(columns, columns[1:]):
        left_center = median_block_center(left)
        right_center = median_block_center(right)
        if right_center - left_center < page_width * 0.18:
            return False
    return True


def looks_like_columnar_text(columns: list[list[PdfTextBlock]]) -> bool:
    """Reject cover/list layouts while allowing normal column text lines."""

    blocks = [block for column in columns for block in column]
    if len(blocks) <= 8:
        return True
    word_counts = [len(re.findall(r"\w+", block.text)) for block in blocks]
    average_words = sum(word_counts) / len(word_counts)
    longish_blocks = sum(count >= 6 for count in word_counts)
    return average_words >= 5.0 and longish_blocks >= len(blocks) * 0.45


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
