"""Text extraction and cleanup helpers for copyable PDF RAG sources."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAGE_MARKER_RE = re.compile(r"^\[\[PAGE\s+(\d+)\]\]$")
TABLE_CONTINUES_FROM_PREVIOUS = "[[TABLE CONTINUES FROM PREVIOUS PAGE]]"
TABLE_CONTINUES_ON_NEXT = "[[TABLE CONTINUES ON NEXT PAGE]]"


@dataclass(frozen=True)
class ExtractedPdf:
    text: str
    page_count: int
    raw_char_count: int
    cleaned_char_count: int
    repeated_lines_removed: int
    page_diagnostics: list[dict[str, object]]


@dataclass(frozen=True)
class PdfTextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True)
class PageExtraction:
    text: str
    diagnostics: dict[str, object]


def extract_pdf_text(path: str | Path) -> ExtractedPdf:
    """Extract text from a copyable PDF and apply conservative cleanup.

    The extractor deliberately does not remove references/bibliography sections.
    It only removes repeated boilerplate lines, repairs hyphenated line breaks,
    and reconstructs paragraphs enough to make downstream chunking less noisy.
    """

    pdf_path = Path(path)
    extracted_pages = _extract_pages(pdf_path)
    raw_pages = [page.text for page in extracted_pages]
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
    cleaned_pages = add_cross_page_table_markers(
        cleaned_pages,
        [page.diagnostics for page in extracted_pages],
    )

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
        page_diagnostics=[page.diagnostics for page in extracted_pages],
    )


def _extract_pages(path: Path) -> list[PageExtraction]:
    try:
        import fitz  # type: ignore

        pages = []
        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                pages.append(extract_page_with_layout(page, page_number=page_number))
        return pages
    except ImportError:
        pass

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return [
            PageExtraction(
                text=page.extract_text() or "",
                diagnostics={
                    "page": index,
                    "layout_class": "unknown",
                    "chosen_mode": "pypdf_plain_text",
                    "column_count": 0,
                    "block_count": 0,
                    "line_count": 0,
                    "warning": "pymupdf_unavailable",
                },
            )
            for index, page in enumerate(reader.pages, start=1)
        ]
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF or pypdf. Install one of them, "
            "for example: pip install pymupdf"
        ) from exc


def extract_page_text_with_layout(page: object) -> str:
    return extract_page_with_layout(page, page_number=0).text


def extract_page_with_layout(page: object, *, page_number: int) -> PageExtraction:
    """Extract one PyMuPDF page with conservative multi-column ordering.

    PyMuPDF's plain text mode can interleave columns depending on the PDF text
    layer. Blocks provide coordinates, so clean two- and three-column bodies can
    be sorted by column before downstream paragraph cleanup.
    """

    get_text = getattr(page, "get_text")
    plain_text = get_text("text") or ""
    blocks = page_text_blocks(page)
    lines = page_text_lines(page)
    if not blocks:
        return PageExtraction(
            text=plain_text,
            diagnostics={
                "page": page_number,
                "layout_class": "unknown",
                "chosen_mode": "plain_text_no_blocks",
                "column_count": 0,
                "block_count": 0,
                "line_count": len(lines),
                "warning": "no_text_blocks",
            },
        )
    width = float(getattr(getattr(page, "rect", None), "width", 0.0) or 0.0)
    height = float(getattr(getattr(page, "rect", None), "height", 0.0) or 0.0)
    layout = classify_page_layout(blocks, width)
    table_bands = detect_table_regions(lines or blocks)
    if table_bands and layout["layout_class"] != "poster_or_cover":
        text = format_page_with_table_regions(blocks, lines or blocks, width, table_bands)
        layout = {
            **layout,
            "chosen_mode": "region_order_with_tables",
            "table_region_count": len(table_bands),
            "warning": append_warning(str(layout.get("warning", "")), "table_regions_row_ordered"),
        }
    elif layout["chosen_mode"] == "plain_text":
        text = plain_text
    elif layout["chosen_mode"] == "row_order_lines":
        ordered = sorted(lines or blocks, key=lambda block: (block.y0, block.x0))
        text = "\n".join(block.text.strip() for block in ordered if block.text.strip())
    else:
        ordered = order_page_blocks(blocks, width)
        text = "\n\n".join(block.text.strip() for block in ordered if block.text.strip())
    diagnostics = {
        "page": page_number,
        **layout,
        "table_starts_page": table_starts_page(table_bands, height),
        "table_ends_page": table_ends_page(table_bands, height),
        "block_count": len(blocks),
        "line_count": len(lines),
    }
    return PageExtraction(text=text, diagnostics=diagnostics)


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


def page_text_lines(page: object) -> list[PdfTextBlock]:
    """Extract line-level text boxes from PyMuPDF's dict representation."""

    get_text = getattr(page, "get_text")
    raw = get_text("dict") or {}
    lines: list[PdfTextBlock] = []
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = [
                span for span in line.get("spans", [])
                if isinstance(span.get("text"), str) and span.get("text", "").strip()
            ]
            if not spans:
                continue
            text = join_line_spans(spans)
            bbox = line.get("bbox") or spans[0].get("bbox")
            if not bbox or len(bbox) < 4 or not text:
                continue
            x0, y0, x1, y1 = bbox[:4]
            lines.append(PdfTextBlock(
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                text=text,
            ))
    return lines


def join_line_spans(spans: list[dict]) -> str:
    """Join PyMuPDF spans without losing word spaces across span boundaries."""

    parts: list[str] = []
    previous_x1: float | None = None
    for span in spans:
        text = str(span.get("text", ""))
        if not text:
            continue
        bbox = span.get("bbox") or []
        x0 = float(bbox[0]) if len(bbox) >= 4 else None
        if (
            parts
            and previous_x1 is not None
            and x0 is not None
            and x0 - previous_x1 > 1.5
            and not parts[-1].endswith((" ", "-", "/", "\u00ad"))
            and not text.startswith((" ", ".", ",", ";", ":", ")", "]"))
        ):
            parts.append(" ")
        parts.append(text)
        if len(bbox) >= 4:
            previous_x1 = float(bbox[2])
    return "".join(parts).strip()


def classify_page_layout(blocks: list[PdfTextBlock], page_width: float) -> dict[str, object]:
    """Classify a page enough to choose a safe extraction order."""

    full_width, body = split_full_width_blocks(blocks, page_width)
    columns = detect_columns(body, page_width)
    metrics = layout_metrics(blocks)
    if columns is not None:
        column_count = len(columns)
        return {
            "layout_class": f"narrative_{column_count}_column",
            "chosen_mode": "column_order",
            "column_count": column_count,
            "table_region_count": 0,
            **metrics,
            "warning": "",
        }

    if looks_table_like_page(blocks):
        return {
            "layout_class": "table_like",
            "chosen_mode": "row_order_lines",
            "column_count": 0,
            "table_region_count": 1,
            **metrics,
            "warning": "table_like_row_order_no_structured_table_parse",
        }
    if looks_poster_or_cover_page(blocks):
        return {
            "layout_class": "poster_or_cover",
            "chosen_mode": "plain_text",
            "column_count": 0,
            "table_region_count": 0,
            **metrics,
            "warning": "plain_text_fallback_for_poster_or_cover",
        }
    if len(full_width) >= max(4, len(blocks) * 0.5):
        layout_class = "narrative_single_column"
    else:
        layout_class = "mixed"
    return {
        "layout_class": layout_class,
        "chosen_mode": "plain_text" if layout_class == "mixed" else "row_order_lines",
        "column_count": 1 if layout_class == "narrative_single_column" else 0,
        "table_region_count": 0,
        **metrics,
        "warning": "mixed_layout_plain_text_fallback" if layout_class == "mixed" else "",
    }


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


def format_page_with_table_regions(
    blocks: list[PdfTextBlock],
    lines: list[PdfTextBlock],
    page_width: float,
    table_bands: list[tuple[float, float]],
) -> str:
    """Format a page while preserving only detected table regions row-wise."""

    parts: list[str] = []
    cursor = min((block.y0 for block in blocks), default=0.0)
    for start, end in table_bands:
        before_blocks = [
            block for block in blocks
            if cursor <= block.y0 and block.y1 < start
        ]
        append_ordered_blocks(parts, before_blocks, page_width)
        table_lines = [
            line for line in lines
            if start <= line.y0 <= end
        ]
        table_text = format_table_lines(table_lines)
        if table_text:
            parts.append(table_text)
        cursor = end
    after_blocks = [block for block in blocks if block.y0 > cursor]
    append_ordered_blocks(parts, after_blocks, page_width)
    return "\n\n".join(part for part in parts if part.strip())


def add_cross_page_table_markers(
    pages: list[str],
    diagnostics: list[dict[str, object]],
) -> list[str]:
    """Add safe hints for tables split across adjacent PDF pages."""

    if not pages:
        return pages
    marked = list(pages)
    for index, page_text in enumerate(marked):
        starts_from_previous = (
            index > 0
            and bool(diagnostics[index - 1].get("table_ends_page"))
            and bool(diagnostics[index].get("table_starts_page"))
        )
        continues_next = (
            index + 1 < len(marked)
            and bool(diagnostics[index].get("table_ends_page"))
            and bool(diagnostics[index + 1].get("table_starts_page"))
        )
        if starts_from_previous and TABLE_CONTINUES_FROM_PREVIOUS not in page_text:
            page_text = f"{TABLE_CONTINUES_FROM_PREVIOUS}\n\n{page_text}".strip()
        if continues_next and TABLE_CONTINUES_ON_NEXT not in page_text:
            page_text = f"{page_text.strip()}\n\n{TABLE_CONTINUES_ON_NEXT}"
        marked[index] = page_text
    return marked


def append_ordered_blocks(parts: list[str], blocks: list[PdfTextBlock], page_width: float) -> None:
    if not blocks:
        return
    ordered = order_page_blocks(blocks, page_width)
    text = "\n\n".join(block.text.strip() for block in ordered if block.text.strip())
    if text.strip():
        parts.append(text.strip())


def detect_table_regions(lines: list[PdfTextBlock]) -> list[tuple[float, float]]:
    """Detect table-like row bands inside a page, not only whole table pages."""

    if len(lines) < 6:
        return []
    rows = group_blocks_by_row(lines, y_tolerance=3.0)
    table_row_indexes = [
        index for index, row in enumerate(rows)
        if looks_like_table_row(row)
    ]
    if not table_row_indexes:
        return []

    bands: list[tuple[float, float]] = []
    run: list[int] = []
    for index in table_row_indexes:
        if not run or index == run[-1] + 1:
            run.append(index)
            continue
        append_table_band(bands, rows, run)
        run = [index]
    append_table_band(bands, rows, run)
    return bands


def table_starts_page(table_bands: list[tuple[float, float]], page_height: float) -> bool:
    if not table_bands or page_height <= 0:
        return False
    start, _end = table_bands[0]
    return start <= page_height * 0.20


def table_ends_page(table_bands: list[tuple[float, float]], page_height: float) -> bool:
    if not table_bands or page_height <= 0:
        return False
    _start, end = table_bands[-1]
    return end >= page_height * 0.80


def append_table_band(
    bands: list[tuple[float, float]],
    rows: list[list[PdfTextBlock]],
    run: list[int],
) -> None:
    if len(run) < 2:
        return
    row_blocks = [block for index in run for block in rows[index]]
    bands.append((
        min(block.y0 for block in row_blocks) - 1.0,
        max(block.y1 for block in row_blocks) + 1.0,
    ))


def looks_like_table_row(row: list[PdfTextBlock]) -> bool:
    if len(row) < 3:
        return False
    texts = [block.text.strip() for block in row if block.text.strip()]
    if len(texts) < 3:
        return False
    word_counts = [len(re.findall(r"\w+", text)) for text in texts]
    numeric_cells = sum(bool(re.search(r"\d", text)) for text in texts)
    short_cells = sum(count <= 4 for count in word_counts)
    return bool(short_cells >= len(texts) * 0.65 and numeric_cells >= 2)


def format_table_lines(lines: list[PdfTextBlock]) -> str:
    rows = group_blocks_by_row(lines, y_tolerance=3.0)
    formatted_rows = []
    for row in rows:
        cells = [cell.text.strip() for cell in sorted(row, key=lambda cell: cell.x0) if cell.text.strip()]
        if cells:
            formatted_rows.append(" | ".join(cells))
    return "\n".join(formatted_rows)


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


def layout_metrics(blocks: list[PdfTextBlock]) -> dict[str, object]:
    word_counts = [len(re.findall(r"\w+", block.text)) for block in blocks]
    if not word_counts:
        return {
            "avg_words_per_block": 0.0,
            "short_block_ratio": 0.0,
            "numeric_block_ratio": 0.0,
        }
    short_blocks = sum(count <= 4 for count in word_counts)
    numeric_blocks = sum(bool(re.search(r"\d", block.text)) for block in blocks)
    return {
        "avg_words_per_block": round(sum(word_counts) / len(word_counts), 2),
        "short_block_ratio": round(short_blocks / len(blocks), 3),
        "numeric_block_ratio": round(numeric_blocks / len(blocks), 3),
    }


def looks_table_like_page(blocks: list[PdfTextBlock]) -> bool:
    if len(blocks) < 8:
        return False
    metrics = layout_metrics(blocks)
    row_groups = group_blocks_by_row(blocks)
    multi_cell_rows = sum(1 for row in row_groups if len(row) >= 3)
    has_many_rows = multi_cell_rows >= max(3, len(row_groups) * 0.25)
    return bool(
        has_many_rows
        and metrics["short_block_ratio"] >= 0.45
        and metrics["numeric_block_ratio"] >= 0.25
    )


def looks_poster_or_cover_page(blocks: list[PdfTextBlock]) -> bool:
    if len(blocks) < 8:
        return False
    metrics = layout_metrics(blocks)
    word_counts = [len(re.findall(r"\w+", block.text)) for block in blocks]
    very_short = sum(count <= 3 for count in word_counts)
    all_caps_or_title = sum(looks_like_heading(block.text) for block in blocks)
    return bool(
        metrics["short_block_ratio"] >= 0.55
        and (very_short >= len(blocks) * 0.35 or all_caps_or_title >= len(blocks) * 0.35)
    )


def group_blocks_by_row(blocks: list[PdfTextBlock], *, y_tolerance: float = 4.0) -> list[list[PdfTextBlock]]:
    rows: list[list[PdfTextBlock]] = []
    for block in sorted(blocks, key=lambda item: (item.y0, item.x0)):
        for row in rows:
            if abs(row[0].y0 - block.y0) <= y_tolerance:
                row.append(block)
                break
        else:
            rows.append([block])
    for row in rows:
        row.sort(key=lambda item: item.x0)
    return rows


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


def append_warning(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing.split(";"):
        return existing
    return f"{existing};{addition}"


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
        if looks_like_table_text_line(line):
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


def looks_like_table_text_line(line: str) -> bool:
    return line.count("|") >= 2


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
