"""Page-level signal filtering for PDF-only RAG retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.rag.chunker import detect_entities, detect_predicate_hints
from src.rag.pdf_extractor import iter_page_marked_blocks
from src.rag.retrieval_intent import RetrievalIntent


def build_page_signals_from_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        if document.get("source_type") != "pdf":
            continue
        text = Path(document["clean_text_path"]).read_text(encoding="utf-8")
        for page, lines in iter_page_marked_blocks(text):
            page_text = "\n".join(lines).strip()
            rows.append({
                "document_id": document["document_id"],
                "source_id": document["source_id"],
                "title": document.get("title", document["source_id"]),
                "page": page,
                "entities": detect_entities(page_text),
                "predicate_hints": detect_predicate_hints(page_text),
                "tokens_estimate": max(1, len(page_text.split())),
            })
    return rows


def filter_pdf_chunks_by_page_signals(
    chunks: list[dict[str, Any]],
    page_signals: list[dict[str, Any]],
    intent: RetrievalIntent,
    *,
    class_members: dict[str, list[str]],
    neighbor_radius: int = 1,
) -> list[dict[str, Any]]:
    """Return chunks from PDFs with strong page hits plus neighboring pages.

    If no PDF has any hit page, the original chunk list is returned as a global
    fallback. If one PDF has no hit pages while another does, the no-hit PDF is
    dropped for this query.
    """

    if not chunks or not page_signals:
        return chunks
    if any(chunk.get("source_type") != "pdf" for chunk in chunks):
        return chunks

    signals = strong_query_signals(intent, class_members=class_members)
    if not signals["entities"] and not signals["predicates"]:
        return chunks

    max_page_by_source: dict[str, int] = defaultdict(int)
    hit_pages_by_source: dict[str, set[int]] = defaultdict(set)
    for row in page_signals:
        source_id = str(row.get("source_id", ""))
        page = int(row.get("page", 0) or 0)
        if not source_id or page <= 0:
            continue
        max_page_by_source[source_id] = max(max_page_by_source[source_id], page)
        if page_has_strong_signal(row, signals):
            hit_pages_by_source[source_id].add(page)

    if not hit_pages_by_source:
        return chunks

    selected_pages_by_source = {
        source_id: expand_pages(pages, max_page_by_source.get(source_id, 0), radius=neighbor_radius)
        for source_id, pages in hit_pages_by_source.items()
    }
    filtered = [
        chunk for chunk in chunks
        if chunk_overlaps_selected_pages(chunk, selected_pages_by_source)
    ]
    return filtered or chunks


def strong_query_signals(
    intent: RetrievalIntent,
    *,
    class_members: dict[str, list[str]],
) -> dict[str, set[str]]:
    entities = {normalize_signal(entity) for entity in intent.entity_terms if entity}
    if intent.target_class and intent.target_class in class_members:
        entities.update(normalize_signal(entity) for entity in class_members[intent.target_class])
    for condition in intent.filter_conditions:
        for key in ("value", "reference_entity"):
            value = normalize_signal(str(condition.get(key, "")))
            if value:
                entities.add(value)
    predicates = {normalize_signal(predicate) for predicate in intent.predicate_terms if predicate}
    if intent.ordering_attribute:
        predicates.add(normalize_signal(intent.ordering_attribute))
    return {
        "entities": {value for value in entities if value},
        "predicates": {value for value in predicates if value},
    }


def page_has_strong_signal(row: dict[str, Any], signals: dict[str, set[str]]) -> bool:
    page_entities = {normalize_signal(entity) for entity in row.get("entities", [])}
    page_predicates = {normalize_signal(predicate) for predicate in row.get("predicate_hints", [])}
    has_entity_hit = bool(page_entities & signals["entities"])
    has_predicate_hit = bool(page_predicates & signals["predicates"])
    if signals["entities"] and signals["predicates"]:
        return has_entity_hit and has_predicate_hit
    if signals["entities"]:
        return has_entity_hit
    return has_predicate_hit


def expand_pages(pages: Iterable[int], max_page: int, *, radius: int) -> set[int]:
    selected: set[int] = set()
    for page in pages:
        for candidate in range(page - radius, page + radius + 1):
            if 1 <= candidate <= max_page:
                selected.add(candidate)
    return selected


def chunk_overlaps_selected_pages(
    chunk: dict[str, Any],
    selected_pages_by_source: dict[str, set[int]],
) -> bool:
    source_id = str(chunk.get("source_id", ""))
    selected_pages = selected_pages_by_source.get(source_id)
    if not selected_pages:
        return False
    page_start = parse_page_number(chunk.get("page_start"))
    page_end = parse_page_number(chunk.get("page_end"))
    if page_start is None or page_end is None:
        return chunk_matches_selected_source(chunk)
    if page_start > page_end:
        page_start, page_end = page_end, page_start
    return any(page in selected_pages for page in range(page_start, page_end + 1))


def chunk_matches_selected_source(chunk: dict[str, Any]) -> bool:
    return chunk.get("content_type") in {"structured_fact", "resolved_fact", "resolved_fact_table"}


def parse_page_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def normalize_signal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
