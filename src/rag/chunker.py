"""Build simple section-aware chunks for RAG retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.rag.pdf_extractor import PAGE_MARKER_RE
from src.rag.structured_satellite_facts import extract_temporal_count_facts


ASTRONOMY_ENTITIES = [
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
    "Ceres",
    "Eris",
    "Makemake",
    "Haumea",
    "Gonggong",
    "Quaoar",
    "Sedna",
    "Orcus",
    "Kuiper Belt",
    "Asteroid Belt",
    "Solar System",
    "Clyde Tombaugh",
]

PREDICATE_HINT_PATTERNS = {
    "classification": r"\b(?:dwarf planet|planet|classified|classification|recognized)\b",
    "location": r"\b(?:kuiper belt|asteroid belt|trans-neptunian|beyond neptune|located|region)\b",
    "discovered_on": r"\b(?:discovered|discovery|found|first observed|year)\b",
    "discovered_by": r"\b(?:discovered by|discoverer|found by|clyde tombaugh)\b",
    "moon_count": r"\b(?:moon|moons|satellite|satellites)\b",
    "ring_count": r"\b(?:ring|rings)\b",
    "mass": r"\b(?:mass|massive|heavier|lighter)\b",
    "distance_from_sun": r"\b(?:distance from the sun|farther from the sun|semi-major axis|orbit)\b",
}


@dataclass
class TextSection:
    heading: str
    text: str


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_chunks_from_documents(
    documents: list[dict[str, Any]],
    *,
    words_per_chunk: int = 140,
    overlap_words: int = 30,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = Path(document["clean_text_path"]).read_text(encoding="utf-8")
        if document.get("source_type") == "pdf":
            chunks.extend(build_pdf_chunks_from_document(
                document,
                text,
                words_per_chunk=words_per_chunk,
                overlap_words=overlap_words,
            ))
            continue
        sections = split_sections(text, fallback_heading=document.get("title") or document["source_id"])
        chunk_index = 0
        for section in sections:
            for chunk_text in split_words(section.text, words_per_chunk, overlap_words):
                if len(chunk_text) < 80:
                    continue
                chunk_id = f"{document['source_id']}_{chunk_index:04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document["document_id"],
                    "source_id": document["source_id"],
                    "source_type": document.get("source_type", "web"),
                    "url": document["url"],
                    "title": document.get("title", document["source_id"]),
                    "file_path": document.get("file_path", ""),
                    "section": section.heading,
                    "text": chunk_text,
                    "target_questions": document.get("target_questions", []),
                    "needed_evidence": document.get("needed_evidence", []),
                    "entities": detect_entities(chunk_text),
                    "predicate_hints": detect_predicate_hints(chunk_text),
                    "tokens_estimate": max(1, len(chunk_text.split())),
                    "trust_level": document.get("trust_level", ""),
                    "content_type": "text",
                })
                chunk_index += 1
        for fact in extract_temporal_count_facts(text):
            chunk_id = f"{document['source_id']}_fact_{chunk_index:04d}"
            predicate_hints = detect_predicate_hints(fact.text)
            if "moon_count" not in predicate_hints:
                predicate_hints.append("moon_count")
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document["document_id"],
                "source_id": document["source_id"],
                "source_type": document.get("source_type", "web"),
                "url": document["url"],
                "title": document.get("title", document["source_id"]),
                "file_path": document.get("file_path", ""),
                "section": fact.heading,
                "text": fact.text,
                "target_questions": document.get("target_questions", []),
                "needed_evidence": document.get("needed_evidence", []),
                "entities": detect_entities(fact.text),
                "predicate_hints": predicate_hints,
                "tokens_estimate": max(1, len(fact.text.split())),
                "trust_level": document.get("trust_level", ""),
                "content_type": "structured_fact",
                "structured_fact_id": fact.fact_id,
                "temporal_fact": fact.metadata(),
            })
            chunk_index += 1
    return chunks


def build_pdf_chunks_from_document(
    document: dict[str, Any],
    text: str,
    *,
    words_per_chunk: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    """Build page-aware chunks for PDF documents without mixing with web sources."""

    chunks: list[dict[str, Any]] = []
    word_pages = pdf_word_pages(text)
    chunk_index = 0
    for chunk_words in split_word_pages(word_pages, words_per_chunk, overlap_words):
        words = [word for word, _page in chunk_words]
        chunk_text = " ".join(words)
        if len(chunk_text) < 80:
            continue
        pages = [page for _word, page in chunk_words]
        page_start = min(pages) if pages else ""
        page_end = max(pages) if pages else ""
        chunk_id = f"{document['source_id']}_{chunk_index:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "document_id": document["document_id"],
            "source_id": document["source_id"],
            "source_type": "pdf",
            "url": document.get("url", ""),
            "title": document.get("title", document["source_id"]),
            "file_path": document.get("file_path", ""),
            "page_start": page_start,
            "page_end": page_end,
            "section": f"Pages {page_start}-{page_end}" if page_start != page_end else f"Page {page_start}",
            "text": chunk_text,
            "target_questions": document.get("target_questions", []),
            "needed_evidence": document.get("needed_evidence", []),
            "entities": detect_entities(chunk_text),
            "predicate_hints": detect_predicate_hints(chunk_text),
            "tokens_estimate": max(1, len(words)),
            "trust_level": document.get("trust_level", ""),
            "content_type": "text",
            "layout_hint": "table_like" if looks_table_like(chunk_text) else "text",
        })
        chunk_index += 1

    clean_text_without_markers = "\n".join(
        line for line in text.splitlines()
        if not PAGE_MARKER_RE.match(line.strip())
    )
    for fact in extract_temporal_count_facts(clean_text_without_markers):
        chunk_id = f"{document['source_id']}_fact_{chunk_index:04d}"
        predicate_hints = detect_predicate_hints(fact.text)
        if "moon_count" not in predicate_hints:
            predicate_hints.append("moon_count")
        chunks.append({
            "chunk_id": chunk_id,
            "document_id": document["document_id"],
            "source_id": document["source_id"],
            "source_type": "pdf",
            "url": document.get("url", ""),
            "title": document.get("title", document["source_id"]),
            "file_path": document.get("file_path", ""),
            "page_start": "",
            "page_end": "",
            "section": fact.heading,
            "text": fact.text,
            "target_questions": document.get("target_questions", []),
            "needed_evidence": document.get("needed_evidence", []),
            "entities": detect_entities(fact.text),
            "predicate_hints": predicate_hints,
            "tokens_estimate": max(1, len(fact.text.split())),
            "trust_level": document.get("trust_level", ""),
            "content_type": "structured_fact",
            "layout_hint": "table_like",
            "structured_fact_id": fact.fact_id,
            "temporal_fact": fact.metadata(),
        })
        chunk_index += 1
    return chunks


def pdf_word_pages(text: str) -> list[tuple[str, int]]:
    page = 1
    word_pages: list[tuple[str, int]] = []
    for line in text.splitlines():
        marker = PAGE_MARKER_RE.match(line.strip())
        if marker:
            page = int(marker.group(1))
            continue
        for word in line.split():
            word_pages.append((word, page))
    return word_pages


def split_word_pages(
    word_pages: list[tuple[str, int]],
    words_per_chunk: int,
    overlap_words: int,
) -> list[list[tuple[str, int]]]:
    if len(word_pages) <= words_per_chunk:
        return [word_pages]
    chunks = []
    start = 0
    step = max(1, words_per_chunk - overlap_words)
    while start < len(word_pages):
        end = min(len(word_pages), start + words_per_chunk)
        chunks.append(word_pages[start:end])
        if end == len(word_pages):
            break
        start += step
    return chunks


def looks_table_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    digit_heavy = sum(1 for line in lines if len(re.findall(r"\b\d{2,4}\b", line)) >= 2)
    if digit_heavy >= 2:
        return True
    return len(re.findall(r"\b\d{4}\b", text)) >= 4 and len(re.findall(r"\s{2,}", text)) >= 3


def split_sections(text: str, fallback_heading: str) -> list[TextSection]:
    lines = text.splitlines()
    sections: list[TextSection] = []
    current_heading = fallback_heading
    current_lines: list[str] = []

    for line in lines:
        heading = parse_heading(line)
        if heading:
            if current_lines:
                sections.append(TextSection(current_heading, "\n".join(current_lines).strip()))
            current_heading = heading
            current_lines = []
            continue
        if line.strip():
            current_lines.append(line.strip())

    if current_lines:
        sections.append(TextSection(current_heading, "\n".join(current_lines).strip()))
    return sections or [TextSection(fallback_heading, text.strip())]


def parse_heading(line: str) -> str | None:
    text = line.strip()
    wiki_heading = re.fullmatch(r"=+\s*(.*?)\s*=+", text)
    if wiki_heading:
        return wiki_heading.group(1).strip()
    if len(text) <= 80 and re.fullmatch(r"[A-Z][A-Za-z0-9 ,:'()/.-]+", text):
        return text
    return None


def split_words(text: str, words_per_chunk: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [" ".join(words)]

    chunks = []
    start = 0
    step = max(1, words_per_chunk - overlap_words)
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def detect_entities(text: str) -> list[str]:
    found = []
    text_lower = text.lower()
    for entity in ASTRONOMY_ENTITIES:
        if re.search(rf"\b{re.escape(entity.lower())}\b", text_lower):
            found.append(entity)
    return found


def detect_predicate_hints(text: str) -> list[str]:
    found = []
    text_lower = text.lower()
    for predicate, pattern in PREDICATE_HINT_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(predicate)
    return found
