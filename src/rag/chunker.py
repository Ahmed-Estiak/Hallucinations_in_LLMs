"""Build simple section-aware chunks for RAG retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
                    "url": document["url"],
                    "title": document.get("title", document["source_id"]),
                    "section": section.heading,
                    "text": chunk_text,
                    "target_questions": document.get("target_questions", []),
                    "needed_evidence": document.get("needed_evidence", []),
                    "entities": detect_entities(chunk_text),
                    "predicate_hints": detect_predicate_hints(chunk_text),
                    "tokens_estimate": max(1, len(chunk_text.split())),
                    "content_type": "text",
                })
                chunk_index += 1
    return chunks


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

