"""Build coverage-preserving routing units for hierarchical RAG retrieval."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.rag.chunker import detect_entities, detect_predicate_hints, split_sections


DEFAULT_ROUTING_UNITS_PATH = Path("data/rag_sources/rag_index/routing_units.jsonl")
DEFAULT_ROUTING_EMBEDDINGS_PATH = Path("data/rag_sources/rag_index/routing_embeddings_bge_m3.jsonl")
DEFAULT_ROUTE_WORDS = 520
DEFAULT_ROUTE_OVERLAP_WORDS = 65


def build_routing_units(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    words_per_route: int = DEFAULT_ROUTE_WORDS,
    overlap_words: int = DEFAULT_ROUTE_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    if words_per_route <= overlap_words:
        raise ValueError("words_per_route must be greater than overlap_words")

    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk["source_id"]].append(chunk)

    units: list[dict[str, Any]] = []
    for document in documents:
        source_id = document["source_id"]
        source_chunks = chunks_by_source.get(source_id, [])
        text_chunks = [chunk for chunk in source_chunks if chunk.get("content_type") != "structured_fact"]
        fact_chunks = [chunk for chunk in source_chunks if chunk.get("content_type") == "structured_fact"]
        text = Path(document["clean_text_path"]).read_text(encoding="utf-8")
        sections = split_sections(text, fallback_heading=document.get("title") or source_id)
        title = document.get("title", source_id)
        url = document.get("url", "")

        units.append(build_source_identity_route(document))
        units.append(build_source_catalog_route(
            document,
            sections=sections,
            source_chunks=source_chunks,
        ))

        routing_text = "\n\n".join(
            f"Section: {section.heading}\n{section.text}"
            for section in sections
            if section.text.strip()
        )
        for window_index, (window_text, word_start, word_end) in enumerate(
            split_window_spans(routing_text, words_per_route, overlap_words)
        ):
            child_chunk_ids = matching_child_chunk_ids(window_text, text_chunks)
            route_id = f"{source_id}__section_{window_index:04d}"
            units.append({
                "route_id": route_id,
                "chunk_id": route_id,
                "document_id": document["document_id"],
                "source_id": source_id,
                "route_type": "section_window",
                "url": url,
                "title": title,
                "section": "Document Routing Window",
                "text": window_text,
                "entities": detect_entities(window_text),
                "predicate_hints": detect_predicate_hints(window_text),
                "child_chunk_ids": child_chunk_ids,
                "window_index": window_index,
                "word_start": word_start,
                "word_end": word_end,
                "tokens_estimate": max(1, len(window_text.split())),
                "trust_level": document.get("trust_level", ""),
            })

        for fact_index, chunk in enumerate(fact_chunks):
            route_id = f"{source_id}__fact_{fact_index:04d}"
            units.append({
                "route_id": route_id,
                "chunk_id": route_id,
                "document_id": document["document_id"],
                "source_id": source_id,
                "route_type": "structured_fact",
                "url": url,
                "title": title,
                "section": chunk.get("section", "Structured Fact"),
                "text": chunk["text"],
                "entities": chunk.get("entities", []),
                "predicate_hints": chunk.get("predicate_hints", []),
                "child_chunk_ids": [chunk["chunk_id"]],
                "structured_fact_id": chunk.get("structured_fact_id", ""),
                "temporal_fact": chunk.get("temporal_fact", {}),
                "tokens_estimate": max(1, len(chunk["text"].split())),
                "trust_level": document.get("trust_level", ""),
            })
    return units


def build_source_identity_route(document: dict[str, Any]) -> dict[str, Any]:
    """Build a precise source locator from title and normalized URL topics."""
    source_id = document["source_id"]
    title = document.get("title", source_id)
    url = document.get("url", "")
    url_topics = url_topics_from_url(url)
    identity_text = "\n".join([
        f"URL topics: {url_topics}",
        f"Source type: {document.get('trust_level', '')}",
    ])
    identity_input = f"{title}\n{url_topics}"
    route_id = f"{source_id}__identity"
    return {
        "route_id": route_id,
        "chunk_id": route_id,
        "document_id": document["document_id"],
        "source_id": source_id,
        "route_type": "source_identity",
        "url": url,
        "title": title,
        "section": "Source Identity",
        "text": identity_text,
        "entities": detect_entities(identity_input),
        "predicate_hints": detect_predicate_hints(identity_input),
        "child_chunk_ids": [],
        "tokens_estimate": max(1, len(identity_text.split())),
        "trust_level": document.get("trust_level", ""),
    }


def build_source_catalog_route(
    document: dict[str, Any],
    *,
    sections: list[Any],
    source_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    source_id = document["source_id"]
    url = document.get("url", "")
    headings = list(dict.fromkeys(section.heading for section in sections))
    entities = list(dict.fromkeys(
        entity
        for chunk in source_chunks
        for entity in chunk.get("entities", [])
    ))
    predicates = list(dict.fromkeys(
        predicate
        for chunk in source_chunks
        for predicate in chunk.get("predicate_hints", [])
    ))
    catalog_text = "\n".join([
        f"Sections: {', '.join(headings)}",
        f"Entities: {', '.join(entities)}",
        f"Predicates: {', '.join(predicates)}",
    ])
    route_id = f"{source_id}__catalog"
    return {
        "route_id": route_id,
        "chunk_id": route_id,
        "document_id": document["document_id"],
        "source_id": source_id,
        "route_type": "source_catalog",
        "url": url,
        # The title belongs to source_identity; omit it here to avoid counting it twice.
        "title": "",
        "section": "Source Catalog",
        "text": catalog_text,
        "entities": entities,
        "predicate_hints": predicates,
        "child_chunk_ids": [],
        "tokens_estimate": max(1, len(catalog_text.split())),
        "trust_level": document.get("trust_level", ""),
    }


def split_window_spans(text: str, words_per_route: int, overlap_words: int) -> list[tuple[str, int, int]]:
    words = text.split()
    if not words:
        return []
    if len(words) <= words_per_route:
        return [(" ".join(words), 0, len(words))]

    windows: list[tuple[str, int, int]] = []
    start = 0
    step = words_per_route - overlap_words
    while start < len(words):
        end = min(len(words), start + words_per_route)
        windows.append((" ".join(words[start:end]), start, end))
        if end == len(words):
            break
        start += step
    return windows


def matching_child_chunk_ids(route_text: str, chunks: list[dict[str, Any]]) -> list[str]:
    normalized_route = normalize_words(route_text)
    matches: list[str] = []
    for chunk in chunks:
        words = normalize_words(chunk.get("text", "")).split()
        if not words:
            continue
        anchor_size = min(12, len(words))
        first_anchor = " ".join(words[:anchor_size])
        last_anchor = " ".join(words[-anchor_size:])
        if first_anchor in normalized_route or last_anchor in normalized_route:
            matches.append(chunk["chunk_id"])
    return matches


def url_topics_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path).replace("_", " ").replace("-", " ")
    return " ".join(path.split("/")).strip()


def normalize_words(text: str) -> str:
    return " ".join(text.lower().split())
