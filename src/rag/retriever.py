"""Lexical retriever for the first RAG+LLM benchmark slice."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.question_classifier import LogicalModifier, QuestionClassifier
from src.question_parser import parse_question


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "did",
    "does",
    "first",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
}

QUERY_EXPANSIONS = {
    "dwarf": ["dwarf", "dwarf planet", "minor planet"],
    "kuiper": ["kuiper", "kuiper belt", "trans-neptunian", "beyond neptune"],
    "discovered": ["discovered", "discovery", "found", "first observed"],
    "located": ["located", "location", "region", "belt"],
    "moons": ["moon", "moons", "satellite", "satellites"],
    "mass": ["mass", "massive", "heavier", "lighter"],
}


@dataclass
class RetrievedChunk:
    chunk: dict[str, Any]
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.chunk)
        data["score"] = round(self.score, 4)
        data["score_reasons"] = self.reasons
        return data


class RagRetriever:
    def __init__(self, chunks_path: str | Path = "data/rag_sources/rag_index/chunks.jsonl") -> None:
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks(self.chunks_path)
        self.question_classifier = QuestionClassifier()

    def retrieve(self, question: str, *, top_k: int = 12, per_source_limit: int = 4) -> list[RetrievedChunk]:
        parsed = parse_question(question)
        classified = self.question_classifier.classify(question)
        query_terms = build_query_terms(question)
        entity_terms = [entity.lower() for entity in parsed["entities"] + classified.major_entities]
        predicate_terms = list(dict.fromkeys(parsed["predicates"] + classified.major_predicates))

        scored: list[RetrievedChunk] = []
        for chunk in self.chunks:
            score, reasons = self._score_chunk(
                chunk,
                query_terms=query_terms,
                entity_terms=entity_terms,
                predicate_terms=predicate_terms,
                has_filter=LogicalModifier.FILTER in classified.logical_modifiers,
                has_ordering=LogicalModifier.ORDERING in classified.logical_modifiers,
            )
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score, reasons=reasons))

        scored.sort(key=lambda item: item.score, reverse=True)
        return cap_per_source(scored, top_k=top_k, per_source_limit=per_source_limit)

    def format_context(self, retrieved_chunks: list[RetrievedChunk], *, max_chars: int = 12000) -> str:
        parts = []
        current_chars = 0
        for index, item in enumerate(retrieved_chunks, start=1):
            chunk = item.chunk
            block = (
                f"[R{index} | source_id={chunk['source_id']} | section={chunk.get('section', '')} | "
                f"score={item.score:.2f}]\n"
                f"URL: {chunk.get('url', '')}\n"
                f"{chunk['text']}\n"
            )
            if current_chars + len(block) > max_chars:
                break
            parts.append(block)
            current_chars += len(block)
        return "\n".join(parts).strip()

    def _score_chunk(
        self,
        chunk: dict[str, Any],
        *,
        query_terms: list[str],
        entity_terms: list[str],
        predicate_terms: list[str],
        has_filter: bool,
        has_ordering: bool,
    ) -> tuple[float, list[str]]:
        text = " ".join([
            chunk.get("title", ""),
            chunk.get("section", ""),
            chunk.get("text", ""),
        ]).lower()
        tokens = Counter(tokenize(text))
        score = 0.0
        reasons: list[str] = []

        for term in query_terms:
            if " " in term:
                if term in text:
                    score += 3.0
                    reasons.append(f"phrase:{term}")
            elif tokens.get(term):
                score += 1.0 + math.log(tokens[term])

        for entity in set(entity_terms):
            if entity and entity in text:
                score += 5.0
                reasons.append(f"entity:{entity}")

        predicate_hints = set(chunk.get("predicate_hints", []))
        for predicate in predicate_terms:
            if predicate in predicate_hints:
                score += 3.0
                reasons.append(f"predicate:{predicate}")

        if has_filter and any(value in text for value in ("kuiper belt", "trans-neptunian", "located")):
            score += 2.5
            reasons.append("filter_context")
        if has_ordering and any(value in text for value in ("discovered", "discovery", "first observed")):
            score += 2.5
            reasons.append("ordering_context")
        if has_ordering and "in order of discovery" in text:
            score += 5.0
            reasons.append("ordered_discovery_section")
        if has_ordering and tokens.get("discovered", 0) >= 2:
            score += 2.0 + math.log(tokens["discovered"])
            reasons.append("multiple_discovery_mentions")
        if "dwarf planet" in text:
            score += 2.0
            reasons.append("dwarf_planet_context")

        return score, reasons

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        chunks = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks


def build_query_terms(question: str) -> list[str]:
    terms = tokenize(question)
    expanded = []
    for term in terms:
        if term in STOPWORDS:
            continue
        expanded.append(term)
        expanded.extend(QUERY_EXPANSIONS.get(term, []))
    return list(dict.fromkeys(expanded))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def cap_per_source(items: list[RetrievedChunk], *, top_k: int, per_source_limit: int) -> list[RetrievedChunk]:
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for item in items:
        source_id = item.chunk.get("source_id", "")
        if counts[source_id] >= per_source_limit:
            continue
        selected.append(item)
        counts[source_id] += 1
        if len(selected) >= top_k:
            break
    return selected
