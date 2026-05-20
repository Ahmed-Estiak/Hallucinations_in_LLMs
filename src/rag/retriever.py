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
from src.rag.retriever_terms import build_query_terms, infer_query_predicates, tokenize
from src.rag.source_selector import SourceSelection, SourceSelector


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


@dataclass
class RagRetrievalResult:
    retrieved_chunks: list[RetrievedChunk]
    retrieval_mode: str
    source_selection: SourceSelection | None = None
    fallback_used: bool = False
    fallback_reason: str = ""


class RagRetriever:
    def __init__(
        self,
        chunks_path: str | Path = "data/rag_sources/rag_index/chunks.jsonl",
        documents_path: str | Path = "data/rag_sources/rag_index/documents.jsonl",
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks(self.chunks_path)
        self.documents_path = Path(documents_path)
        self.question_classifier = QuestionClassifier()
        self.source_selector = SourceSelector(self.chunks, documents_path=self.documents_path)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 12,
        per_source_limit: int = 4,
        mode: str = "global",
        top_n_sources: int = 5,
    ) -> list[RetrievedChunk]:
        return self.retrieve_with_details(
            question,
            top_k=top_k,
            per_source_limit=per_source_limit,
            mode=mode,
            top_n_sources=top_n_sources,
        ).retrieved_chunks

    def retrieve_with_details(
        self,
        question: str,
        *,
        top_k: int = 12,
        per_source_limit: int = 4,
        mode: str = "global",
        top_n_sources: int = 5,
    ) -> RagRetrievalResult:
        if mode not in {"global", "auto-source"}:
            raise ValueError("mode must be one of: global, auto-source")

        source_selection = None
        selected_source_ids = None
        fallback_used = False
        fallback_reason = ""
        if mode == "auto-source":
            source_selection = self.source_selector.select(question, top_n_sources=top_n_sources)
            selected_source_ids = set(source_selection.selected_source_ids)
            if not selected_source_ids:
                fallback_used = True
                fallback_reason = "no_sources_selected"
                selected_source_ids = None

        effective_per_source_limit = self._effective_per_source_limit(question, per_source_limit)
        retrieved = self._retrieve_from_chunks(
            question,
            chunks=[
                chunk for chunk in self.chunks
                if selected_source_ids is None or chunk.get("source_id") in selected_source_ids
            ],
            top_k=top_k,
            per_source_limit=effective_per_source_limit,
        )

        if mode == "auto-source" and is_weak_retrieval(retrieved):
            fallback_used = True
            fallback_reason = "weak_auto_source_chunks"
            retrieved = self._retrieve_from_chunks(
                question,
                chunks=self.chunks,
                top_k=top_k,
                per_source_limit=effective_per_source_limit,
            )

        if source_selection and source_selection.fallback_used:
            fallback_used = True
            fallback_reason = fallback_reason or source_selection.fallback_reason

        return RagRetrievalResult(
            retrieved_chunks=retrieved,
            retrieval_mode=mode,
            source_selection=source_selection,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def _effective_per_source_limit(self, question: str, per_source_limit: int) -> int:
        classified = self.question_classifier.classify(question)
        if str(classified.primary_type.name) == "LIST":
            return min(per_source_limit, 2)
        return per_source_limit

    def _retrieve_from_chunks(
        self,
        question: str,
        *,
        chunks: list[dict[str, Any]],
        top_k: int,
        per_source_limit: int,
    ) -> list[RetrievedChunk]:
        parsed = parse_question(question)
        classified = self.question_classifier.classify(question)
        query_terms = build_query_terms(question)
        entity_terms = [entity.lower() for entity in parsed["entities"] + classified.major_entities]
        predicate_terms = list(dict.fromkeys(
            parsed["predicates"] + classified.major_predicates + infer_query_predicates(question)
        ))

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
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
        if has_filter and any(value in text for value in ("fewer", "less than", "beyond earth", "orbit beyond", "moons")):
            score += 2.5
            reasons.append("comparative_filter_context")
        if "moon_count" in predicate_terms:
            moon_score, moon_reasons = score_moon_count_context(chunk, text)
            score += moon_score
            reasons.extend(moon_reasons)
        if "distance_from_sun" in predicate_terms:
            distance_score, distance_reasons = score_orbit_order_context(chunk, text)
            score += distance_score
            reasons.extend(distance_reasons)
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


def is_weak_retrieval(items: list[RetrievedChunk]) -> bool:
    if len(items) < 3:
        return True
    if not items:
        return True
    return items[0].score < 8.0


def score_moon_count_context(chunk: dict[str, Any], text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    section = str(chunk.get("section", "")).lower()
    title = str(chunk.get("title", "")).lower()

    if "moon" in section or "satellite" in section:
        score += 5.0
        reasons.append("moon_section")
    if any(word in text for word in ("known moons", "confirmed moons", "natural satellites", "confirmed satellites")):
        score += 5.0
        reasons.append("moon_count_terms")
    if re_moon_count_claim(text):
        score += 8.0
        reasons.append("moon_count_claim")
    if title_is_planet(title) and any(word in text for word in ("moon", "moons", "satellite", "satellites")):
        score += 4.0
        reasons.append("planet_moon_context")
    return score, reasons


def score_orbit_order_context(chunk: dict[str, Any], text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    section = str(chunk.get("section", "")).lower()
    title = str(chunk.get("title", "")).lower()

    if section in {"inner planets", "outer planets", "orbits"}:
        score += 5.0
        reasons.append("planet_order_section")
    if title == "solar system" and any(value in text for value in ("inner planets", "outer planets", "au)", "from the sun")):
        score += 4.0
        reasons.append("solar_system_order_context")
    if title_is_planet(title) and any(value in text for value in ("from the sun", "au", "orbit", "fifth planet", "seventh planet", "eighth planet", "fourth planet")):
        score += 3.0
        reasons.append("planet_orbit_context")
    return score, reasons


def re_moon_count_claim(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:has|have|had|includes?|possesses?)\s+(?:at\s+least\s+)?(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred)\s+"
            r"(?:known\s+|confirmed\s+|natural\s+)?(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred)\s+"
            r"(?:\w+\s+){0,4}(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b(?:mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)'?s\s+"
            r"(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred)\s+"
            r"(?:\w+\s+){0,4}(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b(?:mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)\s+has\s+"
            r"(?:\d+|one|two|three|four|five|sixteen|twenty[- ]?nine|hundred).{0,90}?"
            r"\b(?:moon|moons|satellite|satellites)\b",
            text,
        )
    )


def title_is_planet(title: str) -> bool:
    normalized = title.replace(" (planet)", "")
    return normalized in {
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    }
