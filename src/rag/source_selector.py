"""Automatic source selection for RAG retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.question_classifier import LogicalModifier, QuestionClassifier
from src.question_parser import parse_question
from src.rag.retriever_terms import build_query_terms, infer_query_predicates, tokenize


TRUST_BOOSTS = {
    "official": 3.0,
    "reference": 2.0,
    "article": 1.0,
}


@dataclass
class SourceScore:
    source_id: str
    score: float
    reasons: list[str]
    title: str = ""
    url: str = ""
    chunk_count: int = 0
    char_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "title": self.title,
            "url": self.url,
            "chunk_count": self.chunk_count,
            "char_count": self.char_count,
        }


@dataclass
class SourceSelection:
    mode: str
    selected_source_ids: list[str]
    scores: list[SourceScore]
    fallback_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_source_ids": self.selected_source_ids,
            "scores": [score.to_dict() for score in self.scores],
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class SourceProfile:
    source_id: str
    title: str
    url: str
    cleaner: str
    trust_level: str
    char_count: int
    chunk_count: int
    text: str
    tokens: Counter[str]
    entities: set[str]
    predicates: set[str]


class SourceSelector:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        documents_path: str | Path = "data/rag_sources/rag_index/documents.jsonl",
    ) -> None:
        self.chunks = chunks
        self.documents = load_documents(documents_path)
        self.profiles = build_source_profiles(chunks, self.documents)
        self.question_classifier = QuestionClassifier()

    def select(
        self,
        question: str,
        *,
        top_n_sources: int = 5,
        min_score: float = 8.0,
    ) -> SourceSelection:
        planet_list_question = bool(re.search(r"\bwhich\s+planets\b|\blist\s+(?:the\s+)?planets\b", question.lower()))
        if planet_list_question:
            top_n_sources = max(top_n_sources, 12)
        parsed = parse_question(question)
        classified = self.question_classifier.classify(question)
        query_terms = build_query_terms(question)
        entity_terms = [entity.lower() for entity in parsed["entities"] + classified.major_entities]
        predicate_terms = list(dict.fromkeys(
            parsed["predicates"] + classified.major_predicates + infer_query_predicates(question)
        ))
        modifiers = set(classified.logical_modifiers)

        scores = []
        for profile in self.profiles.values():
            score, reasons = score_source(
                profile,
                question=question,
                query_terms=query_terms,
                entity_terms=entity_terms,
                predicate_terms=predicate_terms,
                modifiers=modifiers,
            )
            if score > 0:
                scores.append(SourceScore(
                    source_id=profile.source_id,
                    score=score,
                    reasons=reasons,
                    title=profile.title,
                    url=profile.url,
                    chunk_count=profile.chunk_count,
                    char_count=profile.char_count,
                ))

        scores.sort(key=lambda item: item.score, reverse=True)
        selected = [score.source_id for score in scores if score.score >= min_score][:top_n_sources]
        if planet_list_question:
            selected = append_major_planet_sources(selected, self.profiles)
        fallback_used = False
        fallback_reason = ""
        if len(selected) < 2 and scores:
            selected = [score.source_id for score in scores[:top_n_sources]]
            fallback_used = True
            fallback_reason = "source_score_threshold_too_strict"

        return SourceSelection(
            mode="auto-source",
            selected_source_ids=selected,
            scores=scores,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )


def score_source(
    profile: SourceProfile,
    *,
    question: str,
    query_terms: list[str],
    entity_terms: list[str],
    predicate_terms: list[str],
    modifiers: set[LogicalModifier],
) -> tuple[float, list[str]]:
    title_url = f"{profile.title} {profile.url} {slug_from_url(profile.url)}".lower()
    text = profile.text
    reasons: list[str] = []
    score = 0.0

    for term in query_terms:
        if " " in term:
            if term in title_url:
                score += 6.0
                reasons.append(f"title_url_phrase:{term}")
            phrase_count = text.count(term)
            if phrase_count:
                phrase_score = min(8.0, 2.0 + math.log(phrase_count + 1) * 2.0)
                score += phrase_score
                reasons.append(f"density_phrase:{term}:{phrase_count}")
        else:
            if term in title_url:
                score += 2.5
                reasons.append(f"title_url_token:{term}")
            count = profile.tokens.get(term, 0)
            if count:
                density = count / max(1.0, sum(profile.tokens.values()) / 1000.0)
                density_score = min(6.0, density * 0.8)
                score += density_score
                if density_score >= 1.0:
                    reasons.append(f"density_token:{term}:{density:.2f}")

    for entity in set(entity_terms):
        if not entity:
            continue
        if entity in title_url:
            score += 8.0
            reasons.append(f"title_url_entity:{entity}")
        elif any(entity == known.lower() for known in profile.entities):
            score += 6.0
            reasons.append(f"entity:{entity}")
        elif entity in text:
            score += 3.0
            reasons.append(f"text_entity:{entity}")

    for predicate in predicate_terms:
        if predicate in profile.predicates:
            score += 5.0
            reasons.append(f"predicate:{predicate}")

    if LogicalModifier.FILTER in modifiers and any(value in text for value in ("fewer than", "less than", "beyond", "located", "kuiper belt")):
        score += 3.0
        reasons.append("filter_support")
    if LogicalModifier.ORDERING in modifiers and any(value in text for value in ("discovered", "discovery", "in order", "first")):
        score += 3.0
        reasons.append("ordering_support")
    if LogicalModifier.COMPARISON in modifiers and any(value in text for value in ("greater", "less", "more", "fewer", "mass", "distance")):
        score += 3.0
        reasons.append("comparison_support")
    if LogicalModifier.TIME_LOOKUP in modifiers and re.search(r"\b(?:as of|by|before|after|in)\s+\d{4}\b", text):
        score += 3.0
        reasons.append("time_support")

    trust_boost = TRUST_BOOSTS.get(profile.trust_level, 0.5)
    score += trust_boost
    reasons.append(f"trust:{profile.trust_level}")

    if profile.char_count < 1000:
        score -= 3.0
        reasons.append("short_source_penalty")
    elif profile.char_count > 5000:
        score += 1.0
        reasons.append("substantial_source")

    return max(score, 0.0), reasons


def load_documents(path: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return {}
    documents = {}
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            document = json.loads(line)
            documents[document["source_id"]] = document
    return documents


def build_source_profiles(
    chunks: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> dict[str, SourceProfile]:
    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk["source_id"]].append(chunk)

    profiles = {}
    for source_id, source_chunks in chunks_by_source.items():
        document = documents.get(source_id, {})
        text = " ".join(chunk.get("text", "") for chunk in source_chunks).lower()
        entities = set()
        predicates = set()
        for chunk in source_chunks:
            entities.update(chunk.get("entities", []))
            predicates.update(chunk.get("predicate_hints", []))
        first_chunk = source_chunks[0]
        profiles[source_id] = SourceProfile(
            source_id=source_id,
            title=document.get("title") or first_chunk.get("title", source_id),
            url=document.get("url") or first_chunk.get("url", ""),
            cleaner=document.get("cleaner", ""),
            trust_level=document.get("trust_level", infer_trust_level(document.get("url") or first_chunk.get("url", ""))),
            char_count=int(document.get("char_count") or sum(len(chunk.get("text", "")) for chunk in source_chunks)),
            chunk_count=len(source_chunks),
            text=text,
            tokens=Counter(tokenize(text)),
            entities=entities,
            predicates=predicates,
        )
    return profiles


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    return unquote(parsed.path.replace("/", " ").replace("_", " "))


def infer_trust_level(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "nasa.gov" in domain:
        return "official"
    if "wikipedia.org" in domain or "britannica.com" in domain:
        return "reference"
    return "article"


def append_major_planet_sources(selected: list[str], profiles: dict[str, SourceProfile]) -> list[str]:
    expanded = list(selected)
    for source_id, profile in profiles.items():
        if normalize_planet_title(profile.title) and source_id not in expanded:
            expanded.append(source_id)
    return expanded


def normalize_planet_title(title: str) -> str:
    normalized = title.lower().replace(" (planet)", "").strip()
    if normalized in {"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"}:
        return normalized
    return ""
