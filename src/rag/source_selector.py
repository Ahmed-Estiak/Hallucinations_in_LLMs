"""Automatic source selection for RAG retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.question_classifier import QuestionClassifier
from src.rag.retrieval_intent import (
    RetrievalIntent,
    TARGET_CLASS_ALIASES,
    build_retrieval_intent,
    score_filter_evidence,
    score_ordering_evidence,
    score_target_class_evidence,
)
from src.rag.retriever_terms import tokenize
from src.rag.temporal_evidence import (
    is_temporal_count_intent,
    temporal_fact_match_kind,
    temporal_match_score,
)


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
    coverage_source_ids: list[str] = field(default_factory=list)
    coverage_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_source_ids": self.selected_source_ids,
            "scores": [score.to_dict() for score in self.scores],
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "coverage_source_ids": self.coverage_source_ids,
            "coverage_reasons": self.coverage_reasons,
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
    temporal_facts: list[dict[str, Any]] = field(default_factory=list)
    source_type: str = "web"


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
        intent = build_retrieval_intent(question, classifier=self.question_classifier)

        scores = []
        for profile in self.profiles.values():
            score, reasons = score_source(
                profile,
                intent=intent,
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
        fallback_used = False
        fallback_reason = ""
        if len(selected) < 2 and scores:
            selected = [score.source_id for score in scores[:top_n_sources]]
            fallback_used = True
            fallback_reason = "source_score_threshold_too_strict"
        pdf_only = profiles_are_pdf_only(self.profiles)
        if pdf_only:
            coverage_source_ids = []
            coverage_reasons = []
        else:
            selected, coverage_source_ids, coverage_reasons = apply_constraint_coverage(
                selected,
                self.profiles,
                intent,
            )
        selected, pdf_reasons = apply_pdf_raw_frequency_safety_include(
            selected,
            self.profiles,
            scores,
            intent,
            top_n_sources=top_n_sources,
        )
        coverage_reasons.extend(pdf_reasons)

        return SourceSelection(
            mode="auto-source",
            selected_source_ids=selected,
            scores=scores,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            coverage_source_ids=coverage_source_ids,
            coverage_reasons=coverage_reasons,
        )


def score_source(
    profile: SourceProfile,
    *,
    intent: RetrievalIntent,
) -> tuple[float, list[str]]:
    title_url = f"{profile.title} {profile.url} {slug_from_url(profile.url)}".lower()
    text = profile.text
    reasons: list[str] = []
    score = 0.0

    for term in intent.query_terms:
        if " " in term:
            if has_phrase(title_url, term):
                score += 6.0
                reasons.append(f"title_url_phrase:{term}")
            phrase_count = count_phrase_occurrences(text, term)
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

    for entity in set(intent.entity_terms):
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

    for predicate in intent.predicate_terms:
        if predicate in profile.predicates:
            score += 5.0
            reasons.append(f"predicate:{predicate}")

    target_class_score, target_class_reasons = score_target_class_context(profile, title_url=title_url, intent=intent)
    score += target_class_score
    reasons.extend(target_class_reasons)

    filter_score, filter_reasons = score_filter_evidence(
        text,
        profile.predicates,
        intent.filter_conditions,
        weight=3.0,
    )
    score += filter_score
    reasons.extend(filter_reasons)
    ordering_score, ordering_reasons = score_ordering_evidence(
        text,
        intent.ordering_attribute,
        weight=3.0,
    )
    score += ordering_score
    reasons.extend(ordering_reasons)
    if intent.has_comparison and any(value in text for value in ("greater", "less", "more", "fewer", "mass", "distance")):
        score += 3.0
        reasons.append("comparison_support")
    if intent.has_time_lookup and re.search(
        r"\b(?:as of|by|before|after|in)\s+(?:[a-z]+\s+)?\d{4}\b",
        text,
    ) and not is_temporal_count_intent(intent):
        score += 3.0
        reasons.append("time_support")
    if is_temporal_count_intent(intent):
        matching_facts = [
            (fact, temporal_fact_match_kind(fact, intent))
            for fact in profile.temporal_facts
        ]
        matching_facts = [(fact, kind) for fact, kind in matching_facts if kind]
        if matching_facts:
            _fact, kind = max(matching_facts, key=lambda item: temporal_match_score(item[1]))
            score += temporal_match_score(kind)
            reasons.append(f"temporal_fact:{kind}")

    trust_boost = TRUST_BOOSTS.get(profile.trust_level, 0.5)
    score += trust_boost
    reasons.append(f"trust:{profile.trust_level}")

    if profile.char_count < 500:
        score -= 1.0
        reasons.append("short_source_penalty")
    elif profile.char_count > 5000:
        score += 0.5
        reasons.append("substantial_source")

    return max(score, 0.0), reasons


def score_target_class_context(
    profile: SourceProfile,
    *,
    title_url: str,
    intent: RetrievalIntent,
) -> tuple[float, list[str]]:
    title_score, title_reasons = score_target_class_evidence(
        title_url,
        intent.target_class,
        weight=3.0,
        reason_prefix="target_class_title",
    )
    if title_score:
        return title_score, title_reasons
    return score_target_class_evidence(
        profile.text,
        intent.target_class,
        weight=1.5,
        reason_prefix="target_class_text",
    )


def has_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def count_phrase_occurrences(text: str, phrase: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


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
        temporal_facts = []
        for chunk in source_chunks:
            entities.update(chunk.get("entities", []))
            predicates.update(chunk.get("predicate_hints", []))
            temporal_fact = chunk.get("temporal_fact")
            if isinstance(temporal_fact, dict):
                temporal_facts.append(temporal_fact)
        first_chunk = source_chunks[0]
        profiles[source_id] = SourceProfile(
            source_id=source_id,
            title=document.get("title") or first_chunk.get("title", source_id),
            url=document.get("url") or first_chunk.get("url", ""),
            source_type=document.get("source_type") or first_chunk.get("source_type", "web"),
            cleaner=document.get("cleaner", ""),
            trust_level=document.get("trust_level", infer_trust_level(document.get("url") or first_chunk.get("url", ""))),
            char_count=int(document.get("char_count") or sum(len(chunk.get("text", "")) for chunk in source_chunks)),
            chunk_count=len(source_chunks),
            text=text,
            tokens=Counter(tokenize(text)),
            entities=entities,
            predicates=predicates,
            temporal_facts=temporal_facts,
        )
    return profiles


def apply_pdf_raw_frequency_safety_include(
    selected: list[str],
    profiles: dict[str, SourceProfile],
    scores: list[SourceScore],
    intent: RetrievalIntent,
    *,
    top_n_sources: int,
) -> tuple[list[str], list[str]]:
    """Force-include one PDF with the strongest raw meaningful term count.

    PDF titles and headings can be weak or absent. For PDF-only corpora, this
    keeps one broad but repeatedly relevant document from being missed by pure
    normalized-density ranking. It is only active when every source is PDF.
    """

    if not profiles_are_pdf_only(profiles):
        return selected, []

    score_by_source = {score.source_id: score for score in scores}
    raw_counts = [
        (source_id, raw_meaningful_match_count(profile, intent))
        for source_id, profile in profiles.items()
    ]
    raw_counts = [(source_id, count) for source_id, count in raw_counts if count >= 3]
    if not raw_counts:
        return selected, []

    raw_counts.sort(
        key=lambda item: (
            item[1],
            score_by_source.get(item[0], SourceScore(item[0], 0.0, [])).score,
        ),
        reverse=True,
    )
    raw_source_id, raw_count = raw_counts[0]
    if raw_source_id in selected:
        return selected, []

    expanded = selected[:]
    if len(expanded) >= top_n_sources + 1:
        return expanded, []
    expanded.append(raw_source_id)
    return expanded, [f"pdf_raw_frequency_include:{raw_source_id}:{raw_count}"]


GENERIC_LOW_VALUE_TERMS = {
    "planet",
    "planets",
    "moon",
    "moons",
    "satellite",
    "satellites",
    "object",
    "objects",
    "body",
    "bodies",
}


def raw_meaningful_match_count(profile: SourceProfile, intent: RetrievalIntent) -> int:
    text = profile.text
    count = 0
    for term in set(intent.query_terms):
        if term in GENERIC_LOW_VALUE_TERMS:
            continue
        if " " in term:
            count += count_phrase_occurrences(text, term) * 3
        else:
            count += profile.tokens.get(term, 0)
    for entity in set(intent.entity_terms):
        if entity and entity not in GENERIC_LOW_VALUE_TERMS:
            count += profile.tokens.get(entity, 0) * 2
    for predicate in set(intent.predicate_terms):
        if predicate in profile.predicates:
            count += 2
    for condition in intent.filter_conditions:
        value = str(condition.get("value", "")).strip().lower()
        reference = str(condition.get("reference_entity", "")).strip().lower()
        if value:
            count += count_phrase_occurrences(text, value) * 3
        if reference:
            count += profile.tokens.get(reference, 0) * 2
    return count


def profiles_are_pdf_only(profiles: dict[str, SourceProfile]) -> bool:
    return bool(profiles) and all(profile.source_type == "pdf" for profile in profiles.values())


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


def apply_constraint_coverage(
    selected: list[str],
    profiles: dict[str, SourceProfile],
    intent: RetrievalIntent,
) -> tuple[list[str], list[str], list[str]]:
    """Add relational baseline and candidate evidence sources without answer-specific lists."""
    expanded = list(selected)
    coverage_sources: list[str] = []
    reasons: list[str] = []
    relational_conditions = [
        condition
        for condition in intent.filter_conditions
        if condition.get("operator") in {"<", ">"} and condition.get("attribute") not in {None, "", "unknown"}
    ]
    if not intent.target_class or not relational_conditions:
        return expanded, coverage_sources, reasons

    required_attributes = {
        str(condition["attribute"])
        for condition in relational_conditions
    }

    def add_source(source_id: str, reason: str) -> None:
        if source_id not in coverage_sources:
            coverage_sources.append(source_id)
        if reason not in reasons:
            reasons.append(reason)
        if source_id not in expanded:
            expanded.append(source_id)

    for condition in relational_conditions:
        reference = str(condition.get("reference_entity", "")).strip()
        if not reference:
            continue
        source_id = find_named_source(reference, profiles)
        if source_id:
            add_source(
                source_id,
                f"reference_coverage:{condition['attribute']}:{reference.lower()}",
            )

    for source_id, profile in profiles.items():
        entity = source_target_entity(profile, intent.target_class)
        if not entity:
            continue
        attributes = sorted(required_attributes & profile.predicates)
        if not attributes:
            continue
        add_source(
            source_id,
            f"candidate_coverage:{entity.lower()}:{','.join(attributes)}",
        )
    return expanded, coverage_sources, reasons


def find_named_source(entity: str, profiles: dict[str, SourceProfile]) -> str:
    normalized_entity = normalize_title(entity)
    for source_id, profile in profiles.items():
        if normalize_title(profile.title) == normalized_entity:
            return source_id
    return ""


def source_target_entity(profile: SourceProfile, target_class: str) -> str:
    normalized_title = normalize_title(profile.title)
    matched_entity = next(
        (
            entity
            for entity in profile.entities
            if normalize_title(entity) == normalized_title
        ),
        "",
    )
    if not matched_entity:
        return ""

    aliases = TARGET_CLASS_ALIASES.get(target_class, (target_class.replace("_", " "),))
    singular_aliases = {alias.rstrip("s") for alias in aliases}
    identity_text = profile.text[:2000]
    for alias in singular_aliases:
        if re.search(
            rf"(?<!\w){re.escape(normalized_title)}(?!\w).{{0,80}}\b(?:is|was)\b.{{0,60}}(?<!\w){re.escape(alias)}(?!\w)",
            identity_text,
        ):
            return matched_entity
        if has_phrase(normalized_title, alias):
            return matched_entity
    return ""


def normalize_title(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", title.lower()).strip()
