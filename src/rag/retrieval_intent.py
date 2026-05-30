"""Query-derived intent and generic lexical evidence scoring for RAG retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from src.question_classifier import LogicalModifier, QuestionClassifier
from src.question_parser import parse_question
from src.rag.retriever_terms import build_query_terms, infer_query_predicates


ATTRIBUTE_NORMALIZATION = {
    "discovered": "discovered_on",
    "distance": "distance_from_sun",
    "moons": "moon_count",
}

TARGET_CLASS_ALIASES = {
    "dwarf_planets": ("dwarf planet", "dwarf planets", "minor planet", "minor planets"),
    "planets": ("planet", "planets"),
    "moons": ("moon", "moons", "satellite", "satellites"),
    "asteroids": ("asteroid", "asteroids"),
    "comets": ("comet", "comets"),
}

ORDERING_EVIDENCE = {
    "discovered_on": ("discovered", "discovery", "first observed"),
    "size": ("size", "diameter", "radius"),
    "mass": ("mass", "massive", "heavier", "lighter"),
    "moon_count": ("moon", "moons", "satellite", "satellites"),
    "ring_count": ("ring", "rings"),
    "distance_from_sun": ("orbit", "distance from the sun", "semi-major axis"),
}


@dataclass(frozen=True)
class RetrievalIntent:
    """Normalized retrieval features derived only from the submitted question."""

    query_terms: list[str]
    entity_terms: list[str]
    predicate_terms: list[str]
    target_class: str | None
    filter_conditions: list[dict[str, Any]]
    ordering_attribute: str | None
    has_comparison: bool
    has_time_lookup: bool
    time_value: str | None
    time_semantic: str


def build_retrieval_intent(
    question: str,
    *,
    classifier: QuestionClassifier | None = None,
) -> RetrievalIntent:
    parsed = parse_question(question)
    classified = (classifier or QuestionClassifier()).classify(question)
    predicates = list(dict.fromkeys(
        parsed["predicates"] + classified.major_predicates + infer_query_predicates(question)
    ))
    modifiers = set(classified.logical_modifiers)
    conditions = (
        [normalize_filter_condition(condition) for condition in classified.entity_filter_conditions]
        if LogicalModifier.FILTER in modifiers
        else []
    )
    for condition in conditions:
        attribute = condition.get("attribute")
        if attribute and attribute != "unknown" and attribute not in predicates:
            predicates.append(attribute)
    ordering_attribute = (
        normalize_attribute(classified.ordering_attribute)
        if LogicalModifier.ORDERING in modifiers
        else None
    )
    return RetrievalIntent(
        query_terms=build_query_terms(question, predicate_terms=predicates),
        entity_terms=[
            entity.lower()
            for entity in parsed["entities"] + classified.major_entities
        ],
        predicate_terms=predicates,
        target_class=classified.target_entity_class or classified.list_target,
        filter_conditions=conditions,
        ordering_attribute=ordering_attribute,
        has_comparison=LogicalModifier.COMPARISON in modifiers,
        has_time_lookup=LogicalModifier.TIME_LOOKUP in modifiers,
        time_value=classified.time_value,
        time_semantic=classified.time_semantic.value,
    )


def normalize_filter_condition(condition: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(condition)
    normalized["attribute"] = normalize_attribute(str(normalized.get("attribute", "")))
    return normalized


def normalize_attribute(attribute: str | None) -> str | None:
    if not attribute:
        return None
    return ATTRIBUTE_NORMALIZATION.get(attribute, attribute)


def score_filter_evidence(
    text: str,
    predicate_hints: set[str],
    conditions: list[dict[str, Any]],
    *,
    weight: float,
) -> tuple[float, list[str]]:
    """Reward evidence for the requested filter value, not just any filter topic."""
    score = 0.0
    reasons: list[str] = []
    for condition in conditions:
        attribute = normalize_attribute(str(condition.get("attribute", "")))
        operator = str(condition.get("operator", ""))
        value = str(condition.get("value", "")).strip().lower()
        reference = str(condition.get("reference_entity", "")).strip().lower()

        if attribute == "location" and operator == "==" and value and has_phrase(text, value):
            score += weight
            reasons.append(f"filter:location:{slug(value)}")
        elif attribute == "planet_type" and operator == "==" and value and has_phrase(text, value):
            score += weight
            reasons.append(f"filter:planet_type:{slug(value)}")
        elif attribute in ORDERING_EVIDENCE and operator in {"<", ">", "=="}:
            if attribute in predicate_hints or contains_any(text, ORDERING_EVIDENCE[attribute]):
                score += weight * 0.65
                reasons.append(f"constraint_evidence:{attribute}")
                if reference and has_phrase(text, reference):
                    score += weight * 0.35
                    reasons.append(f"constraint_reference:{attribute}:{slug(reference)}")
                else:
                    reasons.append(f"constraint_candidate:{attribute}")
        elif value and has_phrase(text, value):
            score += weight * 0.5
            reasons.append(f"filter:{slug(attribute or 'value')}:{slug(value)}")
    return score, reasons


def score_ordering_evidence(
    text: str,
    ordering_attribute: str | None,
    *,
    weight: float,
    detailed: bool = False,
) -> tuple[float, list[str]]:
    """Reward ordering evidence only for the attribute being ordered."""
    attribute = normalize_attribute(ordering_attribute)
    evidence_terms = ORDERING_EVIDENCE.get(attribute or "")
    if not evidence_terms or not contains_any(text, evidence_terms):
        return 0.0, []

    if attribute == "discovered_on":
        return score_discovery_ordering_evidence(text, weight=weight, detailed=detailed)

    score = weight
    reasons = [f"ordering:{attribute}"]
    return score, reasons


def score_discovery_ordering_evidence(
    text: str,
    *,
    weight: float,
    detailed: bool,
) -> tuple[float, list[str]]:
    """Score discovery ordering with stronger signals than plain word overlap."""

    score = 0.0
    reasons: list[str] = []
    if has_date_near_discovery(text):
        score += weight * 0.8
        reasons.append("discovery_date_nearby")

    if has_ordinal_discovery_evidence(text):
        score += weight * 2.4
        reasons.append("ordinal_discovery_evidence")
        if "in order of discovery" in text or "order of discovery" in text:
            reasons.append("ordered_discovery_section")

    if detailed:
        entities = discovered_entity_mentions(text)
        if len(entities) >= 2:
            score += weight * 0.7 + math.log(len(entities))
            reasons.append("multiple_discovery_entities")

    if score:
        return score, ["ordering:discovered_on", *reasons]
    return 0.0, []


def has_date_near_discovery(text: str) -> bool:
    date_pattern = r"(?:\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b(?:1[6-9]\d{2}|20\d{2})\b)"
    discovery_pattern = r"\b(?:discovered|discovery|first observed)\b"
    return bool(
        re.search(rf"{date_pattern}.{{0,80}}{discovery_pattern}", text)
        or re.search(rf"{discovery_pattern}.{{0,80}}{date_pattern}", text)
    )


def has_ordinal_discovery_evidence(text: str) -> bool:
    ordinal_pattern = (
        r"\b(?:first|earliest|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"\d+(?:st|nd|rd|th)|in\s+order|order\s+of)\b"
    )
    discovery_pattern = r"\b(?:discovered|discovery|first observed)\b"
    return bool(
        re.search(rf"{ordinal_pattern}.{{0,120}}{discovery_pattern}", text)
        or re.search(rf"{discovery_pattern}.{{0,120}}{ordinal_pattern}", text)
    )


DISCOVERY_ENTITY_STOPWORDS = {
    "astronomers",
    "belt",
    "body",
    "bodies",
    "candidate",
    "candidates",
    "existence",
    "object",
    "objects",
    "planet",
    "planets",
    "world",
}


def discovered_entity_mentions(text: str) -> set[str]:
    mentions: set[str] = set()
    for match in re.finditer(
        r"(?=\b([a-z][a-z0-9-]{2,})\b(?:\s+\w+){0,5}\s+was\b.{0,80}\bdiscovered\b)",
        text,
    ):
        entity = match.group(1)
        if entity not in DISCOVERY_ENTITY_STOPWORDS:
            mentions.add(entity)
    for match in re.finditer(r"\bdiscovery\s+of\s+([a-z][a-z0-9-]{2,})\b", text):
        entity = match.group(1)
        if entity not in DISCOVERY_ENTITY_STOPWORDS:
            mentions.add(entity)
    return mentions


def score_target_class_evidence(
    text: str,
    target_class: str | None,
    *,
    weight: float,
    reason_prefix: str = "target_class",
) -> tuple[float, list[str]]:
    """Give symmetric class evidence credit for any configured target class."""
    if not target_class:
        return 0.0, []
    aliases = TARGET_CLASS_ALIASES.get(target_class, (target_class.replace("_", " "),))
    if contains_any(text, aliases):
        return weight, [f"{reason_prefix}:{target_class}"]
    return 0.0, []


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(has_phrase(text, phrase) for phrase in phrases)


def has_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
