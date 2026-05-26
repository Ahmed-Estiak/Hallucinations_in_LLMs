"""Temporal fact compatibility rules for RAG retrieval."""

from __future__ import annotations

import re
from typing import Any

from src.time_utils import time_window


def is_temporal_count_intent(intent: Any) -> bool:
    return bool(
        intent.has_time_lookup
        and intent.time_value
        and "moon_count" in intent.predicate_terms
        and intent.entity_terms
    )


def is_current_count_intent(intent: Any) -> bool:
    """Return whether a query asks for one subject's present moon count."""
    return bool(
        not intent.has_time_lookup
        and "moon_count" in intent.predicate_terms
        and len(set(intent.entity_terms)) == 1
        and not intent.has_comparison
        and not intent.filter_conditions
        and not intent.ordering_attribute
    )


def temporal_fact_match_kind(fact: dict[str, Any], intent: Any) -> str:
    """Return how a validated temporal fact supports a query, or an empty string."""
    if not is_temporal_count_intent(intent):
        return ""
    if fact.get("validation_status") != "validated":
        return ""
    if fact.get("predicate") != "moon_count":
        return ""
    subject = str(fact.get("subject", "")).lower()
    if subject not in set(intent.entity_terms):
        return ""

    query_window = time_window(intent.time_value)
    observed_window = time_window(fact.get("observed_at"))
    if query_window is None or observed_window is None:
        return ""
    if intent.time_semantic not in {"exact", "as_of"}:
        return ""

    exact_overlap = (
        query_window[0] <= observed_window[1]
        and observed_window[0] <= query_window[1]
    )
    evidence_type = fact.get("evidence_type")
    if evidence_type == "explicit_dated_sentence":
        return "explicit_exact" if exact_overlap else ""
    if evidence_type != "derived_validated_timeline":
        return ""
    if exact_overlap:
        return "timeline_anchor"

    until_window = time_window(fact.get("valid_until_exclusive"))
    if until_window is None:
        return ""
    if observed_window[0] <= query_window[0] < until_window[0]:
        return "timeline_interval"
    return ""


def current_fact_match_kind(fact: dict[str, Any], intent: Any) -> str:
    if not is_current_count_intent(intent):
        return ""
    if fact.get("predicate") != "moon_count":
        return ""
    if str(fact.get("subject", "")).lower() not in set(intent.entity_terms):
        return ""
    if intent.required_claim and fact.get("claim_type") != intent.required_claim:
        return ""
    evidence_type = fact.get("evidence_type")
    if evidence_type == "explicit_current_sentence":
        return "direct_current_assertion"
    if evidence_type == "validated_table_current":
        return "validated_current_table"
    return ""


def temporal_match_score(match_kind: str) -> float:
    return {
        "explicit_exact": 36.0,
        "timeline_anchor": 30.0,
        "timeline_interval": 25.0,
    }.get(match_kind, 0.0)


def current_match_score(match_kind: str) -> float:
    return {
        "direct_current_assertion": 34.0,
        "validated_current_table": 29.0,
    }.get(match_kind, 0.0)


def compatible_temporal_chunks(
    chunks: list[dict[str, Any]],
    intent: Any,
) -> list[tuple[dict[str, Any], str]]:
    compatible = []
    for chunk in chunks:
        fact = chunk.get("temporal_fact")
        if not isinstance(fact, dict):
            continue
        match_kind = temporal_fact_match_kind(fact, intent)
        if match_kind:
            compatible.append((chunk, match_kind))
    compatible.sort(
        key=lambda item: (
            temporal_match_score(item[1]),
            item[0]["temporal_fact"].get("evidence_type") == "explicit_dated_sentence",
        ),
        reverse=True,
    )
    return compatible


def compatible_current_chunks(
    chunks: list[dict[str, Any]],
    intent: Any,
) -> list[tuple[dict[str, Any], str]]:
    """Resolve admissible current claims using a monotonic-count assumption."""
    if not is_current_count_intent(intent):
        return []
    subjects = set(intent.entity_terms)
    historical_values = [
        fact.get("value")
        for chunk in chunks
        if isinstance((fact := chunk.get("temporal_fact")), dict)
        and str(fact.get("subject", "")).lower() in subjects
        and fact.get("evidence_type") in {"explicit_dated_sentence", "derived_validated_timeline"}
        and isinstance(fact.get("value"), int)
    ]
    historical_floor = max(historical_values, default=-1)
    matches = [
        (chunk, match_kind)
        for chunk in chunks
        if isinstance((fact := chunk.get("temporal_fact")), dict)
        and (match_kind := current_fact_match_kind(fact, intent))
        and isinstance(fact.get("value"), int)
        and fact["value"] >= historical_floor
    ]
    if not matches:
        return []

    trusted = [
        item for item in matches
        if item[0].get("trust_level") in {"official", "reference"}
    ]
    eligible = trusted or matches
    max_value = max(item[0]["temporal_fact"]["value"] for item in eligible)
    selected = [
        item for item in eligible
        if item[0]["temporal_fact"]["value"] == max_value
    ]
    selected.sort(
        key=lambda item: (
            item[0].get("trust_level") == "official",
            current_match_score(item[1]),
        ),
        reverse=True,
    )
    return selected


def contains_target_count_assertion(chunk: dict[str, Any], intent: Any) -> bool:
    """Identify raw count claims that can conflict with selected temporal evidence."""
    if chunk.get("temporal_fact") or not is_temporal_count_intent(intent):
        return False
    text = str(chunk.get("text", "")).lower()
    if not any(
        re.search(rf"\b{re.escape(entity)}\b", text)
        for entity in intent.entity_terms
    ):
        return False
    return bool(
        re.search(
            r"\b(?:has|have|had|includes?|possesses?)\s+(?:at\s+least\s+)?"
            r"\d[\d,]*\s+(?:officially\s+|known\s+|confirmed\s+)?"
            r"(?:natural\s+)?(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b\d[\d,]*\s+(?:officially\s+recognized\s+|officially\s+|known\s+|"
            r"confirmed\s+)?(?:natural\s+)?(?:moon|moons|satellite|satellites)\b",
            text,
        )
    )


def contains_numeric_moon_count_assertion(chunk: dict[str, Any]) -> bool:
    """Detect raw numeric count statements once a resolved fact is available."""
    if chunk.get("temporal_fact"):
        return False
    text = str(chunk.get("text", "")).lower()
    return bool(
        re.search(
            r"\b(?:has|have|had|includes?|possesses?)\s+(?:at\s+least\s+)?"
            r"\d[\d,]*\s+(?:officially\s+|known\s+|confirmed\s+)?"
            r"(?:natural\s+)?(?:moon|moons|satellite|satellites)\b",
            text,
        )
        or re.search(
            r"\b\d[\d,]*\s+(?:officially\s+recognized\s+|officially\s+|known\s+|"
            r"confirmed\s+)?(?:natural\s+)?(?:moon|moons|satellite|satellites)\b",
            text,
        )
    )
