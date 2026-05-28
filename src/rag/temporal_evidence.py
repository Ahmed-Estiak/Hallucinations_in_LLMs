"""Temporal fact compatibility rules for RAG retrieval."""

from __future__ import annotations

import copy
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
    """Return how one temporal fact supports a query, or an empty string.

    This helper is still useful for local scoring. Final as-of count resolution is
    handled by compatible_temporal_chunks(), which merges all extracted facts for
    the same subject/predicate before deciding interval coverage.
    """
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
    """Return temporal evidence after cross-source timeline reconciliation.

    All validated dated count facts for the requested subject/predicate are first
    merged into one timeline. An intermediate as-of date is supported only by the
    latest known anchor before the query and only if another later known anchor
    exists. This prevents a single source's derived interval from hiding a newer
    intermediate date found in another source.
    """
    if not is_temporal_count_intent(intent):
        return []
    query_window = time_window(intent.time_value)
    if query_window is None or intent.time_semantic not in {"exact", "as_of"}:
        return []

    timeline = _temporal_timeline(chunks, intent)
    exact = []
    for chunk in timeline:
        fact = chunk["temporal_fact"]
        observed_window = time_window(fact.get("observed_at"))
        if not _windows_overlap(query_window, observed_window):
            continue
        next_observed_at = _next_observed_at(timeline, observed_window[0])
        exact.append((
            _resolved_temporal_chunk(chunk, match_kind=_exact_match_kind(fact), next_observed_at=next_observed_at),
            _exact_match_kind(fact),
        ))
    if exact:
        return _sort_temporal_matches(exact)

    anchors_before = [
        chunk for chunk in timeline
        if (observed_window := time_window(chunk["temporal_fact"].get("observed_at")))
        and observed_window[0] <= query_window[0]
    ]
    anchors_after = [
        chunk for chunk in timeline
        if (observed_window := time_window(chunk["temporal_fact"].get("observed_at")))
        and observed_window[0] > query_window[0]
    ]
    if not anchors_before or not anchors_after:
        return []

    latest_start = max(
        time_window(chunk["temporal_fact"].get("observed_at"))[0]
        for chunk in anchors_before
    )
    next_start = min(
        time_window(chunk["temporal_fact"].get("observed_at"))[0]
        for chunk in anchors_after
    )
    latest_chunks = [
        chunk for chunk in anchors_before
        if time_window(chunk["temporal_fact"].get("observed_at"))[0] == latest_start
    ]
    next_chunk = min(
        anchors_after,
        key=lambda chunk: time_window(chunk["temporal_fact"].get("observed_at"))[0],
    )
    next_observed_at = next_chunk["temporal_fact"].get("observed_at", "")
    interval_matches = [
        (_resolved_temporal_chunk(chunk, match_kind="timeline_interval", next_observed_at=next_observed_at), "timeline_interval")
        for chunk in latest_chunks
    ]
    return _sort_temporal_matches(interval_matches)


def _temporal_timeline(chunks: list[dict[str, Any]], intent: Any) -> list[dict[str, Any]]:
    subjects = set(intent.entity_terms)
    timeline = []
    for chunk in chunks:
        fact = chunk.get("temporal_fact")
        if not isinstance(fact, dict):
            continue
        if fact.get("validation_status") != "validated":
            continue
        if fact.get("predicate") != "moon_count":
            continue
        if str(fact.get("subject", "")).lower() not in subjects:
            continue
        if fact.get("evidence_type") not in {"explicit_dated_sentence", "derived_validated_timeline"}:
            continue
        if time_window(fact.get("observed_at")) is None:
            continue
        timeline.append(chunk)
    timeline.sort(key=_temporal_sort_key)
    return timeline


def _temporal_sort_key(chunk: dict[str, Any]) -> tuple[Any, ...]:
    fact = chunk["temporal_fact"]
    observed_window = time_window(fact.get("observed_at"))
    return (
        observed_window[0],
        fact.get("evidence_type") != "explicit_dated_sentence",
        chunk.get("trust_level") != "official",
        chunk.get("chunk_id", ""),
    )


def _windows_overlap(left: Any, right: Any) -> bool:
    return bool(left and right and left[0] <= right[1] and right[0] <= left[1])


def _exact_match_kind(fact: dict[str, Any]) -> str:
    if fact.get("evidence_type") == "explicit_dated_sentence":
        return "explicit_exact"
    return "timeline_anchor"


def _sort_temporal_matches(
    matches: list[tuple[dict[str, Any], str]],
) -> list[tuple[dict[str, Any], str]]:
    matches.sort(
        key=lambda item: (
            temporal_match_score(item[1]),
            item[0]["temporal_fact"].get("evidence_type") == "explicit_dated_sentence",
            item[0].get("trust_level") == "official",
        ),
        reverse=True,
    )
    return matches


def _next_observed_at(timeline: list[dict[str, Any]], observed_start: Any) -> str:
    later_chunks = [
        chunk for chunk in timeline
        if (window := time_window(chunk["temporal_fact"].get("observed_at")))
        and window[0] > observed_start
    ]
    if not later_chunks:
        return ""
    next_chunk = min(
        later_chunks,
        key=lambda chunk: time_window(chunk["temporal_fact"].get("observed_at"))[0],
    )
    return next_chunk["temporal_fact"].get("observed_at", "")


def _resolved_temporal_chunk(chunk: dict[str, Any], *, match_kind: str, next_observed_at: str) -> dict[str, Any]:
    resolved = copy.deepcopy(chunk)
    fact = resolved["temporal_fact"]
    fact["valid_until_exclusive"] = next_observed_at
    if next_observed_at:
        fact["interval_semantics"] = "cross_source_carry_forward_until_next_known_change"
    subject = fact.get("subject", "")
    value = fact.get("value", "")
    observed_at = fact.get("observed_at", "")
    text = (
        f"Resolved temporal fact: As of {_display_date(observed_at)}, {subject} had "
        f"{value} moons."
    )
    if next_observed_at:
        text += (
            " Across all extracted temporal evidence for this subject and count type, "
            f"this value is used until the next known count change in {_display_date(next_observed_at)}."
        )
    elif match_kind == "timeline_interval":
        text += " No later known count change is available, so this interval is not open-ended."
    resolved["text"] = text
    resolved["chunk_id"] = f"{resolved.get('chunk_id', 'temporal_fact')}__resolved_{match_kind}"
    return resolved


def _display_date(value: str) -> str:
    if not value:
        return ""
    month_names = {
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }
    parts = value.split("-")
    if len(parts) == 1:
        return value
    year, month = parts[:2]
    if len(parts) == 3:
        return f"{month_names.get(month, month)} {int(parts[2])}, {year}"
    return f"{month_names.get(month, month)} {year}"


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
