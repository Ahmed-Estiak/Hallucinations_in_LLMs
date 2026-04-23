"""
Shared time parsing and deterministic temporal selection helpers.
"""
from __future__ import annotations

import calendar
import re
from typing import Iterable, Optional

TIME_RANGE_SEPARATOR = ".."


def parse_time_parts(value: Optional[str]) -> tuple[Optional[int], Optional[int], Optional[int]] | None:
    """Parse YYYY / YYYY-MM / YYYY-MM-DD into comparable parts."""
    if value is None:
        return None

    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        return int(text), None, None

    if re.match(r"^\d{4}-\d{2}$", text):
        year, month = text.split("-")
        year_int = int(year)
        month_int = int(month)
        if not 1 <= month_int <= 12:
            return None
        return year_int, month_int, None

    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        year, month, day = text.split("-")
        year_int = int(year)
        month_int = int(month)
        day_int = int(day)
        if not 1 <= month_int <= 12:
            return None
        max_day = calendar.monthrange(year_int, month_int)[1]
        if not 1 <= day_int <= max_day:
            return None
        return year_int, month_int, day_int

    return None


def time_window(value: Optional[str]) -> tuple[int, int] | None:
    """
    Convert a time token into an inclusive numeric window.

    YYYY -> whole year
    YYYY-MM -> whole month
    YYYY-MM-DD -> exact day
    """
    parts = parse_time_parts(value)
    if parts is None:
        return None

    year, month, day = parts
    if month is None:
        return year * 10000 + 101, year * 10000 + 1231
    if day is None:
        last_day = calendar.monthrange(year, month)[1]
        return year * 10000 + month * 100 + 1, year * 10000 + month * 100 + last_day
    numeric = year * 10000 + month * 100 + day
    return numeric, numeric


def _window_span(window: tuple[int, int]) -> int:
    return window[1] - window[0]


def _select_fact_by_time(facts: Iterable[dict], *, mode: str) -> Optional[dict]:
    """
    Pick the earliest or latest dated fact.

    If no facts have a valid time window, fall back to the first undated fact in
    input order.
    """
    best_fact = None
    best_window = None
    fallback_fact = None
    for fact in facts:
        if fallback_fact is None:
            fallback_fact = fact
        window = time_window(fact.get("time"))
        if window is None:
            continue

        if mode == "latest":
            is_better = (
                best_window is None or
                window[1] > best_window[1] or
                (window[1] == best_window[1] and _window_span(window) < _window_span(best_window))
            )
        else:
            is_better = (
                best_window is None or
                window[0] < best_window[0] or
                (window[0] == best_window[0] and _window_span(window) < _window_span(best_window))
            )

        if is_better:
            best_fact = fact
            best_window = window

    return best_fact or fallback_fact


def latest_fact(facts: Iterable[dict]) -> Optional[dict]:
    """Pick the latest fact, preferring more precise timestamps on ties."""
    return _select_fact_by_time(facts, mode="latest")


def earliest_fact(facts: Iterable[dict]) -> Optional[dict]:
    """Pick the earliest fact, preferring more precise timestamps on ties."""
    return _select_fact_by_time(facts, mode="earliest")


def fact_matches_time(fact_time: Optional[str], constraint_time: Optional[str], semantic: Optional[str]) -> bool:
    """Deterministic temporal match rule used for filtering.

    Note:
    - EXACT here behaves as an as-of snapshot rule, not strict equality.
    - Undated facts do not match when a time constraint is present.
    """
    if not constraint_time:
        return True
    if not fact_time:
        return False

    fact_window = time_window(fact_time)
    if fact_window is None:
        return False

    semantic_name = (semantic or "EXACT").upper()
    if semantic_name == "BETWEEN" and TIME_RANGE_SEPARATOR in str(constraint_time):
        start_text, end_text = str(constraint_time).split(TIME_RANGE_SEPARATOR, 1)
        start_window = time_window(start_text)
        end_window = time_window(end_text)
        if start_window is None or end_window is None:
            return False
        return fact_window[0] >= start_window[0] and fact_window[1] <= end_window[1]

    constraint_window = time_window(constraint_time)
    if constraint_window is None:
        return False

    if semantic_name == "BEFORE":
        return fact_window[1] < constraint_window[0]
    if semantic_name == "AFTER":
        return fact_window[0] > constraint_window[1]

    # EXACT behaves like an as-of snapshot: valid at or before the constraint.
    return fact_window[0] <= constraint_window[1]


def select_best_temporal_fact(
    facts: Iterable[dict],
    constraint_time: Optional[str],
    semantic: Optional[str],
) -> Optional[dict]:
    """Choose the single best fact under the deterministic temporal policy."""
    facts_list = list(facts)
    if not facts_list:
        return None

    if not constraint_time:
        return latest_fact(facts_list)

    semantic_name = (semantic or "EXACT").upper()
    eligible = [fact for fact in facts_list if fact_matches_time(fact.get("time"), constraint_time, semantic_name)]
    if not eligible:
        return None

    if semantic_name == "AFTER":
        return earliest_fact(eligible)
    return latest_fact(eligible)
