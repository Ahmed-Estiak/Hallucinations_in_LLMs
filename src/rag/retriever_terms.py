"""Shared query term helpers for source and chunk retrieval."""

from __future__ import annotations

import re


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
    "than",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
    "yet",
}

QUERY_EXPANSIONS = {
    "beyond": ["beyond", "farther", "outer", "orbit beyond"],
    "confirmed": ["confirmed", "officially recognized", "known"],
    "discovered": ["discovered", "discovery", "found", "first observed"],
    "dwarf": ["dwarf", "dwarf planet", "dwarf planets", "minor planet", "minor planets"],
    "fewer": ["fewer", "less than"],
    "jupiter": ["jupiter", "jovian"],
    "kuiper": ["kuiper", "kuiper belt", "trans-neptunian", "beyond neptune"],
    "located": ["located", "location", "region", "belt"],
    "mars": ["mars"],
    "mass": ["mass", "massive", "heavier", "lighter"],
    "moon": ["moon", "moons", "satellite", "satellites"],
    "moons": ["moon", "moons", "satellite", "satellites"],
    "orbit": ["orbit", "orbital", "distance", "semi-major axis"],
    "orbits": ["orbit", "orbital", "distance", "semi-major axis"],
    "planet": ["planet", "planets"],
    "planets": ["planet", "planets"],
    "saturn": ["saturn", "saturnian"],
    "uranus": ["uranus", "uranian"],
    "neptune": ["neptune", "neptunian"],
}

ATTRIBUTE_QUERY_EXPANSIONS = {
    "distance_from_sun": ["orbit", "orbital", "distance", "distance from the sun", "semi-major axis"],
    "moon_count": ["moon", "moons", "satellite", "satellites"],
    "ring_count": ["ring", "rings"],
    "mass": ["mass", "massive"],
    "size": ["size", "diameter", "radius"],
}


def build_query_terms(question: str, *, predicate_terms: list[str] | None = None) -> list[str]:
    terms = tokenize(question)
    expanded = []
    for term in terms:
        if term in STOPWORDS:
            continue
        expanded.append(term)
        expanded.extend(QUERY_EXPANSIONS.get(term, []))
    for predicate in predicate_terms or []:
        expanded.extend(ATTRIBUTE_QUERY_EXPANSIONS.get(predicate, []))
    return list(dict.fromkeys(expanded))


def infer_query_predicates(question: str) -> list[str]:
    question_lower = question.lower()
    predicates = []
    if re.search(r"\b(?:moon|moons|satellite|satellites|fewer\s+moons|more\s+moons)\b", question_lower):
        predicates.append("moon_count")
    if re.search(r"\b(?:ring|rings|fewer\s+rings|more\s+rings)\b", question_lower):
        predicates.append("ring_count")
    if re.search(r"\b(?:orbit|orbits|beyond|farther|closer|distance|from the sun)\b", question_lower):
        predicates.append("distance_from_sun")
    if re.search(r"\b(?:dwarf planet|classified|classification|recognized as)\b", question_lower):
        predicates.append("classification")
    if re.search(r"\b(?:located|location|kuiper belt|asteroid belt|found in)\b", question_lower):
        predicates.append("location")
    if re.search(r"\b(?:discovered|discovery|found|first observed|first)\b", question_lower):
        predicates.append("discovered_on")
    if re.search(r"\b(?:mass|massive|heavier|lighter)\b", question_lower):
        predicates.append("mass")
    return list(dict.fromkeys(predicates))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
