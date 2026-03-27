import re
from typing import Any, Dict, Optional


def normalize_text(text: Any) -> str:
    """
    Basic normalization for single-entity answers:
    - convert to string
    - remove leading/trailing spaces
    - lowercase everything
    """
    if text is None:
        return ""

    return str(text).strip().lower()


LEADING_NOISE_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+answer|answer)\s*[:\-]\s*|^(?:the\s+answer\s+is|it\s+is|it's)\s+"
)
MULTI_ENTITY_SEPARATOR_RE = re.compile(r",|;|\band\b|\bor\b|\s+/\s+")
AMBIGUITY_MARKER_RE = re.compile(r"\b(?:not|never|maybe|perhaps|probably|possibly|or)\b|,|;|\s+/\s+")


def _canonicalize_entity_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s'\"(\[]+", "", text)
    text = re.sub(r"[\s'\")\].,!?;:]+$", "", text)
    return text


def _tokenize_person_name(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text)


def _is_person_like_truth(text: str) -> bool:
    tokens = _tokenize_person_name(text)
    return len(tokens) >= 2 and " " in text


def _can_match_given_tokens(candidate_tokens: list[str], truth_tokens: list[str]) -> bool:
    unused_truth_tokens = truth_tokens[:]

    for candidate_token in candidate_tokens:
        matched_index = None

        for index, truth_token in enumerate(unused_truth_tokens):
            if candidate_token == truth_token:
                matched_index = index
                break

            if len(candidate_token) == 1 and truth_token.startswith(candidate_token):
                matched_index = index
                break

        if matched_index is None:
            return False

        unused_truth_tokens.pop(matched_index)

    return True


def _match_person_name_variant(candidate_text: str, truth_text: str) -> Dict[str, Any]:
    candidate_tokens = _tokenize_person_name(candidate_text)
    truth_tokens = _tokenize_person_name(truth_text)

    if len(candidate_tokens) < 2 or len(truth_tokens) < 2:
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    truth_surname = truth_tokens[-1]
    truth_given_tokens = truth_tokens[:-1]

    if truth_surname not in candidate_tokens:
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    candidate_given_tokens = [token for token in candidate_tokens if token != truth_surname]
    if not candidate_given_tokens:
        return {
            "matched": False,
            "manual_check": True,
            "reason": "ambiguous_partial_person_name",
        }

    if not _can_match_given_tokens(candidate_given_tokens, truth_given_tokens):
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    return {
        "matched": True,
        "manual_check": True,
        "reason": "matched_person_name_variant",
    }


def _has_ambiguity_markers(text: str) -> bool:
    return bool(AMBIGUITY_MARKER_RE.search(text))


def _find_exact_truth_span(answer_text: str, truth_text: str) -> Dict[str, Any]:
    if not truth_text:
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    truth_pattern = re.compile(rf"(?<![a-z0-9]){re.escape(truth_text)}(?![a-z0-9])")
    if not truth_pattern.search(answer_text):
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    if _has_ambiguity_markers(answer_text):
        return {
            "matched": False,
            "manual_check": True,
            "reason": "ambiguous_embedded_entity_match",
        }

    return {
        "matched": True,
        "manual_check": True,
        "reason": "matched_embedded_entity_span",
    }


def _find_person_name_variant_span(answer_text: str, truth_text: str) -> Dict[str, Any]:
    answer_tokens = _tokenize_person_name(answer_text)
    truth_tokens = _tokenize_person_name(truth_text)

    if len(answer_tokens) < 2 or len(truth_tokens) < 2:
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    candidate_windows = []
    min_window = 2
    max_window = min(len(answer_tokens), len(truth_tokens) + 1)

    for window_size in range(min_window, max_window + 1):
        for start in range(len(answer_tokens) - window_size + 1):
            window_tokens = answer_tokens[start:start + window_size]
            candidate_windows.append(" ".join(window_tokens))

    matches = [
        candidate_window
        for candidate_window in candidate_windows
        if _match_person_name_variant(candidate_window, truth_text)["matched"]
    ]

    if not matches:
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    max_match_length = max(len(match.split()) for match in matches)
    strongest_matches = {match for match in matches if len(match.split()) == max_match_length}

    if len(strongest_matches) > 1 or _has_ambiguity_markers(answer_text):
        return {
            "matched": False,
            "manual_check": True,
            "reason": "ambiguous_embedded_person_name_match",
        }

    return {
        "matched": True,
        "manual_check": True,
        "reason": "matched_embedded_person_name_variant",
    }


def _extract_entity_candidate(text: str) -> Dict[str, Optional[str]]:
    canonical_text = _canonicalize_entity_text(text)
    if not canonical_text:
        return {
            "candidate": None,
            "manual_check": False,
            "reason": "invalid_entity_format",
        }

    stripped_prefix = LEADING_NOISE_RE.sub("", canonical_text)
    if stripped_prefix != canonical_text:
        candidate = _canonicalize_entity_text(stripped_prefix)
        return {
            "candidate": candidate or None,
            "manual_check": True,
            "reason": "parsed_entity_with_formatting_noise",
        }

    if MULTI_ENTITY_SEPARATOR_RE.search(canonical_text):
        return {
            "candidate": None,
            "manual_check": True,
            "reason": "ambiguous_multiple_entities",
        }

    return {
        "candidate": canonical_text,
        "manual_check": False,
        "reason": "parsed_clean_entity",
    }


def evaluate_entity(answer: Any, truth_value: str) -> Dict[str, Any]:
    """
    entity evaluator logic:
    - normalize answer
    - normalize ground truth
    - exact string match only
    """
    normalized_answer = normalize_text(answer)
    normalized_truth = _canonicalize_entity_text(normalize_text(truth_value))
    parsed_result = _extract_entity_candidate(normalized_answer)
    candidate = parsed_result["candidate"]
    manual_check = bool(parsed_result["manual_check"])
    is_person_like_truth = _is_person_like_truth(normalized_truth)

    if candidate is None:
        if is_person_like_truth and parsed_result["reason"] == "ambiguous_multiple_entities":
            person_match_result = _match_person_name_variant(normalized_answer, normalized_truth)
            if person_match_result["matched"]:
                return {
                    "is_correct": True,
                    "manual_check": True,
                    "normalized_answer": _canonicalize_entity_text(normalized_answer),
                    "normalized_truth": normalized_truth,
                    "reason": "matched",
                }

        return {
            "is_correct": False,
            "manual_check": manual_check or (
                is_person_like_truth and parsed_result["reason"] == "ambiguous_partial_person_name"
            ),
            "normalized_answer": normalized_answer,
            "normalized_truth": normalized_truth,
            "reason": str(parsed_result["reason"]),
        }

    is_correct = candidate == normalized_truth
    if not is_correct and is_person_like_truth:
        person_match_result = _match_person_name_variant(candidate, normalized_truth)
        if person_match_result["matched"]:
            return {
                "is_correct": True,
                "manual_check": True,
                "normalized_answer": candidate,
                "normalized_truth": normalized_truth,
                "reason": "matched",
            }

        if person_match_result["manual_check"]:
            return {
                "is_correct": False,
                "manual_check": True,
                "normalized_answer": candidate,
                "normalized_truth": normalized_truth,
                "reason": str(person_match_result["reason"]),
            }

    exact_span_result = _find_exact_truth_span(normalized_answer, normalized_truth)
    if exact_span_result["matched"]:
        return {
            "is_correct": True,
            "manual_check": True,
            "normalized_answer": normalized_answer,
            "normalized_truth": normalized_truth,
            "reason": "matched",
        }

    if exact_span_result["manual_check"]:
        return {
            "is_correct": False,
            "manual_check": True,
            "normalized_answer": normalized_answer,
            "normalized_truth": normalized_truth,
            "reason": str(exact_span_result["reason"]),
        }

    if is_person_like_truth:
        person_span_result = _find_person_name_variant_span(normalized_answer, normalized_truth)
        if person_span_result["matched"]:
            return {
                "is_correct": True,
                "manual_check": True,
                "normalized_answer": normalized_answer,
                "normalized_truth": normalized_truth,
                "reason": "matched",
            }

        if person_span_result["manual_check"]:
            return {
                "is_correct": False,
                "manual_check": True,
                "normalized_answer": normalized_answer,
                "normalized_truth": normalized_truth,
                "reason": str(person_span_result["reason"]),
            }

    should_manual_check = manual_check and (
        is_correct or str(parsed_result["reason"]) == "ambiguous_multiple_entities"
    )

    mismatch_reason = (
        "entity_not_matched"
        if parsed_result["reason"] in {"parsed_clean_entity", "parsed_entity_with_formatting_noise"}
        else str(parsed_result["reason"])
    )

    return {
        "is_correct": is_correct,
        "manual_check": should_manual_check,
        "normalized_answer": candidate,
        "normalized_truth": normalized_truth,
        "reason": "matched" if is_correct else mismatch_reason
    }
