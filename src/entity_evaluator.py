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


# Common answer wrappers are stripped before entity comparison, but they still
# trigger manual_check because the answer was not a clean entity-only response.
LEADING_NOISE_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+answer|answer)\s*[:\-]\s*|^(?:the\s+answer\s+is|it\s+is|it's)\s+"
)

# Separators usually mean the response contains multiple candidate entities.
MULTI_ENTITY_SEPARATOR_RE = re.compile(r",|;|\band\b|\bor\b|\s+/\s+")

# These markers make embedded matches unsafe to score automatically.
AMBIGUITY_MARKER_RE = re.compile(r"\b(?:not|never|maybe|perhaps|probably|possibly|or)\b|,|;|\s+/\s+")


def _canonicalize_entity_text(text: str) -> str:
    """
    Normalize spacing and trim surrounding punctuation from an entity string.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s'\"(\[]+", "", text)
    text = re.sub(r"[\s'\")\].,!?;:]+$", "", text)
    return text


def _tokenize_person_name(text: str) -> list[str]:
    """
    Tokenize person names into lowercase alphabetic parts for flexible matching.
    """
    return re.findall(r"[a-z]+", text)


def _is_person_like_truth(text: str) -> bool:
    """
    Treat multi-token truth values as possible person names.

    Person-like truths get extra matching for initials, reordered noise, and
    embedded name spans, while single-token entities keep stricter matching.
    """
    tokens = _tokenize_person_name(text)
    return len(tokens) >= 2 and " " in text


def _can_match_given_tokens(candidate_tokens: list[str], truth_tokens: list[str]) -> bool:
    """
    Check whether candidate given-name tokens match the truth tokens.

    A candidate token may be either the full token or a single-letter initial,
    and each truth token can only be consumed once.
    """
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
    """
    Match person-name variants that share the surname and compatible given names.

    This handles cases like full names with initials, while surname-only matches
    remain ambiguous and require manual review.
    """
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
    """
    Detect markers that make an otherwise found entity span ambiguous.
    """
    return bool(AMBIGUITY_MARKER_RE.search(text))


def _find_exact_truth_span(answer_text: str, truth_text: str) -> Dict[str, Any]:
    """
    Look for the full truth entity inside a longer answer.

    Embedded exact matches are marked for manual review because surrounding text
    can still negate, qualify, or list multiple entities.
    """
    if not truth_text:
        return {
            "matched": False,
            "manual_check": False,
            "reason": "entity_not_matched",
        }

    # Boundary checks avoid matching the truth inside a larger alphanumeric word.
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
    """
    Search longer answers for a person-name variant span.

    Candidate windows are limited around the truth-name length so the matcher can
    catch initials or small variants without accepting arbitrary long text.
    """
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

    # Build contiguous token windows that could represent the person name.
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

    # If more than one strongest span matches, the answer may contain multiple
    # people or competing interpretations.
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
    """
    Extract the best single entity candidate from a normalized answer.

    Returns candidate=None for invalid or multi-entity answers, while preserving
    manual_check and reason metadata for downstream scoring.
    """
    canonical_text = _canonicalize_entity_text(text)
    if not canonical_text:
        return {
            "candidate": None,
            "manual_check": False,
            "reason": "invalid_entity_format",
        }

    # Strip harmless leading answer labels but keep manual_check enabled.
    stripped_prefix = LEADING_NOISE_RE.sub("", canonical_text)
    if stripped_prefix != canonical_text:
        candidate = _canonicalize_entity_text(stripped_prefix)
        return {
            "candidate": candidate or None,
            "manual_check": True,
            "reason": "parsed_entity_with_formatting_noise",
        }

    # Multiple entities are intentionally not reduced to the first item.
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

    # If no single clean candidate exists, only person-name variants get a
    # limited fallback path; other entity types remain invalid/ambiguous.
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

    # Prefer exact canonical entity matches before trying fuzzy person-name paths.
    is_correct = candidate == normalized_truth
    should_manual_check = manual_check and (
        is_correct or str(parsed_result["reason"]) == "ambiguous_multiple_entities"
    )

    if is_correct:
        return {
            "is_correct": True,
            "manual_check": should_manual_check,
            "normalized_answer": candidate,
            "normalized_truth": normalized_truth,
            "reason": "matched",
        }

    # Person names can be valid even when the canonical strings differ, such as
    # "j k rowling" matching "joanne rowling".
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

    # Longer answers can still be correct if they contain the exact truth span,
    # but they are always routed through manual_check because context matters.
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
        # Final fallback for longer answers containing person-name variants.
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

    # Clean parse failures should report a semantic mismatch instead of exposing
    # parser-stage labels as final mismatch reasons.
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
