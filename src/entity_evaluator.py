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


def _canonicalize_entity_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s'\"“”‘’(\[]+", "", text)
    text = re.sub(r"[\s'\"“”‘’)\].,!?;:]+$", "", text)
    return text


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

    if candidate is None:
        return {
            "is_correct": False,
            "manual_check": manual_check,
            "normalized_answer": normalized_answer,
            "normalized_truth": normalized_truth,
            "reason": str(parsed_result["reason"]),
        }

    is_correct = candidate == normalized_truth

    mismatch_reason = (
        "entity_not_matched"
        if parsed_result["reason"] == "parsed_clean_entity"
        else str(parsed_result["reason"])
    )

    return {
        "is_correct": is_correct,
        "manual_check": manual_check,
        "normalized_answer": candidate,
        "normalized_truth": normalized_truth,
        "reason": "matched" if is_correct else mismatch_reason
    }
