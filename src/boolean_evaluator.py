import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: Any) -> str:
    """
    Basic normalization for boolean answers:
    - convert to string
    - remove leading/trailing spaces
    - lowercase everything
    """
    if text is None:
        return ""

    return str(text).strip().lower()


# Token-level patterns keep parsing stricter than a plain substring check, so
# words like "yesterday" or "notebook" are not accidentally treated as answers.
BOOLEAN_TOKEN_RE = re.compile(r"\b(yes|no)\b")
NEGATED_BOOLEAN_RE = re.compile(r"\b(?:not|never)\s+(yes|no)\b")
SAFE_BOOLEAN_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+answer|answer)\s*[:\-]\s*(yes|no)[\s\W]*$|"
    r"^(?:the\s+answer\s+is|it\s+is|it's)\s+(yes|no)[\s\W]*$"
)

# If a boolean token appears near hedging language, the parsed value is still
# useful but should be reviewed manually before being trusted.
HEDGING_WORDS = {
    "maybe",
    "probably",
    "perhaps",
    "possibly",
    "likely",
    "unlikely",
    "guess",
    "guessed",
    "guessing",
    "think",
    "believe",
    "assume",
    "seems",
    "appears",
}

# These words are treated as harmless formatting/context around a single yes/no
# token, for example "final answer: yes".
CONNECTOR_WORDS = {
    "answer",
    "final",
    "so",
    "is",
    "was",
    "would",
    "be",
    "the",
    "a",
    "an",
}


def _get_context_words(text: str, start: int, end: int) -> Tuple[List[str], List[str]]:
    """
    Return the nearest words around a matched boolean token.

    The parser uses this small context window to detect hedging or harmless
    connector words without interpreting the full answer text semantically.
    """
    left_text = text[:start]
    right_text = text[end:]

    left_words = re.findall(r"[a-z]+", left_text)
    right_words = re.findall(r"[a-z]+", right_text)

    return left_words[-3:], right_words[:3]


def _is_clean_boolean_response(text: str) -> bool:
    """
    Check whether the response is only a yes/no token plus punctuation/spacing.
    """
    return bool(re.fullmatch(r"[\s\W]*(yes|no)[\s\W]*", text))


def _parse_boolean_answer(text: str) -> Dict[str, Any]:
    """
    Parse a normalized answer into a boolean value and review metadata.

    Returns a dict with:
    - predicted_value: True for yes, False for no, or None if no safe value exists
    - manual_check: True when the answer was parseable but needs human review
    - reason: a stable machine-readable explanation for downstream reporting
    """
    # Empty or non-boolean answers cannot produce a prediction.
    if not text:
        return {
            "predicted_value": None,
            "manual_check": False,
            "reason": "invalid_boolean_format",
        }

    # "not yes" and similar phrases invert or complicate the literal token, so
    # avoid guessing and send the answer to manual review.
    if NEGATED_BOOLEAN_RE.search(text):
        return {
            "predicted_value": None,
            "manual_check": True,
            "reason": "ambiguous_negated_boolean",
        }

    # Accept common final-answer wrappers as safe automated parses.
    safe_prefix_match = SAFE_BOOLEAN_PREFIX_RE.fullmatch(text)
    if safe_prefix_match:
        token = safe_prefix_match.group(1) or safe_prefix_match.group(2)
        return {
            "predicted_value": token == "yes",
            "manual_check": False,
            "reason": "parsed_safe_prefixed_boolean",
        }

    # From here on, parsing is based on explicit yes/no token occurrences.
    matches = list(BOOLEAN_TOKEN_RE.finditer(text))
    if not matches:
        return {
            "predicted_value": None,
            "manual_check": False,
            "reason": "invalid_boolean_format",
        }

    # Mixed yes/no tokens usually indicate explanation, correction, or conflict.
    distinct_tokens = {match.group(1) for match in matches}
    if len(distinct_tokens) > 1:
        return {
            "predicted_value": None,
            "manual_check": True,
            "reason": "ambiguous_multiple_boolean_values",
        }

    token = matches[0].group(1)
    predicted_value = token == "yes"

    # Repeating the same token is parseable, but often signals noisy output.
    if len(matches) > 1:
        return {
            "predicted_value": predicted_value,
            "manual_check": True,
            "reason": "ambiguous_repeated_boolean",
        }

    match = matches[0]
    left_words, right_words = _get_context_words(text, match.start(), match.end())

    # Nearby hedging keeps the prediction but flags it for manual inspection.
    context_words = set(left_words + right_words)
    if context_words & HEDGING_WORDS:
        return {
            "predicted_value": predicted_value,
            "manual_check": True,
            "reason": "ambiguous_boolean_with_hedging",
        }

    clean_response = _is_clean_boolean_response(text)
    allowed_context = context_words <= CONNECTOR_WORDS

    # A standalone "yes" or "no" is the highest-confidence parse.
    if clean_response:
        return {
            "predicted_value": predicted_value,
            "manual_check": False,
            "reason": "parsed_clean_boolean",
        }

    # Harmless wrapper words still get a prediction, but the non-clean format is
    # surfaced for review so downstream metrics can choose whether to include it.
    if allowed_context:
        return {
            "predicted_value": predicted_value,
            "manual_check": True,
            "reason": "parsed_boolean_with_formatting_noise",
        }

    # Any other surrounding text may change the meaning, so keep the literal
    # token as the best available prediction while requiring manual review.
    return {
        "predicted_value": predicted_value,
        "manual_check": True,
        "reason": "ambiguous_boolean_with_extra_context",
    }


def evaluate_boolean(answer: Any, truth_value: bool) -> Dict[str, Any]:
    """
    boolean evaluator logic:
    - normalize answer
    - convert 'yes' -> True, 'no' -> False
    - compare predicted boolean with ground truth
    """
    normalized_answer = normalize_text(answer)
    parsed_result = _parse_boolean_answer(normalized_answer)
    predicted_value: Optional[bool] = parsed_result["predicted_value"]
    manual_check = parsed_result["manual_check"]

    # Invalid or unsafe parses are always marked incorrect automatically; the
    # manual_check flag tells reviewers whether a human decision is still needed.
    if predicted_value is None:
        return {
            "is_correct": False,
            "manual_check": manual_check,
            "normalized_answer": normalized_answer,
            "predicted_value": predicted_value,
            "truth_value": truth_value,
            "reason": parsed_result["reason"]
        }

    is_correct = predicted_value == truth_value

    # Preserve the parser's reason for incorrect answers, but collapse correct
    # answers to "matched" for simpler reporting.
    return {
        "is_correct": is_correct,
        "manual_check": manual_check,
        "normalized_answer": normalized_answer,
        "predicted_value": predicted_value,
        "truth_value": truth_value,
        "reason": "matched" if is_correct else parsed_result["reason"]
    }
