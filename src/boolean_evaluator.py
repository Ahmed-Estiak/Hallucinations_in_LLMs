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


BOOLEAN_TOKEN_RE = re.compile(r"\b(yes|no)\b")
NEGATED_BOOLEAN_RE = re.compile(r"\b(?:not|never)\s+(yes|no)\b")
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
    left_text = text[:start]
    right_text = text[end:]

    left_words = re.findall(r"[a-z]+", left_text)
    right_words = re.findall(r"[a-z]+", right_text)

    return left_words[-3:], right_words[:3]


def _is_clean_boolean_response(text: str) -> bool:
    return bool(re.fullmatch(r"[\s\W]*(yes|no)[\s\W]*", text))


def _parse_boolean_answer(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "predicted_value": None,
            "manual_check": False,
            "reason": "invalid_boolean_format",
        }

    if NEGATED_BOOLEAN_RE.search(text):
        return {
            "predicted_value": None,
            "manual_check": True,
            "reason": "ambiguous_negated_boolean",
        }

    matches = list(BOOLEAN_TOKEN_RE.finditer(text))
    if not matches:
        return {
            "predicted_value": None,
            "manual_check": False,
            "reason": "invalid_boolean_format",
        }

    distinct_tokens = {match.group(1) for match in matches}
    if len(distinct_tokens) > 1:
        return {
            "predicted_value": None,
            "manual_check": True,
            "reason": "ambiguous_multiple_boolean_values",
        }

    token = matches[0].group(1)
    predicted_value = token == "yes"

    if len(matches) > 1:
        return {
            "predicted_value": predicted_value,
            "manual_check": True,
            "reason": "ambiguous_repeated_boolean",
        }

    match = matches[0]
    left_words, right_words = _get_context_words(text, match.start(), match.end())

    context_words = set(left_words + right_words)
    if context_words & HEDGING_WORDS:
        return {
            "predicted_value": predicted_value,
            "manual_check": True,
            "reason": "ambiguous_boolean_with_hedging",
        }

    clean_response = _is_clean_boolean_response(text)
    allowed_context = context_words <= CONNECTOR_WORDS

    if clean_response:
        return {
            "predicted_value": predicted_value,
            "manual_check": False,
            "reason": "parsed_clean_boolean",
        }

    if allowed_context:
        return {
            "predicted_value": predicted_value,
            "manual_check": True,
            "reason": "parsed_boolean_with_formatting_noise",
        }

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

    return {
        "is_correct": is_correct,
        "manual_check": manual_check,
        "normalized_answer": normalized_answer,
        "predicted_value": predicted_value,
        "truth_value": truth_value,
        "reason": "matched" if is_correct else parsed_result["reason"]
    }
