import re
import math
from typing import Any, Dict, List, Optional


NUMBER_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
SAFE_NUMBER_PREFIX_RE = re.compile(
    rf"^(?:the\s+)?(?:final\s+answer|answer)\s*[:\-]\s*({NUMBER_PATTERN})[\s\W]*$|"
    rf"^(?:the\s+answer\s+is|it\s+is|it's)\s+({NUMBER_PATTERN})[\s\W]*$"
)
HEDGED_NUMBER_RE = re.compile(
    rf"\b(?:about|approximately|approx|around|at least|more than)\b[^\d\-+]*({NUMBER_PATTERN})\b"
)


def normalize_text(text: Any) -> str:
    """
    Basic normalization for model answers.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numbers(text: Any) -> List[float]:
    """
    Extract all numbers from text.
    Example:
    'abc 45 yy 49 zz 59,39' -> [45.0, 49.0, 59.0, 39.0]
    """
    text = "" if text is None else str(text)
    matches = re.findall(NUMBER_PATTERN, text)

    if not matches:
        return []

    return [float(x) for x in matches]


def exact_or_approx_match(pred_num: float, truth_num: float, tol: float = 1e-9) -> bool:
    """
    Exact match first; otherwise allow a tiny tolerance.
    """
    if pred_num == truth_num:
        return True

    return math.isclose(pred_num, truth_num, rel_tol=tol, abs_tol=tol)


def _is_clean_single_number(text: str) -> bool:
    return bool(re.fullmatch(rf"[\s\W]*{NUMBER_PATTERN}[\s\W]*", text))


def _parse_single_number_answer(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "predicted_numbers": [],
            "manual_check": False,
            "reason": "no_number_found",
        }

    safe_prefix_match = SAFE_NUMBER_PREFIX_RE.fullmatch(text)
    if safe_prefix_match:
        token = safe_prefix_match.group(1) or safe_prefix_match.group(2)
        return {
            "predicted_numbers": [float(token)],
            "manual_check": False,
            "reason": "parsed_safe_prefixed_number",
        }

    hedged_match = HEDGED_NUMBER_RE.search(text)
    if hedged_match:
        return {
            "predicted_numbers": [float(hedged_match.group(1))],
            "manual_check": True,
            "reason": "ambiguous_hedged_number",
        }

    predicted_numbers = extract_numbers(text)
    if not predicted_numbers:
        return {
            "predicted_numbers": [],
            "manual_check": False,
            "reason": "no_number_found",
        }

    if len(predicted_numbers) > 1:
        return {
            "predicted_numbers": predicted_numbers,
            "manual_check": True,
            "reason": "ambiguous_multiple_numbers",
        }

    if _is_clean_single_number(text):
        return {
            "predicted_numbers": predicted_numbers,
            "manual_check": False,
            "reason": "parsed_clean_number",
        }

    return {
        "predicted_numbers": predicted_numbers,
        "manual_check": False,
        "reason": "parsed_number_with_extra_context",
    }


def evaluate_single_number(answer: Any, truth_value: float) -> Dict[str, Any]:
    """
    single_number evaluator logic:
    - normalize answer
    - parse clean, noisy, or ambiguous numeric output
    - if no number -> incorrect
    - if multiple numbers are present -> incorrect and manual_check = True
    - if a single extracted number matches truth -> correct
    """
    normalized_answer = normalize_text(answer)
    parsed_result = _parse_single_number_answer(normalized_answer)
    predicted_numbers = parsed_result["predicted_numbers"]
    manual_check = parsed_result["manual_check"]

    if not predicted_numbers:
        return {
            "is_correct": False,
            "manual_check": manual_check,
            "normalized_answer": normalized_answer,
            "predicted_numbers": predicted_numbers,
            "reason": parsed_result["reason"]
        }

    if len(predicted_numbers) > 1:
        matched = any(exact_or_approx_match(num, truth_value) for num in predicted_numbers)
        return {
            "is_correct": False,
            "manual_check": matched,
            "normalized_answer": normalized_answer,
            "predicted_numbers": predicted_numbers,
            "reason": "ambiguous_multiple_numbers"
        }

    if parsed_result["reason"] == "ambiguous_hedged_number":
        return {
            "is_correct": False,
            "manual_check": True,
            "normalized_answer": normalized_answer,
            "predicted_numbers": predicted_numbers,
            "reason": "ambiguous_hedged_number"
        }

    matched = exact_or_approx_match(predicted_numbers[0], truth_value)

    return {
        "is_correct": matched,
        "manual_check": manual_check,
        "normalized_answer": normalized_answer,
        "predicted_numbers": predicted_numbers,
        "reason": "matched" if matched else "truth_not_found_in_predicted_numbers"
    }
